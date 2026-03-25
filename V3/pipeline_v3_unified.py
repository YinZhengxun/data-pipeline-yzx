'''#!/usr/bin/env python3
"""
Unified V3 pipeline:
1) Extract WAV
2) Split WAV into overlapping chunks
3) Run ASR+diarization(+alignment) for each chunk
4) Clean and combine chunk WhisperX JSON outputs
5) Run BERTopic over combined transcript
6) Run Flair NER over chunked topic texts
7) Merge topic+NER annotations back into whisper-style JSON
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from clean_and_combine_whisperx_overlap import clean_one_file, combine_files
from split_wav_chunks import split_wav


SCRIPT_DIR = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[V3] {msg}")


def run_cmd(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log("CMD: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR), env=env)


def ensure_wav(input_path: Path, wav_dir: Path, resume: bool) -> Path:
    wav_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".wav":
        return input_path.resolve()

    output_wav = wav_dir / f"{input_path.stem}.wav"
    if resume and output_wav.exists():
        log(f"Reuse existing WAV: {output_wav}")
        return output_wav

    log(f"Extract WAV from: {input_path}")
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(str(output_wav), format="wav")
    log(f"WAV saved: {output_wav}")
    return output_wav


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Invalid or empty chunk manifest: {manifest_path}")

    manifest.sort(key=lambda x: int(x.get("chunk_index", 0)))
    return manifest


def resolve_chunk_path(record: dict[str, Any], chunks_dir: Path) -> Path:
    chunk_path = Path(record.get("chunk_path", ""))
    if chunk_path.exists():
        return chunk_path

    chunk_filename = record.get("chunk_filename")
    if not chunk_filename:
        raise FileNotFoundError(f"Chunk record missing chunk_filename: {record}")

    fallback = chunks_dir / chunk_filename
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Chunk file not found for record: {record}")


def run_chunk_asr(
    manifest: list[dict[str, Any]],
    chunks_dir: Path,
    chunk_runs_dir: Path,
    raw_json_dir: Path,
    hf_token: str | None,
    language: str | None,
    num_speakers: int | None,
    asr_backend: str,
    asr_model: str,
    gpu_index: int,
    skip_alignment: bool,
    resume: bool,
    continue_on_chunk_error: bool,
    max_chunks: int | None,
) -> tuple[int, list[str]]:
    chunk_runs_dir.mkdir(parents=True, exist_ok=True)
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed: list[str] = []

    total = len(manifest) if max_chunks is None else min(len(manifest), max_chunks)

    for idx, record in enumerate(manifest[:total], start=1):
        chunk_wav = resolve_chunk_path(record, chunks_dir)
        chunk_base = chunk_wav.stem
        target_json = raw_json_dir / f"{chunk_base}_whisperx.json"

        if resume and target_json.exists():
            log(f"[{idx}/{total}] Skip ASR (exists): {target_json.name}")
            processed += 1
            continue

        log(f"[{idx}/{total}] ASR chunk: {chunk_wav.name}")

        out_dir = chunk_runs_dir / f"output_{chunk_base}"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_single_file_pipeline_AG.py"),
            "--episode_path",
            str(chunk_wav),
            "--out_dir",
            str(out_dir),
            "--gpu_index",
            str(gpu_index),
            "--asr_backend",
            asr_backend,
            "--asr_model",
            asr_model,
        ]

        if language:
            cmd.extend(["--language", language])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])
        if num_speakers is not None:
            cmd.extend(["--num_speakers", str(num_speakers)])
        if skip_alignment:
            cmd.append("--skip_alignment")

        try:
            run_cmd(cmd, extra_env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})

            generated_json = out_dir / "json" / f"{chunk_base}_whisperx.json"
            if not generated_json.exists():
                candidates = sorted((out_dir / "json").glob("*_whisperx.json"))
                if len(candidates) == 1:
                    generated_json = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"Cannot find chunk whisper JSON in {(out_dir / 'json')}"
                    )

            shutil.copy2(generated_json, target_json)
            processed += 1

        except Exception as exc:  # noqa: BLE001
            failed.append(chunk_wav.name)
            log(f"Chunk failed: {chunk_wav.name} -> {exc}")
            if not continue_on_chunk_error:
                raise

    return processed, failed


def clean_and_combine(raw_json_dir: Path, clean_dir: Path, combined_json: Path, overlap_sec: float) -> int:
    raw_files = sorted(glob.glob(str(raw_json_dir / "chunk_*_whisperx.json")))
    if not raw_files:
        raise FileNotFoundError(f"No chunk json found in: {raw_json_dir}")

    clean_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files: list[str] = []

    for src in raw_files:
        cleaned_path, _meta = clean_one_file(src, str(clean_dir))
        cleaned_files.append(cleaned_path)

    combined_json.parent.mkdir(parents=True, exist_ok=True)
    combine_files(cleaned_files, str(combined_json), overlap_sec)
    return len(raw_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3 one-click pipeline: wav -> chunks -> ASR -> clean/combine -> topics -> NER -> final JSON"
    )
    parser.add_argument("--episode_path", required=True, help="Input video/audio path")
    parser.add_argument("--work_dir", default="output_v3_pipeline", help="Pipeline output directory")

    parser.add_argument("--chunk_sec", type=int, default=600, help="Chunk length in seconds")
    parser.add_argument("--overlap_sec", type=float, default=15.0, help="Chunk overlap in seconds")
    parser.add_argument("--max_chunks", type=int, default=None, help="Debug: only process first N chunks")

    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--language", default=None, help="ASR/alignment language code, e.g. de")
    parser.add_argument("--num_speakers", type=int, default=None, help="Fixed number of speakers")
    parser.add_argument(
        "--asr_backend",
        default="transformers",
        choices=["transformers", "openai"],
        help="ASR backend used in chunk transcription",
    )
    parser.add_argument(
        "--asr_model",
        default="openai/whisper-small",
        help="ASR model name (smaller model helps memory)",
    )
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip forced alignment in ASR")

    parser.add_argument("--topic_chunk_size", type=int, default=8, help="Segments per topic chunk")
    parser.add_argument(
        "--topic_embedding_model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for BERTopic",
    )
    parser.add_argument(
        "--ner_model",
        default="flair/ner-german-large",
        help="Flair NER model",
    )

    parser.add_argument("--skip_topics", action="store_true", help="Skip BERTopic step")
    parser.add_argument("--skip_ner", action="store_true", help="Skip NER step")

    parser.add_argument(
        "--continue_on_chunk_error",
        action="store_true",
        help="Continue if individual chunks fail",
    )
    parser.add_argument("--no_resume", action="store_true", help="Re-run all steps even if outputs exist")

    args = parser.parse_args()

    if args.skip_topics and not args.skip_ner:
        raise ValueError("--skip_topics cannot be used without --skip_ner")

    resume = not args.no_resume

    input_path = Path(args.episode_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_name = input_path.stem
    work_dir = Path(args.work_dir).resolve()

    wav_dir = work_dir / "wav"
    chunks_dir = work_dir / "chunks"
    chunk_runs_dir = work_dir / "chunk_runs"
    raw_json_dir = work_dir / "chunk_json_raw"
    clean_dir = work_dir / "chunk_json_clean"
    combined_dir = work_dir / "combined"
    topics_dir = work_dir / "topics"
    final_dir = work_dir / "final"

    manifest_path = chunks_dir / "chunks_manifest.json"
    combined_json = combined_dir / f"{base_name}_combined_whisperx.json"
    topics_json = topics_dir / "chunked_transcript_with_topics.json"
    ner_json = topics_dir / "chunked_transcript_with_topics_ner.json"
    final_json = final_dir / f"{base_name}_final_whisperx_topic_ner.json"
    summary_json = work_dir / "pipeline_summary.json"

    start_ts = time.time()

    log("Step 1/7: Extract WAV")
    wav_path = ensure_wav(input_path, wav_dir, resume=resume)

    log("Step 2/7: Split WAV into chunks")
    if resume and manifest_path.exists():
        log(f"Reuse existing manifest: {manifest_path}")
    else:
        split_wav(
            input_wav=str(wav_path),
            output_dir=str(chunks_dir),
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
        )

    manifest = load_manifest(manifest_path)
    log(f"Chunks in manifest: {len(manifest)}")

    log("Step 3/7: Chunk ASR transcription")
    processed_chunks, failed_chunks = run_chunk_asr(
        manifest=manifest,
        chunks_dir=chunks_dir,
        chunk_runs_dir=chunk_runs_dir,
        raw_json_dir=raw_json_dir,
        hf_token=args.hf_token,
        language=args.language,
        num_speakers=args.num_speakers,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        gpu_index=args.gpu_index,
        skip_alignment=args.skip_alignment,
        resume=resume,
        continue_on_chunk_error=args.continue_on_chunk_error,
        max_chunks=args.max_chunks,
    )
    if failed_chunks:
        log(f"Failed chunks: {len(failed_chunks)}")

    log("Step 4/7: Clean and combine chunk JSON")
    combined_count = clean_and_combine(
        raw_json_dir=raw_json_dir,
        clean_dir=clean_dir,
        combined_json=combined_json,
        overlap_sec=args.overlap_sec,
    )
    log(f"Combined {combined_count} chunk JSON files")

    if not args.skip_topics:
        log("Step 5/7: BERTopic")
        if not (resume and topics_json.exists()):
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "bertopic_from_combined_json.py"),
                    "--input_json",
                    str(combined_json),
                    "--output_dir",
                    str(topics_dir),
                    "--chunk_size",
                    str(args.topic_chunk_size),
                    "--embedding_model",
                    args.topic_embedding_model,
                ]
            )
        else:
            log(f"Skip BERTopic (exists): {topics_json}")

        if not args.skip_ner:
            log("Step 6/7: Flair NER")
            if not (resume and ner_json.exists()):
                run_cmd(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "flair_ner_on_chunked_json.py"),
                        "--input_json",
                        str(topics_json),
                        "--output_json",
                        str(ner_json),
                        "--model",
                        args.ner_model,
                    ]
                )
            else:
                log(f"Skip NER (exists): {ner_json}")

    log("Step 7/7: Merge annotations back to whisper JSON")
    final_dir.mkdir(parents=True, exist_ok=True)
    chunked_for_merge = ner_json if (not args.skip_ner and ner_json.exists()) else topics_json

    if args.skip_topics:
        raise RuntimeError("Cannot run merge because topics were skipped.")

    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_chunk_topics_ner_back_to_whisper_json.py"),
            "--whisper_json",
            str(combined_json),
            "--chunked_json",
            str(chunked_for_merge),
            "--output_json",
            str(final_json),
            "--chunk_size",
            str(args.topic_chunk_size),
        ]
    )

    elapsed = round(time.time() - start_ts, 2)

    summary = {
        "input": str(input_path),
        "work_dir": str(work_dir),
        "wav": str(wav_path),
        "manifest": str(manifest_path),
        "processed_chunks": processed_chunks,
        "failed_chunks": failed_chunks,
        "combined_json": str(combined_json),
        "topics_json": str(topics_json) if topics_json.exists() else None,
        "ner_json": str(ner_json) if ner_json.exists() else None,
        "final_json": str(final_json),
        "elapsed_seconds": elapsed,
        "params": vars(args),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("Pipeline complete")
    log(f"Final JSON: {final_json}")
    log(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()'''

#!/usr/bin/env python3
"""
Unified V3 pipeline:
1) Extract/normalize WAV
2) Split WAV into overlapping chunks
3) Run ASR+diarization(+alignment) for each chunk
4) Clean and combine chunk WhisperX JSON outputs
5) Run BERTopic over combined transcript
6) Run Flair NER over chunked topic texts
7) Merge topic+NER annotations back into whisper-style JSON
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from clean_and_combine_whisperx_overlap import clean_one_file, combine_files
from split_wav_chunks import split_wav


SCRIPT_DIR = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[V3] {msg}")


def run_cmd(
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log("CMD: " + " ".join(cmd))
    subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd or SCRIPT_DIR),
        env=env,
    )


def ensure_python_executable(python_path: str | None, fallback: str = sys.executable) -> str:
    if python_path is None:
        return fallback

    p = Path(python_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Python executable not found: {p}")
    if not os.access(str(p), os.X_OK):
        raise PermissionError(f"Python executable is not executable: {p}")
    return str(p)


def ensure_wav(input_path: Path, wav_dir: Path, resume: bool) -> Path:
    """
    Always create a normalized 16kHz mono WAV in wav_dir.
    This is more stable than directly reusing an arbitrary input WAV.
    """
    wav_dir.mkdir(parents=True, exist_ok=True)
    output_wav = wav_dir / f"{input_path.stem}.wav"

    if resume and output_wav.exists():
        log(f"Reuse existing normalized WAV: {output_wav}")
        return output_wav

    log(f"Normalize audio to 16kHz mono WAV from: {input_path}")
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(str(output_wav), format="wav")
    log(f"WAV saved: {output_wav}")
    return output_wav


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Invalid or empty chunk manifest: {manifest_path}")

    manifest.sort(key=lambda x: int(x.get("chunk_index", 0)))
    return manifest


def resolve_chunk_path(record: dict[str, Any], chunks_dir: Path) -> Path:
    chunk_path = Path(record.get("chunk_path", ""))
    if chunk_path.exists():
        return chunk_path

    chunk_filename = record.get("chunk_filename")
    if not chunk_filename:
        raise FileNotFoundError(f"Chunk record missing chunk_filename: {record}")

    fallback = chunks_dir / chunk_filename
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Chunk file not found for record: {record}")


def run_chunk_asr(
    manifest: list[dict[str, Any]],
    chunks_dir: Path,
    chunk_runs_dir: Path,
    raw_json_dir: Path,
    hf_token: str | None,
    language: str | None,
    num_speakers: int | None,
    asr_backend: str,
    asr_model: str,
    gpu_index: int,
    skip_alignment: bool,
    resume: bool,
    continue_on_chunk_error: bool,
    max_chunks: int | None,
    asr_python: str,
) -> tuple[int, list[str]]:
    chunk_runs_dir.mkdir(parents=True, exist_ok=True)
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed: list[str] = []

    total = len(manifest) if max_chunks is None else min(len(manifest), max_chunks)

    for idx, record in enumerate(manifest[:total], start=1):
        chunk_wav = resolve_chunk_path(record, chunks_dir)
        chunk_base = chunk_wav.stem
        target_json = raw_json_dir / f"{chunk_base}_whisperx.json"

        if resume and target_json.exists():
            log(f"[{idx}/{total}] Skip ASR (exists): {target_json.name}")
            processed += 1
            continue

        log(f"[{idx}/{total}] ASR chunk: {chunk_wav.name}")

        out_dir = chunk_runs_dir / f"output_{chunk_base}"
        cmd = [
            asr_python,
            str(SCRIPT_DIR / "process_single_file_pipeline_AG.py"),
            "--episode_path",
            str(chunk_wav),
            "--out_dir",
            str(out_dir),
            "--gpu_index",
            str(gpu_index),
            "--asr_backend",
            asr_backend,
            "--asr_model",
            asr_model,
        ]

        if language:
            cmd.extend(["--language", language])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])
        if num_speakers is not None:
            cmd.extend(["--num_speakers", str(num_speakers)])
        if skip_alignment:
            cmd.append("--skip_alignment")

        try:
            run_cmd(cmd, extra_env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})

            json_dir = out_dir / "json"
            generated_json = json_dir / f"{chunk_base}_whisperx.json"

            if not generated_json.exists():
                candidates = sorted(json_dir.glob("*_whisperx.json"))
                if len(candidates) == 1:
                    generated_json = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"Cannot find chunk whisper JSON in {json_dir}"
                    )

            shutil.copy2(generated_json, target_json)
            processed += 1

        except Exception as exc:  # noqa: BLE001
            failed.append(chunk_wav.name)
            log(f"Chunk failed: {chunk_wav.name} -> {exc}")
            if not continue_on_chunk_error:
                raise

    return processed, failed


def clean_and_combine(
    raw_json_dir: Path,
    clean_dir: Path,
    combined_json: Path,
    overlap_sec: float,
) -> int:
    raw_files = sorted(glob.glob(str(raw_json_dir / "chunk_*_whisperx.json")))
    if not raw_files:
        raise FileNotFoundError(f"No chunk json found in: {raw_json_dir}")

    clean_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files: list[str] = []

    for src in raw_files:
        cleaned_path, _meta = clean_one_file(src, str(clean_dir))
        cleaned_files.append(cleaned_path)

    combined_json.parent.mkdir(parents=True, exist_ok=True)
    combine_files(cleaned_files, str(combined_json), overlap_sec)
    return len(raw_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3 one-click pipeline: wav -> chunks -> ASR -> clean/combine -> topics -> NER -> final JSON"
    )
    parser.add_argument("--episode_path", required=True, help="Input video/audio path")
    parser.add_argument("--work_dir", default="output_v3_pipeline", help="Pipeline output directory")

    parser.add_argument("--chunk_sec", type=int, default=600, help="Chunk length in seconds")
    parser.add_argument("--overlap_sec", type=float, default=15.0, help="Chunk overlap in seconds")
    parser.add_argument("--max_chunks", type=int, default=None, help="Debug: only process first N chunks")

    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--language", default=None, help="ASR/alignment language code, e.g. de")
    parser.add_argument("--num_speakers", type=int, default=None, help="Fixed number of speakers")
    parser.add_argument(
        "--asr_backend",
        default="transformers",
        choices=["transformers", "openai"],
        help="ASR backend used in chunk transcription",
    )
    parser.add_argument(
        "--asr_model",
        default="openai/whisper-small",
        help="ASR model name (smaller model helps memory)",
    )
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index for ASR")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip forced alignment in ASR")

    parser.add_argument("--topic_chunk_size", type=int, default=8, help="Segments per topic chunk")
    parser.add_argument(
        "--topic_embedding_model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for BERTopic",
    )
    parser.add_argument(
        "--ner_model",
        default="flair/ner-german-large",
        help="Flair NER model",
    )

    parser.add_argument(
        "--skip_topics",
        action="store_true",
        help="Skip BERTopic, NER, and merge; only produce combined WhisperX JSON",
    )
    parser.add_argument(
        "--skip_ner",
        action="store_true",
        help="Skip NER but still run BERTopic and merge topic annotations back",
    )

    parser.add_argument(
        "--continue_on_chunk_error",
        action="store_true",
        help="Continue if individual chunks fail",
    )
    parser.add_argument("--no_resume", action="store_true", help="Re-run all steps even if outputs exist")

    parser.add_argument(
        "--asr_python",
        default="/home/zhyin/cineminds/data-env/bin/python",
        help="Python executable for ASR/chunk transcription environment",
    )
    parser.add_argument(
        "--nlp_python",
        default="/home/zhyin/cineminds/topic-env/bin/python",
        help="Python executable for BERTopic/Flair/merge environment",
    )

    args = parser.parse_args()

    resume = not args.no_resume

    input_path = Path(args.episode_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    asr_python = ensure_python_executable(args.asr_python)
    nlp_python = ensure_python_executable(args.nlp_python)

    base_name = input_path.stem
    work_dir = Path(args.work_dir).resolve()

    wav_dir = work_dir / "wav"
    chunks_dir = work_dir / "chunks"
    chunk_runs_dir = work_dir / "chunk_runs"
    raw_json_dir = work_dir / "chunk_json_raw"
    clean_dir = work_dir / "chunk_json_clean"
    combined_dir = work_dir / "combined"
    topics_dir = work_dir / "topics"
    final_dir = work_dir / "final"

    manifest_path = chunks_dir / "chunks_manifest.json"
    combined_json = combined_dir / f"{base_name}_combined_whisperx.json"
    topics_json = topics_dir / "chunked_transcript_with_topics.json"
    ner_json = topics_dir / "chunked_transcript_with_topics_and_ner.json"
    final_json = final_dir / f"{base_name}_final_whisperx_topic_ner.json"
    summary_json = work_dir / "pipeline_summary.json"

    start_ts = time.time()

    log(f"ASR python: {asr_python}")
    log(f"NLP python: {nlp_python}")

    log("Step 1/7: Extract WAV")
    wav_path = ensure_wav(input_path, wav_dir, resume=resume)

    log("Step 2/7: Split WAV into chunks")
    if resume and manifest_path.exists():
        log(f"Reuse existing manifest: {manifest_path}")
    else:
        split_wav(
            input_wav=str(wav_path),
            output_dir=str(chunks_dir),
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
        )

    manifest = load_manifest(manifest_path)
    log(f"Chunks in manifest: {len(manifest)}")

    log("Step 3/7: Chunk ASR transcription")
    processed_chunks, failed_chunks = run_chunk_asr(
        manifest=manifest,
        chunks_dir=chunks_dir,
        chunk_runs_dir=chunk_runs_dir,
        raw_json_dir=raw_json_dir,
        hf_token=args.hf_token,
        language=args.language,
        num_speakers=args.num_speakers,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        gpu_index=args.gpu_index,
        skip_alignment=args.skip_alignment,
        resume=resume,
        continue_on_chunk_error=args.continue_on_chunk_error,
        max_chunks=args.max_chunks,
        asr_python=asr_python,
    )
    if failed_chunks:
        log(f"Failed chunks: {len(failed_chunks)}")

    log("Step 4/7: Clean and combine chunk JSON")
    combined_count = clean_and_combine(
        raw_json_dir=raw_json_dir,
        clean_dir=clean_dir,
        combined_json=combined_json,
        overlap_sec=args.overlap_sec,
    )
    log(f"Combined {combined_count} chunk JSON files")

    if args.skip_topics:
        elapsed = round(time.time() - start_ts, 2)
        summary = {
            "input": str(input_path),
            "work_dir": str(work_dir),
            "wav": str(wav_path),
            "manifest": str(manifest_path),
            "processed_chunks": processed_chunks,
            "failed_chunks": failed_chunks,
            "combined_json": str(combined_json),
            "topics_json": None,
            "ner_json": None,
            "final_json": None,
            "elapsed_seconds": elapsed,
            "params": vars(args),
        }
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        log("Pipeline complete (stopped after combined JSON because --skip_topics was set)")
        log(f"Combined JSON: {combined_json}")
        log(f"Summary: {summary_json}")
        return

    log("Step 5/7: BERTopic")
    if not (resume and topics_json.exists()):
        run_cmd(
            [
                nlp_python,
                str(SCRIPT_DIR / "bertopic_from_combined_json.py"),
                "--input_json",
                str(combined_json),
                "--output_dir",
                str(topics_dir),
                "--chunk_size",
                str(args.topic_chunk_size),
                "--embedding_model",
                args.topic_embedding_model,
            ]
        )
    else:
        log(f"Skip BERTopic (exists): {topics_json}")

    if args.skip_ner:
        log("Step 6/7: Skip NER (--skip_ner)")
    else:
        log("Step 6/7: Flair NER")
        if not (resume and ner_json.exists()):
            run_cmd(
                [
                    nlp_python,
                    str(SCRIPT_DIR / "flair_ner_on_chunked_json.py"),
                    "--input_json",
                    str(topics_json),
                    "--output_json",
                    str(ner_json),
                    "--model",
                    args.ner_model,
                ]
            )
        else:
            log(f"Skip NER (exists): {ner_json}")

    log("Step 7/7: Merge annotations back to whisper JSON")
    final_dir.mkdir(parents=True, exist_ok=True)

    chunked_for_merge = ner_json if (not args.skip_ner and ner_json.exists()) else topics_json
    if not chunked_for_merge.exists():
        raise FileNotFoundError(f"Chunked annotation JSON not found for merge: {chunked_for_merge}")

    if not combined_json.exists():
        raise FileNotFoundError(f"Combined whisper JSON not found: {combined_json}")

    run_cmd(
        [
            nlp_python,
            str(SCRIPT_DIR / "merge_chunk_topics_ner_back_to_whisper_json.py"),
            "--whisper_json",
            str(combined_json),
            "--chunked_json",
            str(chunked_for_merge),
            "--output_json",
            str(final_json),
            "--chunk_size",
            str(args.topic_chunk_size),
        ]
    )

    elapsed = round(time.time() - start_ts, 2)

    final_num_segments = None
    if final_json.exists():
        try:
            with final_json.open("r", encoding="utf-8") as f:
                final_data = json.load(f)
            final_num_segments = len(final_data.get("segments", []))
        except Exception:
            final_num_segments = None

    summary = {
        "input": str(input_path),
        "work_dir": str(work_dir),
        "wav": str(wav_path),
        "manifest": str(manifest_path),
        "num_manifest_chunks": len(manifest),
        "processed_chunks": processed_chunks,
        "failed_chunks": failed_chunks,
        "combined_json": str(combined_json),
        "topics_json": str(topics_json) if topics_json.exists() else None,
        "ner_json": str(ner_json) if ner_json.exists() else None,
        "final_json": str(final_json) if final_json.exists() else None,
        "final_num_segments": final_num_segments,
        "elapsed_seconds": elapsed,
        "params": vars(args),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("Pipeline complete")
    log(f"Final JSON: {final_json}")
    log(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()