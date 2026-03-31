#!/usr/bin/env python3
"""
V3 Unified Pipeline - One-Click Video/Audio Processing

Stages:
  1. Extract WAV (16kHz mono) from input media
  2. Split WAV into overlapping chunks (10min, 15s overlap)
  3. Run ASR + diarization + alignment on each chunk
  4. Clean hallucination/overlap artifacts from chunks
  5. Merge cleaned chunks into single transcript
  6. Run BERTopic topic modeling
  7. Run Flair NER on topic chunks
  8. Merge topic + NER annotations back to whisper-style JSON

Usage:
  python pipeline_v3_unified.py --episode_path /path/to/video.mp4 --work_dir /path/to/output
  python pipeline_v3_unified.py --episode_path /path/to/video.mp4 --work_dir /path/to/output --skip_topics
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
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# German stopwords (validated for topic modeling improvement)
# =============================================================================
GERMAN_STOPWORDS = [
    "ich", "du", "er", "sie", "es", "wir", "ihr", "man",
    "mich", "dich", "ihn", "ihm", "uns", "euch",
    "mein", "dein", "sein", "ihr", "unser", "euer",
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "einem", "einen", "eines",
    "diese", "dieser", "diesem", "diesen",
    "jeder", "jede", "jedes", "jeden", "jedem",
    "und", "oder", "aber", "doch", "denn", "sondern",
    "auch", "noch", "nur", "schon", "sehr", "ganz",
    "so", "wie", "als", "dann", "da", "dort", "hier",
    "mal", "halt", "eben", "eigentlich", "einfach", "wirklich",
    "quasi", "natürlich", "vielleicht", "irgendwie", "immer",
    "oft", "manchmal", "wieder", "weiter",
    "ist", "sind", "war", "waren", "bin", "bist", "seid",
    "hat", "haben", "hatte", "hatten", "gibt", "gab",
    "wird", "werden", "wurde", "wurden",
    "kann", "können", "konnte", "konnten",
    "muss", "müssen", "musste", "mussten",
    "will", "wollen", "wollte", "wollten",
    "zu", "mit", "von", "für", "auf", "in", "an", "bei",
    "über", "unter", "nach", "vor", "durch", "gegen", "ohne",
    "um", "aus", "bis", "am", "im", "ins", "beim", "vom",
    "nicht", "kein", "keine", "keiner", "keinem", "keinen",
    "ja", "nein", "okay", "ok", "also", "äh", "ähm", "hm",
    "dass", "ob", "wenn", "weil",
]


# =============================================================================
# Utility functions
# =============================================================================

def log(msg: str) -> None:
    print(f"[V3] {msg}")


def log_stage(stage: int, total: int, msg: str) -> None:
    print(f"[V3] Stage {stage}/{total}: {msg}")


def run_cmd(
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log(f"CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd or SCRIPT_DIR), env=env)


def ensure_python(python_path: str | None) -> str:
    """Validate and return a Python executable path.

    Important:
    Do NOT call .resolve() here, because venv python is often a symlink.
    Resolving it may collapse /path/to/venv/bin/python into /usr/bin/python3.x
    and lose the virtualenv context.
    """
    if python_path is None:
        return sys.executable

    p = Path(python_path).expanduser()

    if not p.exists():
        raise FileNotFoundError(f"Python executable not found: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Python path is not a file: {p}")
    if not os.access(str(p), os.X_OK):
        raise PermissionError(f"Python not executable: {p}")

    return str(p)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Stage 1: WAV extraction
# =============================================================================

def ensure_wav(input_path: Path, wav_dir: Path, resume: bool) -> Path:
    """Normalize any audio/video to 16kHz mono WAV."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    output_wav = wav_dir / f"{input_path.stem}.wav"

    if resume and output_wav.exists():
        log(f"  Reuse existing WAV: {output_wav}")
        return output_wav

    log(f"  Converting to 16kHz mono WAV: {input_path}")
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(str(output_wav), format="wav")
    log(f"  WAV saved: {output_wav}")
    return output_wav


# =============================================================================
# Stage 2: WAV chunking
# =============================================================================

def split_wav_chunks(
    wav_path: Path,
    chunks_dir: Path,
    chunk_sec: int,
    overlap_sec: float,
    resume: bool,
) -> Path:
    """Split WAV into overlapping chunks using split_wav_chunks.py."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks_dir / "chunks_manifest.json"

    if resume and manifest_path.exists():
        log(f"  Reuse existing manifest: {manifest_path}")
        return manifest_path

    log(f"  Splitting WAV: {chunk_sec}s chunks, {overlap_sec}s overlap")
    sys.path.insert(0, str(SCRIPT_DIR))
    from split_wav_chunks import split_wav

    split_wav(
        input_wav=str(wav_path),
        output_dir=str(chunks_dir),
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
    )
    return manifest_path


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Invalid or empty manifest: {manifest_path}")
    manifest.sort(key=lambda x: int(x.get("chunk_index", 0)))
    return manifest


# =============================================================================
# Stage 3: Chunk ASR
# =============================================================================

def resolve_chunk_path(record: dict[str, Any], chunks_dir: Path) -> Path:
    """Resolve the actual path to a chunk file."""
    chunk_path = Path(record.get("chunk_path", ""))
    if chunk_path.exists():
        return chunk_path
    chunk_filename = record.get("chunk_filename")
    if not chunk_filename:
        raise FileNotFoundError(f"Chunk record missing chunk_filename: {record}")
    fallback = chunks_dir / chunk_filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Chunk file not found: {record}")


def run_chunk_asr(
    manifest: list[dict[str, Any]],
    chunks_dir: Path,
    chunk_runs_dir: Path,
    raw_json_dir: Path,
    asr_python: str,
    hf_token: str | None,
    language: str | None,
    num_speakers: int | None,
    asr_backend: str,
    asr_model: str,
    gpu_index: int,
    skip_alignment: bool,
    resume: bool,
    continue_on_error: bool,
    max_chunks: int | None,
) -> tuple[int, list[str]]:
    """Run ASR + diarization on each chunk."""
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
            log(f"  [{idx}/{total}] Skip (exists): {target_json.name}")
            processed += 1
            continue

        log(f"  [{idx}/{total}] ASR: {chunk_wav.name}")
        out_dir = chunk_runs_dir / f"output_{chunk_base}"

        cmd = [
            asr_python,
            str(SCRIPT_DIR / "process_single_file_pipeline_AG.py"),
            "--episode_path", str(chunk_wav),
            "--out_dir", str(out_dir),
            "--gpu_index", str(gpu_index),
            "--asr_backend", asr_backend,
            "--asr_model", asr_model,
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
                        f"Cannot find whisper JSON in {json_dir}"
                    )
            shutil.copy2(generated_json, target_json)
            processed += 1
        except Exception as exc:
            failed.append(chunk_wav.name)
            log(f"  Chunk failed: {chunk_wav.name} -> {exc}")
            if not continue_on_error:
                raise

    return processed, failed


# =============================================================================
# Stage 4 & 5: Clean + Combine
# =============================================================================

def clean_and_combine(
    raw_json_dir: Path,
    clean_dir: Path,
    combined_json: Path,
    overlap_sec: float,
) -> int:
    """Clean hallucination artifacts and combine chunk JSONs."""
    raw_files = sorted(glob.glob(str(raw_json_dir / "chunk_*_whisperx.json")))
    if not raw_files:
        raise FileNotFoundError(f"No chunk JSONs in: {raw_json_dir}")

    sys.path.insert(0, str(SCRIPT_DIR))
    from clean_and_combine_whisperx_overlap import clean_one_file, combine_files

    clean_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files: list[str] = []

    for src in raw_files:
        cleaned_path, _ = clean_one_file(src, str(clean_dir))
        cleaned_files.append(cleaned_path)

    combined_json.parent.mkdir(parents=True, exist_ok=True)
    combine_files(cleaned_files, str(combined_json), overlap_sec)
    return len(raw_files)


# =============================================================================
# Stage 6: BERTopic
# =============================================================================

def run_bertopic(
    combined_json: Path,
    topics_dir: Path,
    nlp_python: str,
    topic_chunk_size: int,
    embedding_model: str,
    use_german_stopwords: bool,
    resume: bool,
) -> Path:
    """Run BERTopic topic modeling on combined transcript."""
    topics_dir.mkdir(parents=True, exist_ok=True)
    topics_json = topics_dir / "chunked_transcript_with_topics.json"

    if resume and topics_json.exists():
        log(f"  Skip BERTopic (exists): {topics_json}")
        return topics_json

    cmd = [
        nlp_python,
        str(SCRIPT_DIR / "bertopic_from_combined_json.py"),
        "--input_json", str(combined_json),
        "--output_dir", str(topics_dir),
        "--chunk_size", str(topic_chunk_size),
        "--embedding_model", embedding_model,
    ]
    if use_german_stopwords:
        cmd.append("--use_german_stopwords")
    run_cmd(cmd)
    return topics_json


# =============================================================================
# Stage 7: Flair NER
# =============================================================================

def run_flair_ner(
    topics_json: Path,
    topics_dir: Path,
    nlp_python: str,
    ner_model: str,
    gpu_index: int | None,
    force_cpu: bool,
    resume: bool,
) -> Path:
    """Run Flair NER on topic chunks."""
    ner_json = topics_dir / "chunked_transcript_with_topics_ner.json"

    if resume and ner_json.exists():
        log(f"  Skip NER (exists): {ner_json}")
        return ner_json

    cmd = [
        nlp_python,
        str(SCRIPT_DIR / "flair_ner_on_chunked_json.py"),
        "--input_json", str(topics_json),
        "--output_json", str(ner_json),
        "--model", ner_model,
    ]
    if gpu_index is not None:
        cmd.extend(["--gpu_index", str(gpu_index)])
    if force_cpu:
        cmd.append("--force_cpu")
    run_cmd(cmd)
    return ner_json


# =============================================================================
# Stage 8: Merge back to whisper JSON
# =============================================================================

def merge_annotations(
    combined_json: Path,
    chunked_json: Path,
    final_json: Path,
    nlp_python: str,
    topic_chunk_size: int,
) -> None:
    """Merge topic + NER annotations back into whisper-style JSON."""
    final_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nlp_python,
        str(SCRIPT_DIR / "merge_chunk_topics_ner_back_to_whisper_json.py"),
        "--whisper_json", str(combined_json),
        "--chunked_json", str(chunked_json),
        "--output_json", str(final_json),
        "--chunk_size", str(topic_chunk_size),
    ]
    run_cmd(cmd)


# =============================================================================
# Summary
# =============================================================================

def write_summary(
    summary_path: Path,
    input_path: Path,
    work_dir: Path,
    wav_path: Path,
    manifest_path: Path,
    manifest_len: int,
    processed_chunks: int,
    failed_chunks: list[str],
    combined_json: Path,
    topics_json: Path | None,
    ner_json: Path | None,
    final_json: Path | None,
    elapsed_sec: float,
    params: dict[str, Any],
) -> None:
    final_num_segments = None
    if final_json and final_json.exists():
        try:
            with final_json.open("r", encoding="utf-8") as f:
                d = json.load(f)
            final_num_segments = len(d.get("segments", []))
        except Exception:
            pass

    summary = {
        "input": str(input_path),
        "work_dir": str(work_dir),
        "wav": str(wav_path),
        "manifest": str(manifest_path),
        "num_manifest_chunks": manifest_len,
        "processed_chunks": processed_chunks,
        "failed_chunks": failed_chunks,
        "combined_json": str(combined_json),
        "topics_json": str(topics_json) if topics_json else None,
        "ner_json": str(ner_json) if ner_json else None,
        "final_json": str(final_json) if final_json else None,
        "final_num_segments": final_num_segments,
        "elapsed_seconds": elapsed_sec,
        "params": {k: v for k, v in params.items() if not k.endswith("_python")},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =============================================================================
# Main pipeline
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V3 Unified Pipeline: wav→chunks→ASR→clean→topics→NER→final JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with GPU 0
  python pipeline_v3_unified.py --episode_path video.mp4 --work_dir output_v3

  # Skip topics/NER (only ASR + combine)
  python pipeline_v3_unified.py --episode_path video.mp4 --work_dir output_v3 --skip_topics

  # Resume interrupted run
  python pipeline_v3_unified.py --episode_path video.mp4 --work_dir output_v3

  # Use specific chunk range for debugging
  python pipeline_v3_unified.py --episode_path video.mp4 --work_dir output_v3 --max_chunks 3
        """,
    )

    # === Input/Output ===
    parser.add_argument("--episode_path", required=True, help="Input video/audio file path")
    parser.add_argument("--work_dir",
                        default="/home/zhyin/cineminds/data-pipeline-main/V3/V3_data",
                        help="Pipeline output directory")
    parser.add_argument("--config", default=None, help="YAML config file (overrides defaults)")

    # === Audio chunking ===
    parser.add_argument("--chunk_sec", type=int, default=600, help="Chunk length in seconds (default: 600)")
    parser.add_argument("--overlap_sec", type=float, default=15.0, help="Chunk overlap in seconds (default: 15)")
    parser.add_argument("--max_chunks", type=int, default=None, help="Debug: limit to first N chunks")

    # === ASR ===
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--language", default=None, help="ASR language code, e.g. de")
    parser.add_argument("--num_speakers", type=int, default=None, help="Fixed number of speakers")
    parser.add_argument("--asr_backend", default="transformers",
                        choices=["transformers", "openai"],
                        help="ASR backend (default: transformers)")
    parser.add_argument("--asr_model", default="openai/whisper-small",
                        help="ASR model name (default: openai/whisper-small)")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index for ASR")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip forced alignment")

    # === Topic modeling ===
    parser.add_argument("--topic_chunk_size", type=int, default=8,
                        help="Segments per topic chunk (user-validated: 8 is better than 5)")
    parser.add_argument("--use_german_stopwords", action="store_true", default=True,
                        help="Enable German stopwords for topic modeling")
    parser.add_argument("--no_german_stopwords", action="store_false", dest="use_german_stopwords",
                        help="Disable German stopwords")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="SentenceTransformer embedding model")
    parser.add_argument("--topic_embedding_model", dest="embedding_model",
                        help="Alias for --embedding_model")

    # === NER ===
    parser.add_argument("--ner_model", default="flair/ner-german-large",
                        help="Flair NER model (default: flair/ner-german-large)")
    parser.add_argument("--ner_force_cpu", action="store_true",
                        help="Force NER to run on CPU")

    # === Stage control ===
    parser.add_argument("--skip_topics", action="store_true",
                        help="Skip BERTopic, NER, and merge (produce only combined WhisperX JSON)")
    parser.add_argument("--skip_ner", action="store_true",
                        help="Skip NER (run BERTopic but not NER, then merge)")

    # === Runtime ===
    parser.add_argument("--continue_on_chunk_error", action="store_true",
                        help="Continue if individual chunks fail")
    parser.add_argument("--no_resume", action="store_true",
                        help="Re-run all steps even if outputs exist")

    # === Environment ===
    parser.add_argument("--asr_python",
                        default=None,
                        help="Python for ASR environment (default: auto-detect data-env)")
    parser.add_argument("--nlp_python",
                        default=None,
                        help="Python for NLP environment (default: auto-detect topic-env)")

    return parser


def auto_detect_python(env_name: str, candidates: list[str]) -> str | None:
    """Try to auto-detect a Python executable from a virtual environment."""
    for cand in candidates:
        p = Path(cand)
        if p.exists() and p.is_file():
            return str(p)
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Build effective params dict (exclude internal Python paths for cleanliness)
    params = {k: v for k, v in vars(args).items()
               if k not in ("config", "asr_python", "nlp_python")}

    # Load config file if provided
    if args.config:
        config = load_config(Path(args.config))
        # Config values can fill in defaults for unspecified CLI args
        for key, val in config.items():
            if key not in params or params[key] is None:
                params[key] = val
        # Re-apply after config merge
        for k, v in params.items():
            setattr(args, k, v)

    resume = not args.no_resume

    # Resolve input
    input_path = Path(args.episode_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Auto-detect Python environments
    asr_python = args.asr_python
    if asr_python is None:
        detected = auto_detect_python("data-env", [
            "/home/zhyin/cineminds/data-env/bin/python",
            str(SCRIPT_DIR.parent / "data-env/bin/python"),
            str(Path.home() / "cineminds/data-env/bin/python"),
        ])
        if detected:
            asr_python = detected

    nlp_python = args.nlp_python
    if nlp_python is None:
        detected = auto_detect_python("topic-env", [
            "/home/zhyin/cineminds/topic-env/bin/python",
            str(SCRIPT_DIR.parent / "topic-env/bin/python"),
            str(Path.home() / "cineminds/topic-env/bin/python"),
        ])
        if detected:
            nlp_python = detected

    # Validate pythons
    if asr_python:
        asr_python = ensure_python(asr_python)
    if nlp_python:
        nlp_python = ensure_python(nlp_python)

    log(f"ASR python: {asr_python or 'inherit'}")
    log(f"NLP python: {nlp_python or 'inherit'}")

    base_name = input_path.stem
    work_dir = Path(args.work_dir).resolve()

    # Directory layout
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
    TOTAL_STAGES = 8

    # -------------------------------------------------------------------------
    # Stage 1: Extract WAV
    # -------------------------------------------------------------------------
    log_stage(1, TOTAL_STAGES, "Extract WAV (16kHz mono)")
    wav_path = ensure_wav(input_path, wav_dir, resume=resume)

    # -------------------------------------------------------------------------
    # Stage 2: Split WAV
    # -------------------------------------------------------------------------
    log_stage(2, TOTAL_STAGES, "Split WAV into overlapping chunks")
    manifest_path = split_wav_chunks(
        wav_path=wav_path,
        chunks_dir=chunks_dir,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        resume=resume,
    )
    manifest = load_manifest(manifest_path)
    log(f"  Created {len(manifest)} chunks")

    # -------------------------------------------------------------------------
    # Stage 3: Chunk ASR
    # -------------------------------------------------------------------------
    log_stage(3, TOTAL_STAGES, f"Chunk ASR ({len(manifest)} chunks)")
    processed_chunks, failed_chunks = run_chunk_asr(
        manifest=manifest,
        chunks_dir=chunks_dir,
        chunk_runs_dir=chunk_runs_dir,
        raw_json_dir=raw_json_dir,
        asr_python=asr_python or sys.executable,
        hf_token=args.hf_token,
        language=args.language,
        num_speakers=args.num_speakers,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        gpu_index=args.gpu_index,
        skip_alignment=args.skip_alignment,
        resume=resume,
        continue_on_error=args.continue_on_chunk_error,
        max_chunks=args.max_chunks,
    )
    if failed_chunks:
        log(f"  WARNING: {len(failed_chunks)} chunks failed: {failed_chunks}")

    # -------------------------------------------------------------------------
    # Stage 4 & 5: Clean + Combine
    # -------------------------------------------------------------------------
    log_stage(4, TOTAL_STAGES, "Clean hallucination artifacts from chunks")
    combined_count = clean_and_combine(
        raw_json_dir=raw_json_dir,
        clean_dir=clean_dir,
        combined_json=combined_json,
        overlap_sec=args.overlap_sec,
    )
    log(f"  Combined {combined_count} chunk JSONs → {combined_json.name}")

    # -------------------------------------------------------------------------
    # Early exit if skipping topics
    # -------------------------------------------------------------------------
    if args.skip_topics:
        elapsed = round(time.time() - start_ts, 2)
        write_summary(
            summary_path=summary_json,
            input_path=input_path,
            work_dir=work_dir,
            wav_path=wav_path,
            manifest_path=manifest_path,
            manifest_len=len(manifest),
            processed_chunks=processed_chunks,
            failed_chunks=failed_chunks,
            combined_json=combined_json,
            topics_json=None,
            ner_json=None,
            final_json=None,
            elapsed_sec=elapsed,
            params=params,
        )
        log(f"Pipeline complete (topics/ner skipped)")
        log(f"Combined JSON: {combined_json}")
        log(f"Summary: {summary_json}")
        return

    # -------------------------------------------------------------------------
    # Stage 6: BERTopic
    # -------------------------------------------------------------------------
    log_stage(6, TOTAL_STAGES, "BERTopic topic modeling")
    topics_json = run_bertopic(
        combined_json=combined_json,
        topics_dir=topics_dir,
        nlp_python=nlp_python or sys.executable,
        topic_chunk_size=args.topic_chunk_size,
        embedding_model=args.embedding_model,
        use_german_stopwords=args.use_german_stopwords,
        resume=resume,
    )

    # -------------------------------------------------------------------------
    # Stage 7: Flair NER
    # -------------------------------------------------------------------------
    chunked_for_merge: Path
    if args.skip_ner:
        log_stage(7, TOTAL_STAGES, "Skip NER (--skip_ner)")
        chunked_for_merge = topics_json
    else:
        log_stage(7, TOTAL_STAGES, "Flair NER")
        ner_json = run_flair_ner(
            topics_json=topics_json,
            topics_dir=topics_dir,
            nlp_python=nlp_python or sys.executable,
            ner_model=args.ner_model,
            gpu_index=args.gpu_index,
            force_cpu=args.ner_force_cpu,
            resume=resume,
        )
        chunked_for_merge = ner_json

    # -------------------------------------------------------------------------
    # Stage 8: Merge annotations back
    # -------------------------------------------------------------------------
    log_stage(8, TOTAL_STAGES, "Merge topic+NER into whisper JSON")
    merge_annotations(
        combined_json=combined_json,
        chunked_json=chunked_for_merge,
        final_json=final_json,
        nlp_python=nlp_python or sys.executable,
        topic_chunk_size=args.topic_chunk_size,
    )

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    elapsed = round(time.time() - start_ts, 2)
    write_summary(
        summary_path=summary_json,
        input_path=input_path,
        work_dir=work_dir,
        wav_path=wav_path,
        manifest_path=manifest_path,
        manifest_len=len(manifest),
        processed_chunks=processed_chunks,
        failed_chunks=failed_chunks,
        combined_json=combined_json,
        topics_json=topics_json,
        ner_json=ner_json,
        final_json=final_json,
        elapsed_sec=elapsed,
        params=params,
    )

    log(f"Pipeline complete in {elapsed}s")
    log(f"Final JSON: {final_json}")
    log(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
