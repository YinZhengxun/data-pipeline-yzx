#!/usr/bin/env python3
"""
Run Swiss German ASR on a full file (no external chunk split).

This is useful when you want to avoid chunk boundary effects and control
Transformers ASR memory with --asr_batch_size instead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swiss German full-file ASR (no chunk splitting)."
    )
    parser.add_argument("--episode_path", required=True, help="Input WAV/MP4 path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--language", default="de", help="Language code (default: de)")
    parser.add_argument("--num_speakers", type=int, default=None, help="Fixed number of speakers")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index")
    parser.add_argument("--asr_backend", default="transformers", choices=["transformers", "openai"])
    parser.add_argument(
        "--asr_model",
        default="Flurin17/whisper-large-v3-turbo-swiss-german",
        help="ASR model name",
    )
    parser.add_argument("--skip_alignment", action="store_true", help="Skip word-level alignment")
    parser.add_argument("--asr_batch_size", type=int, default=1, help="Transformers ASR batch size (default: 1)")
    parser.add_argument("--asr_chunk_length_s", type=float, default=30.0, help="ASR internal chunk length")
    parser.add_argument("--asr_stride_length_s", type=float, default=5.0, help="ASR internal stride length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "process_single_file_pipeline_swiss.py"
    if not target.exists():
        raise FileNotFoundError(f"Cannot find script: {target}")

    cmd = [
        sys.executable,
        str(target),
        "--episode_path",
        args.episode_path,
        "--out_dir",
        args.out_dir,
        "--language",
        args.language,
        "--gpu_index",
        str(args.gpu_index),
        "--asr_backend",
        args.asr_backend,
        "--asr_model",
        args.asr_model,
        "--asr_batch_size",
        str(args.asr_batch_size),
        "--asr_chunk_length_s",
        str(args.asr_chunk_length_s),
        "--asr_stride_length_s",
        str(args.asr_stride_length_s),
    ]
    if args.hf_token:
        cmd.extend(["--hf_token", args.hf_token])
    if args.num_speakers is not None:
        cmd.extend(["--num_speakers", str(args.num_speakers)])
    if args.skip_alignment:
        cmd.append("--skip_alignment")

    print("[SWISS-ASR-FULLFILE] Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
