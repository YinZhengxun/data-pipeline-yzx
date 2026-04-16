#!/usr/bin/env python3
"""
Swiss German ASR entrypoint (format-compatible with process_single_file_pipeline_AG.py).

This script intentionally reuses `asr_whisperx_AG.transcribe_and_diarize` so output JSON
stays in the same WhisperX-style structure already used by the current pipeline:
- top-level: segments, word_segments
- segment-level: start, end, text, words, speaker, detected_language, detected_language_probability
- word-level: word, start, end, score, speaker
"""

import argparse
import os
import warnings

from asr_whisperx_AG import transcribe_and_diarize

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

DEFAULT_SWISS_GERMAN_MODEL = "Flurin17/whisper-large-v3-turbo-swiss-german"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swiss German ASR -> Diarization -> Forced Alignment (WhisperX-compatible JSON)"
    )
    parser.add_argument("--episode_path", required=True, help="Path to video/audio file")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    parser.add_argument(
        "--language",
        default="de",
        help="Language code for ASR/alignment (default: de)",
    )
    parser.add_argument("--num_speakers", type=int, default=None, help="Fixed number of speakers")
    parser.add_argument("--out_dir", default="output_pipeline_swiss", help="Output directory")
    parser.add_argument(
        "--asr_backend",
        default="transformers",
        choices=["transformers", "openai"],
        help="ASR backend (default: transformers)",
    )
    parser.add_argument(
        "--asr_model",
        default=DEFAULT_SWISS_GERMAN_MODEL,
        help=(
            "ASR model name (default: Flurin17 Swiss German model). "
            "For baseline fallback, pass distil-whisper/distil-large-v3.5."
        ),
    )
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip word-level alignment")
    parser.add_argument("--asr_batch_size", type=int, default=None, help="Transformers ASR pipeline batch size")
    parser.add_argument("--asr_chunk_length_s", type=float, default=30.0, help="Transformers long-form chunk length (seconds)")
    parser.add_argument("--asr_stride_length_s", type=float, default=5.0, help="Transformers long-form stride length (seconds)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    transcribe_and_diarize(
        video_path=args.episode_path,
        hf_token=args.hf_token,
        output_dir=args.out_dir,
        asr_backend=args.asr_backend,
        asr_model_name=args.asr_model,
        diarization_model="pyannote/speaker-diarization-3.1",
        num_speakers=args.num_speakers,
        gpu_index=args.gpu_index,
        forced_language=args.language,
        skip_alignment=args.skip_alignment,
        asr_batch_size=args.asr_batch_size,
        asr_chunk_length_s=args.asr_chunk_length_s,
        asr_stride_length_s=args.asr_stride_length_s,
    )


if __name__ == "__main__":
    main()
