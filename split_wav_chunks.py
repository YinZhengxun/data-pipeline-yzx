import os
import json
import math
import argparse
import soundfile as sf


def split_wav(input_wav: str, output_dir: str, chunk_sec: int = 600, overlap_sec: int = 15):
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be > 0")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    os.makedirs(output_dir, exist_ok=True)

    audio, sr = sf.read(input_wav)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    total_samples = len(audio)
    total_sec = total_samples / sr
    step_sec = chunk_sec - overlap_sec

    manifest = []
    chunk_index = 0
    start_sec = 0.0

    while start_sec < total_sec:
        end_sec = min(start_sec + chunk_sec, total_sec)

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        chunk_audio = audio[start_sample:end_sample]

        chunk_name = f"chunk_{chunk_index:04d}_{int(start_sec)}_{int(end_sec)}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)

        sf.write(chunk_path, chunk_audio, sr)

        manifest.append(
            {
                "chunk_index": chunk_index,
                "chunk_filename": chunk_name,
                "chunk_path": chunk_path,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "sample_rate": sr,
            }
        )

        chunk_index += 1
        start_sec += step_sec

    manifest_path = os.path.join(output_dir, "chunks_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Created {len(manifest)} chunks.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav", required=True, help="Path to input wav file")
    parser.add_argument("--output_dir", required=True, help="Directory to save chunks")
    parser.add_argument("--chunk_sec", type=int, default=600, help="Chunk length in seconds")
    parser.add_argument("--overlap_sec", type=int, default=15, help="Overlap length in seconds")
    args = parser.parse_args()

    split_wav(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
    )
