#!/usr/bin/env python3
import json
import argparse


def get_non_empty_segments(segments):
    return [seg for seg in segments if seg.get("text", "").strip()]


def chunk_segments(segments, chunk_size=8):
    chunks = []
    current = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        current.append(seg)

        if len(current) == chunk_size:
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_json", required=True, help="Original combined WhisperX JSON with words")
    parser.add_argument("--chunked_json", required=True, help="Chunked JSON with topics and NER")
    parser.add_argument("--output_json", required=True, help="Final whisper-style enriched JSON")
    parser.add_argument("--chunk_size", type=int, default=8, help="Chunk size used during topic modeling")
    args = parser.parse_args()

    # Load original whisper-style JSON
    with open(args.whisper_json, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)

    # Load chunk-level enriched JSON
    with open(args.chunked_json, "r", encoding="utf-8") as f:
        chunked_data = json.load(f)

    segments = whisper_data.get("segments", [])
    chunks = chunked_data.get("chunks", [])

    if not segments:
        raise ValueError("No segments found in whisper_json")
    if not chunks:
        raise ValueError("No chunks found in chunked_json")

    non_empty_segments = get_non_empty_segments(segments)
    grouped_segments = chunk_segments(non_empty_segments, chunk_size=args.chunk_size)

    if len(grouped_segments) != len(chunks):
        raise ValueError(
            f"Mismatch: reconstructed {len(grouped_segments)} segment-groups "
            f"but chunked JSON has {len(chunks)} chunks"
        )

    # Map chunk-level annotations back to each segment
    enriched_count = 0
    for seg_group, chunk in zip(grouped_segments, chunks):
        topic_id = chunk.get("topic_id")
        topic_words = chunk.get("topic_words", [])
        entities = chunk.get("entities", [])
        chunk_id = chunk.get("chunk_id")

        for seg in seg_group:
            seg["chunk_id"] = chunk_id
            seg["topic_id"] = topic_id
            seg["topic_words"] = topic_words
            seg["entities"] = entities
            enriched_count += 1

    # Also keep top-level metadata if useful
    whisper_data["chunk_level_annotation_source"] = {
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunks),
        "annotation_fields": ["topic_id", "topic_words", "entities"]
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(whisper_data, f, ensure_ascii=False, indent=2)

    print(f"Loaded original segments: {len(segments)}")
    print(f"Non-empty segments grouped into: {len(grouped_segments)} chunks")
    print(f"Chunk annotations loaded: {len(chunks)}")
    print(f"Enriched segments: {enriched_count}")
    print(f"Saved final whisper-style enriched JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
