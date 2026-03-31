#!/usr/bin/env python3
import argparse
import json

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger


def extract_entities(text, tagger):
    sentence = Sentence(text)
    tagger.predict(sentence)

    entities = []
    for span in sentence.get_spans("ner"):
        entities.append(
            {
                "text": span.text,
                "label": span.tag,
                "score": round(float(span.score), 4),
                "start_pos": span.start_position,
                "end_pos": span.end_position,
            }
        )
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to chunked transcript JSON with topics",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to save enriched JSON with NER",
    )
    parser.add_argument(
        "--model",
        default="flair/ner-german-large",
        help="Flair NER model",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=None,
        help="CUDA GPU index to use (e.g. 1)",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force running on CPU",
    )
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    if args.force_cpu:
        flair.device = torch.device("cpu")
    elif args.gpu_index is not None and cuda_available:
        if 0 <= args.gpu_index < cuda_count:
            flair.device = torch.device(f"cuda:{args.gpu_index}")
        else:
            print(
                f"WARNING: --gpu_index={args.gpu_index} is out of range "
                f"(available: 0..{cuda_count - 1}). Falling back to cuda:0."
            )
            flair.device = torch.device("cuda:0")
    elif cuda_available:
        flair.device = torch.device("cuda:0")
    else:
        flair.device = torch.device("cpu")

    print(f"Using Flair device: {flair.device}")
    print(f"Loading Flair model: {args.model}")
    tagger = SequenceTagger.load(args.model)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        raise ValueError("No 'chunks' found in input JSON")

    print(f"Found {len(chunks)} chunks")
    total_entities = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()

        if not text:
            chunk["entities"] = []
            continue

        entities = extract_entities(text, tagger)
        chunk["entities"] = entities
        total_entities += len(entities)

        if flair.device.type == "cuda":
            torch.cuda.empty_cache()

        if i % 20 == 0 or i == len(chunks):
            print(f"Processed {i}/{len(chunks)} chunks")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved output: {args.output_json}")
    print(f"Total entities found: {total_entities}")


if __name__ == "__main__":
    main()
