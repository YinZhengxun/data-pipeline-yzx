#!/usr/bin/env python3
import os
import json
import argparse

from flair.data import Sentence
from flair.models import SequenceTagger


def extract_entities(text, tagger):
    sentence = Sentence(text)
    tagger.predict(sentence)

    entities = []
    for span in sentence.get_spans("ner"):
        entities.append({
            "text": span.text,
            "label": span.tag,
            "score": round(float(span.score), 4),
            "start_pos": span.start_position,
            "end_pos": span.end_position
        })
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to chunked transcript JSON with topics")
    parser.add_argument("--output_json", required=True, help="Path to save enriched JSON with NER")
    parser.add_argument("--model", default="flair/ner-german-large", help="Flair NER model")
    args = parser.parse_args()

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

        if i % 20 == 0 or i == len(chunks):
            print(f"Processed {i}/{len(chunks)} chunks")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved output: {args.output_json}")
    print(f"Total entities found: {total_entities}")


if __name__ == "__main__":
    main()
