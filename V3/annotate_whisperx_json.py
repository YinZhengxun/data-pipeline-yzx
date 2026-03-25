#!/usr/bin/env python3
"""
Enrich a WhisperX-style JSON with:
1) BERTopic topic modeling on sentence chunks (semantic or fixed-size)
2) Named Entity Recognition (Flair or BERT fallback)

The script keeps the original JSON structure and augments `segments` with:
- topic
- topic_id
- topic_keywords
- named_entities

For dict-style input JSON (with `segments` key), it also appends
`analysis_metadata` and `topic_chunks` at top level.
"""

import argparse
import copy
import datetime as dt
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add BERTopic + NER annotations to an existing WhisperX JSON."
    )
    parser.add_argument("--input_json", required=True, help="Path to input WhisperX JSON")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Path to output JSON (default: <input>_annotated.json)",
    )
    parser.add_argument(
        "--chunk_mode",
        choices=["semantic", "fixed"],
        default="semantic",
        help="Chunking mode for topic modeling",
    )
    parser.add_argument(
        "--sentences_per_chunk",
        type=int,
        default=5,
        help="Used by fixed chunking mode",
    )
    parser.add_argument(
        "--semantic_similarity_threshold",
        type=float,
        default=0.42,
        help="Lower threshold => more chunk splits in semantic mode",
    )
    parser.add_argument(
        "--semantic_min_sentences",
        type=int,
        default=3,
        help="Minimum sentences per semantic chunk before split by similarity",
    )
    parser.add_argument(
        "--semantic_max_sentences",
        type=int,
        default=8,
        help="Maximum sentences per semantic chunk",
    )
    parser.add_argument(
        "--embedding_model",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model for semantic chunking and BERTopic",
    )
    parser.add_argument(
        "--min_topic_size",
        type=int,
        default=2,
        help="BERTopic minimum topic size",
    )
    parser.add_argument(
        "--skip_topics",
        action="store_true",
        help="Skip BERTopic step",
    )
    parser.add_argument(
        "--skip_ner",
        action="store_true",
        help="Skip NER step",
    )
    parser.add_argument(
        "--ner_model",
        choices=["flair", "bert"],
        default="flair",
        help="NER backend",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=None,
        help="GPU index for NER (if available)",
    )
    return parser.parse_args()


def load_json(path: str) -> Tuple[object, List[Dict], bool]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, data, True
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return data, data["segments"], False
    raise ValueError(
        "Unsupported JSON structure. Expected either a list of segments "
        "or a dict containing a 'segments' list."
    )


def default_output_path(input_json: str) -> str:
    root, ext = os.path.splitext(input_json)
    return f"{root}_annotated{ext or '.json'}"


def split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    text = (text or "").strip()
    if not text:
        return []

    parts = SENTENCE_SPLIT_RE.split(text)
    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for part in parts:
        sent = part.strip()
        if not sent:
            continue
        pos = text.find(sent, cursor)
        if pos < 0:
            pos = cursor
        start = pos
        end = start + len(sent)
        out.append((sent, start, end))
        cursor = end
    return out


def build_sentence_records(segments: List[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for seg_idx, seg in enumerate(segments):
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        sentences = split_sentences_with_offsets(seg_text)
        if not sentences:
            sentences = [(seg_text, 0, len(seg_text))]
        for sentence_idx, (sent_text, start_char, end_char) in enumerate(sentences):
            records.append(
                {
                    "segment_idx": seg_idx,
                    "sentence_idx_in_segment": sentence_idx,
                    "text": sent_text,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )
    return records


def fixed_chunk_sentence_indices(total_sentences: int, sentences_per_chunk: int) -> List[List[int]]:
    if sentences_per_chunk <= 0:
        sentences_per_chunk = 5
    chunks: List[List[int]] = []
    current: List[int] = []
    for i in range(total_sentences):
        current.append(i)
        if len(current) >= sentences_per_chunk:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def semantic_chunk_sentence_indices(
    sentence_records: List[Dict],
    embedding_model: str,
    similarity_threshold: float,
    min_sentences: int,
    max_sentences: int,
) -> List[List[int]]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Semantic chunking requires sentence-transformers and numpy. "
            "Install with: pip install sentence-transformers numpy"
        ) from e

    texts = [r["text"] for r in sentence_records]
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    min_sentences = max(1, min_sentences)
    max_sentences = max(min_sentences, max_sentences)

    chunks: List[List[int]] = []
    current = [0]
    current_sum = embeddings[0].copy()

    for i in range(1, len(sentence_records)):
        centroid = current_sum / max(1, len(current))
        denom = float(np.linalg.norm(centroid)) + 1e-12
        centroid_norm = centroid / denom
        similarity = float(np.dot(embeddings[i], centroid_norm))

        should_split = False
        if len(current) >= max_sentences:
            should_split = True
        elif len(current) >= min_sentences and similarity < similarity_threshold:
            should_split = True

        if should_split:
            chunks.append(current)
            current = [i]
            current_sum = embeddings[i].copy()
        else:
            current.append(i)
            current_sum += embeddings[i]

    if current:
        chunks.append(current)
    return chunks


def build_chunks(
    sentence_records: List[Dict],
    chunk_mode: str,
    sentences_per_chunk: int,
    embedding_model: str,
    similarity_threshold: float,
    min_sentences: int,
    max_sentences: int,
) -> Tuple[List[Dict], str]:
    if not sentence_records:
        return [], chunk_mode

    used_mode = chunk_mode
    if chunk_mode == "semantic":
        try:
            chunk_indices = semantic_chunk_sentence_indices(
                sentence_records=sentence_records,
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
            )
        except Exception as e:
            print(
                f"[WARN] Semantic chunking unavailable ({e}). "
                "Falling back to fixed-size chunking."
            )
            used_mode = "fixed"
            chunk_indices = fixed_chunk_sentence_indices(
                total_sentences=len(sentence_records),
                sentences_per_chunk=sentences_per_chunk,
            )
    else:
        chunk_indices = fixed_chunk_sentence_indices(
            total_sentences=len(sentence_records),
            sentences_per_chunk=sentences_per_chunk,
        )

    chunks: List[Dict] = []
    for chunk_id, sent_ids in enumerate(chunk_indices):
        texts = [sentence_records[i]["text"] for i in sent_ids]
        seg_ids = [sentence_records[i]["segment_idx"] for i in sent_ids]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "sentence_indices": sent_ids,
                "segment_indices": seg_ids,
                "text": " ".join(texts).strip(),
            }
        )
    return chunks, used_mode


def simple_keywords(text: str, top_k: int = 5) -> List[str]:
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "you", "your",
        "are", "was", "were", "have", "has", "had", "they", "them", "their",
        "und", "der", "die", "das", "ein", "eine", "ist", "mit", "auf", "von",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    counts = Counter(t for t in tokens if t not in stop)
    return [w for w, _ in counts.most_common(top_k)]


def normalize_topic_name(topic_id: int, raw_name: Optional[str], keywords: List[str]) -> str:
    if topic_id == -1:
        return "Other"
    name = (raw_name or "").strip()
    prefix = f"{topic_id}_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    name = name.replace("_", " ").strip()
    if name:
        return name.title()
    if keywords:
        return " / ".join(keywords[:3]).title()
    return f"Topic_{topic_id}"


def run_bertopic(chunks: List[Dict], embedding_model: str, min_topic_size: int) -> Tuple[List[int], Dict[int, Dict], Dict]:
    texts = [c["text"] for c in chunks if c.get("text")]
    if not texts:
        return [], {}, {"status": "skipped", "reason": "no chunk text"}

    if len(texts) < len(chunks):
        # Keep 1:1 with chunks by using fallback per-chunk text when empty.
        texts = [c.get("text", "") or "." for c in chunks]
    else:
        texts = [c.get("text", "") for c in chunks]

    if len(texts) == 1:
        keywords = simple_keywords(texts[0], top_k=5)
        topic_meta = {
            0: {"name": normalize_topic_name(0, "Topic_0", keywords), "keywords": keywords}
        }
        return [0], topic_meta, {"status": "ok", "note": "single chunk => single topic"}

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "BERTopic requires bertopic and sentence-transformers. "
            "Install with: pip install bertopic sentence-transformers"
        ) from e

    embedder = SentenceTransformer(embedding_model)
    topic_model = BERTopic(
        embedding_model=embedder,
        min_topic_size=max(2, min(min_topic_size, len(texts))),
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    raw_name_by_id: Dict[int, str] = {}
    for _, row in topic_info.iterrows():
        tid = int(row.get("Topic", -1))
        raw_name_by_id[tid] = str(row.get("Name", "")).strip()

    topic_meta: Dict[int, Dict] = {}
    for tid in sorted(set(int(t) for t in topics)):
        words = []
        if tid != -1:
            words = [w for w, _ in (topic_model.get_topic(tid) or [])[:5]]
        name = normalize_topic_name(tid, raw_name_by_id.get(tid), words)
        topic_meta[tid] = {"name": name, "keywords": words}

    return list(map(int, topics)), topic_meta, {"status": "ok", "num_topics": len(topic_meta)}


def assign_topics_to_segments(
    segments: List[Dict],
    sentence_records: List[Dict],
    chunks: List[Dict],
    chunk_topics: List[int],
    topic_meta: Dict[int, Dict],
) -> Tuple[int, int]:
    if not chunks or not chunk_topics:
        return 0, 0

    segment_votes: Dict[int, List[int]] = defaultdict(list)
    for chunk, topic_id in zip(chunks, chunk_topics):
        for sent_idx in chunk["sentence_indices"]:
            seg_idx = sentence_records[sent_idx]["segment_idx"]
            segment_votes[seg_idx].append(int(topic_id))
        chunk["topic_id"] = int(topic_id)
        meta = topic_meta.get(int(topic_id), {"name": "Other", "keywords": []})
        chunk["topic"] = meta["name"]
        chunk["topic_keywords"] = meta.get("keywords", [])

    assigned = 0
    for seg_idx, seg in enumerate(segments):
        votes = segment_votes.get(seg_idx, [])
        if not votes:
            continue
        best_topic = Counter(votes).most_common(1)[0][0]
        meta = topic_meta.get(best_topic, {"name": "Other", "keywords": []})
        seg["topic_id"] = int(best_topic)
        seg["topic"] = meta["name"]
        seg["topic_keywords"] = meta.get("keywords", [])
        assigned += 1

    return assigned, len(segment_votes)


def map_char_to_time(seg: Dict, char_pos: int) -> float:
    text = seg.get("text", "") or ""
    seg_start = float(seg.get("start", 0.0) or 0.0)
    seg_end = float(seg.get("end", seg_start) or seg_start)
    if seg_end < seg_start:
        seg_end = seg_start
    duration = seg_end - seg_start
    if duration <= 0 or not text:
        return round(seg_start, 3)
    ratio = max(0.0, min(1.0, float(char_pos) / max(1, len(text))))
    return round(seg_start + ratio * duration, 3)


def run_ner(segments: List[Dict], ner_model: str, gpu_index: Optional[int]) -> Tuple[List[Dict], Dict]:
    all_entities: List[Dict] = []
    label_counts: Counter = Counter()

    if ner_model == "flair":
        try:
            import torch
            import flair
            from flair.data import Sentence
            from flair.models import SequenceTagger
        except Exception as e:
            raise RuntimeError(
                "Flair NER requested but Flair is not installed. "
                "Install with: pip install flair"
            ) from e

        if torch.cuda.is_available():
            use_idx = gpu_index if gpu_index is not None else 0
            flair.device = torch.device(f"cuda:{use_idx}")
            print(f"[NER] Using flair on cuda:{use_idx}")
        else:
            flair.device = torch.device("cpu")
            print("[NER] Using flair on cpu")

        tagger = SequenceTagger.load("flair/ner-english-large")

        for seg_idx, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            if not text:
                seg["named_entities"] = []
                continue
            sentence = Sentence(text)
            tagger.predict(sentence)
            entities = []
            for span in sentence.get_spans("ner"):
                start_char = int(span.start_pos)
                end_char = int(span.end_pos)
                ent = {
                    "text": span.text,
                    "label": span.tag,
                    "score": round(float(span.score), 4),
                    "start_char": start_char,
                    "end_char": end_char,
                    "start": map_char_to_time(seg, start_char),
                    "end": map_char_to_time(seg, end_char),
                }
                entities.append(ent)
                label_counts[ent["label"]] += 1
                all_entities.append({"segment_index": seg_idx, **ent})
            seg["named_entities"] = entities
    else:
        try:
            import torch
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError(
                "BERT NER requested but transformers/torch are missing. "
                "Install with: pip install transformers torch"
            ) from e

        device = -1
        if torch.cuda.is_available():
            device = gpu_index if gpu_index is not None else 0
            print(f"[NER] Using transformers NER on cuda:{device}")
        else:
            print("[NER] Using transformers NER on cpu")

        nlp_ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=device,
        )

        for seg_idx, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            if not text:
                seg["named_entities"] = []
                continue
            entities = []
            for pred in nlp_ner(text):
                start_char = int(pred.get("start", 0))
                end_char = int(pred.get("end", 0))
                ent = {
                    "text": pred.get("word", ""),
                    "label": pred.get("entity_group", ""),
                    "score": round(float(pred.get("score", 0.0)), 4),
                    "start_char": start_char,
                    "end_char": end_char,
                    "start": map_char_to_time(seg, start_char),
                    "end": map_char_to_time(seg, end_char),
                }
                entities.append(ent)
                label_counts[ent["label"]] += 1
                all_entities.append({"segment_index": seg_idx, **ent})
            seg["named_entities"] = entities

    return all_entities, {"entity_count": len(all_entities), "label_counts": dict(label_counts)}


def ensure_analysis_metadata(data: Dict) -> Dict:
    meta = data.get("analysis_metadata")
    if not isinstance(meta, dict):
        meta = {}
        data["analysis_metadata"] = meta
    return meta


def main() -> int:
    args = parse_args()
    input_json = args.input_json
    output_json = args.output_json or default_output_path(input_json)

    if not os.path.exists(input_json):
        print(f"[ERROR] Input JSON not found: {input_json}")
        return 1

    try:
        raw_data, segments, input_is_list = load_json(input_json)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return 1

    if not segments:
        print("[ERROR] No segments found in JSON.")
        return 1

    # Work on a deep copy to avoid in-place accidental mutations.
    data = copy.deepcopy(raw_data)
    if input_is_list:
        out_segments = data
    else:
        out_segments = data["segments"]

    print(f"[INFO] Loaded {len(out_segments)} segments from {input_json}")

    sentence_records = build_sentence_records(out_segments)
    print(f"[INFO] Built {len(sentence_records)} sentence units for chunking")

    chunks, used_chunk_mode = build_chunks(
        sentence_records=sentence_records,
        chunk_mode=args.chunk_mode,
        sentences_per_chunk=args.sentences_per_chunk,
        embedding_model=args.embedding_model,
        similarity_threshold=args.semantic_similarity_threshold,
        min_sentences=args.semantic_min_sentences,
        max_sentences=args.semantic_max_sentences,
    )
    print(f"[INFO] Created {len(chunks)} chunks (mode={used_chunk_mode})")

    topic_summary: Dict = {"status": "skipped"}
    if not args.skip_topics:
        try:
            chunk_topics, topic_meta, topic_summary = run_bertopic(
                chunks=chunks,
                embedding_model=args.embedding_model,
                min_topic_size=args.min_topic_size,
            )
            assigned_segments, _ = assign_topics_to_segments(
                segments=out_segments,
                sentence_records=sentence_records,
                chunks=chunks,
                chunk_topics=chunk_topics,
                topic_meta=topic_meta,
            )
            topic_summary.update(
                {
                    "assigned_segments": assigned_segments,
                    "chunk_count": len(chunks),
                    "topic_count": len(topic_meta),
                }
            )
            print(
                f"[TOPIC] Done: {topic_summary.get('topic_count', 0)} topics, "
                f"{assigned_segments} segments assigned."
            )
        except Exception as e:
            print(f"[WARN] Topic modeling failed: {e}")
            topic_summary = {"status": "failed", "error": str(e)}
    else:
        print("[TOPIC] Skipped (--skip_topics)")

    ner_summary: Dict = {"status": "skipped"}
    if not args.skip_ner:
        try:
            all_entities, ner_stats = run_ner(
                segments=out_segments,
                ner_model=args.ner_model,
                gpu_index=args.gpu_index,
            )
            ner_summary = {"status": "ok", **ner_stats}
            print(f"[NER] Done: {len(all_entities)} entities.")
            if not input_is_list and isinstance(data, dict):
                data["named_entities"] = all_entities
        except Exception as e:
            print(f"[WARN] NER failed: {e}")
            ner_summary = {"status": "failed", "error": str(e)}
    else:
        print("[NER] Skipped (--skip_ner)")

    if not input_is_list and isinstance(data, dict):
        analysis_meta = ensure_analysis_metadata(data)
        analysis_meta["generated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        analysis_meta["input_json"] = os.path.abspath(input_json)
        analysis_meta["chunking"] = {
            "mode": used_chunk_mode,
            "sentences_per_chunk": args.sentences_per_chunk,
            "semantic_similarity_threshold": args.semantic_similarity_threshold,
            "semantic_min_sentences": args.semantic_min_sentences,
            "semantic_max_sentences": args.semantic_max_sentences,
            "sentence_count": len(sentence_records),
            "chunk_count": len(chunks),
        }
        analysis_meta["topic_modeling"] = topic_summary
        analysis_meta["ner"] = ner_summary
        data["topic_chunks"] = chunks

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Annotated JSON written to: {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
