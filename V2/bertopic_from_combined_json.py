'''#!/usr/bin/env python3
import os
import json
import time
import argparse

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def chunk_segments(segments, chunk_size=5):
    chunks = []
    current = []
    chunk_id = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        current.append(seg)

        if len(current) == chunk_size:
            chunks.append(build_chunk(current, chunk_id))
            chunk_id += 1
            current = []

    if current:
        chunks.append(build_chunk(current, chunk_id))

    return chunks


def build_chunk(segment_group, chunk_id):
    texts = [seg.get("text", "").strip() for seg in segment_group if seg.get("text", "").strip()]
    speakers = [seg.get("speaker", "UNKNOWN") for seg in segment_group]

    return {
        "chunk_id": chunk_id,
        "start": segment_group[0].get("start", 0.0),
        "end": segment_group[-1].get("end", 0.0),
        "num_segments": len(segment_group),
        "speaker_sequence": speakers,
        "text": " ".join(texts)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to combined WhisperX JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of segments per chunk")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        raise ValueError("No segments found in input JSON")

    print(f"Loaded {len(segments)} segments")

    chunks = chunk_segments(segments, chunk_size=args.chunk_size)
    print(f"Built {len(chunks)} text chunks")

    chunks_json_path = os.path.join(args.output_dir, "chunked_transcript.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks: {chunks_json_path}")

    texts = [c["text"] for c in chunks if c["text"].strip()]
    if not texts:
        raise ValueError("No non-empty chunk texts found")

    print(f"Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model, device="cpu")

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        random_state=42,
        n_jobs=1
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(
        min_df=1,
        max_features=1000
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=False
    )

    print("Running BERTopic...")
    t0 = time.time()
    topics, probs = topic_model.fit_transform(texts)
    elapsed = time.time() - t0
    print(f"BERTopic finished in {elapsed:.2f}s")

    for chunk, topic_id in zip(chunks, topics):
        chunk["topic_id"] = int(topic_id)
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            chunk["topic_words"] = [w for w, _ in topic_words[:10]] if topic_words else []
        else:
            chunk["topic_words"] = []

    enriched_json_path = os.path.join(args.output_dir, "chunked_transcript_with_topics.json")
    with open(enriched_json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    print(f"Saved topic-enriched chunks: {enriched_json_path}")

    topic_info = topic_model.get_topic_info()
    csv_path = os.path.join(args.output_dir, "topic_info.csv")
    topic_info.to_csv(csv_path, index=False)
    print(f"Saved topic info CSV: {csv_path}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=12)
        html_path = os.path.join(args.output_dir, "topics_barchart.html")
        fig.write_html(html_path)
        print(f"Saved barchart HTML: {html_path}")
    except Exception as e:
        print(f"Skipping barchart HTML: {e}")


if __name__ == "__main__":
    main()
'''
#!/usr/bin/env python3
import os
import json
import time
import argparse

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


GERMAN_STOPWORDS = [
    "ich", "du", "er", "sie", "es", "wir", "ihr", "man",
    "mich", "dich", "ihn", "ihm", "uns", "euch",
    "mein", "meine", "meiner", "meinem", "meinen",
    "dein", "deine", "deiner", "deinem", "deinen",
    "sein", "seine", "seiner", "seinem", "seinen",
    "ihr", "ihre", "ihren", "ihrem", "ihres",
    "unser", "unsere", "euer", "eure",

    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "einem", "einen", "eines",
    "dies", "diese", "dieser", "diesem", "diesen",
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
    "dass", "ob", "wenn", "weil"
]


def chunk_segments(segments, chunk_size=8):
    chunks = []
    current = []
    chunk_id = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        current.append(seg)

        if len(current) == chunk_size:
            chunks.append(build_chunk(current, chunk_id))
            chunk_id += 1
            current = []

    if current:
        chunks.append(build_chunk(current, chunk_id))

    return chunks


def build_chunk(segment_group, chunk_id):
    texts = [seg.get("text", "").strip() for seg in segment_group if seg.get("text", "").strip()]
    speakers = [seg.get("speaker", "UNKNOWN") for seg in segment_group]

    return {
        "chunk_id": chunk_id,
        "start": segment_group[0].get("start", 0.0),
        "end": segment_group[-1].get("end", 0.0),
        "num_segments": len(segment_group),
        "speaker_sequence": speakers,
        "text": " ".join(texts)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to combined WhisperX JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=8, help="Number of segments per chunk")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="SentenceTransformer model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        raise ValueError("No segments found in input JSON")

    print(f"Loaded {len(segments)} segments")

    chunks = chunk_segments(segments, chunk_size=args.chunk_size)
    print(f"Built {len(chunks)} text chunks")

    chunks_json_path = os.path.join(args.output_dir, "chunked_transcript.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks: {chunks_json_path}")

    texts = [c["text"] for c in chunks if c["text"].strip()]
    if not texts:
        raise ValueError("No non-empty chunk texts found")

    print(f"Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model, device="cpu")

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        random_state=42,
        n_jobs=1
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        stop_words=GERMAN_STOPWORDS,
        min_df=1,
        max_features=1000
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=False
    )

    print("Running BERTopic...")
    t0 = time.time()
    topics, probs = topic_model.fit_transform(texts)
    elapsed = time.time() - t0
    print(f"BERTopic finished in {elapsed:.2f}s")

    for chunk, topic_id in zip(chunks, topics):
        chunk["topic_id"] = int(topic_id)
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            chunk["topic_words"] = [w for w, _ in topic_words[:10]] if topic_words else []
        else:
            chunk["topic_words"] = []

    enriched_json_path = os.path.join(args.output_dir, "chunked_transcript_with_topics.json")
    with open(enriched_json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    print(f"Saved topic-enriched chunks: {enriched_json_path}")

    topic_info = topic_model.get_topic_info()
    csv_path = os.path.join(args.output_dir, "topic_info.csv")
    topic_info.to_csv(csv_path, index=False)
    print(f"Saved topic info CSV: {csv_path}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=12)
        html_path = os.path.join(args.output_dir, "topics_barchart.html")
        fig.write_html(html_path)
        print(f"Saved barchart HTML: {html_path}")
    except Exception as e:
        print(f"Skipping barchart HTML: {e}")


if __name__ == "__main__":
    main()
