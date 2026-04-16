#!/usr/bin/env python3
"""
Post-process full-file WhisperX JSON for downstream topic/NER usage.

What this script does:
1. Split overly long segments using word-level timestamps + punctuation boundaries.
2. Fill missing speaker labels (word-level first, then segment-level).
3. Normalize detected language labels (default to `de`, keep only confident English by default).

Input/Output format remains WhisperX-compatible:
- top-level: segments, word_segments
- segment fields: start, end, text, words, speaker, detected_language, detected_language_probability
- word fields: word, start, end, score, speaker
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

SENTENCE_END_CHARS = {".", "!", "?", ";", ":"}
ENGLISH_HINT_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "you",
    "your",
    "i",
    "we",
    "they",
    "he",
    "she",
    "it",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "thanks",
    "thank",
    "sorry",
    "nice",
    "cool",
    "okay",
    "yes",
    "no",
}
TOKEN_RE = re.compile(r"[A-Za-z']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process full-file WhisperX JSON: split long segments + speaker/language cleanup."
    )
    parser.add_argument("--input_json", required=True, help="Path to full-file WhisperX JSON")
    parser.add_argument("--output_json", default=None, help="Output path (default: <input>_postprocessed.json)")
    parser.add_argument("--max_segment_sec", type=float, default=20.0, help="Hard max duration per segment")
    parser.add_argument("--target_segment_sec", type=float, default=12.0, help="Preferred duration near punctuation")
    parser.add_argument("--min_segment_sec", type=float, default=4.0, help="Minimum desired split duration")
    parser.add_argument("--default_language", default="de", help="Fallback/default language label")
    parser.add_argument("--english_min_prob", type=float, default=0.80, help="Min prob to keep non-default English")
    parser.add_argument(
        "--keep_other_languages",
        action="store_true",
        help="Keep non-default, non-English labels when confidence is high (otherwise force to default)",
    )
    parser.add_argument(
        "--other_lang_min_prob",
        type=float,
        default=0.90,
        help="Min confidence when --keep_other_languages is enabled",
    )
    parser.add_argument("--sort_output", action="store_true", help="Sort segments/word_segments by start time")
    return parser.parse_args()


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def is_sentence_end_token(token: str) -> bool:
    t = (token or "").strip()
    if not t:
        return False
    return t[-1] in SENTENCE_END_CHARS


def normalize_text_from_words(words: list[dict[str, Any]]) -> str:
    text = " ".join((w.get("word") or "").strip() for w in words if (w.get("word") or "").strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def majority_speaker_from_words(words: list[dict[str, Any]]) -> str | None:
    speakers = [w.get("speaker") for w in words if w.get("speaker")]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]


def fill_word_speakers(words: list[dict[str, Any]], default_segment_speaker: str | None) -> int:
    """Fill missing word speakers in-place. Returns count filled."""
    filled = 0
    if not words:
        return filled

    prev = None
    for w in words:
        if w.get("speaker"):
            prev = w["speaker"]
        elif prev:
            w["speaker"] = prev
            filled += 1

    nxt = None
    for w in reversed(words):
        if w.get("speaker"):
            nxt = w["speaker"]
        elif nxt:
            w["speaker"] = nxt
            filled += 1

    if default_segment_speaker:
        for w in words:
            if not w.get("speaker"):
                w["speaker"] = default_segment_speaker
                filled += 1

    return filled


def looks_like_english(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "thank you" in t:
        return True
    tokens = TOKEN_RE.findall(t)
    if len(tokens) < 2:
        return False
    hits = sum(1 for tok in tokens if tok in ENGLISH_HINT_WORDS)
    ratio = hits / max(1, len(tokens))
    return hits >= 2 and ratio >= 0.25


def normalize_language(
    lang: str | None,
    prob: float | None,
    text: str,
    default_language: str,
    english_min_prob: float,
    keep_other_languages: bool,
    other_lang_min_prob: float,
) -> tuple[str, float | None]:
    if not lang:
        return default_language, None

    lang = str(lang).strip().lower()
    if not lang:
        return default_language, None
    if lang == default_language:
        return lang, prob

    if lang == "en":
        if prob is not None and prob >= english_min_prob and looks_like_english(text):
            return lang, prob
        return default_language, None

    if keep_other_languages and prob is not None and prob >= other_lang_min_prob:
        return lang, prob

    return default_language, None


def cleanup_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in words or []:
        word = (w.get("word") or "").strip()
        start = safe_float(w.get("start"))
        end = safe_float(w.get("end"))
        if not word or start is None or end is None or end < start:
            continue
        ww = {
            "word": word,
            "start": round(start, 3),
            "end": round(end, 3),
        }
        if "score" in w:
            score = safe_float(w.get("score"))
            ww["score"] = round(score, 3) if score is not None else None
        if w.get("speaker"):
            ww["speaker"] = w.get("speaker")
        out.append(ww)
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def build_segment_from_words(
    words: list[dict[str, Any]],
    source_segment: dict[str, Any],
    default_language: str,
    english_min_prob: float,
    keep_other_languages: bool,
    other_lang_min_prob: float,
) -> dict[str, Any]:
    text = normalize_text_from_words(words)
    start = round(float(words[0]["start"]), 3)
    end = round(float(words[-1]["end"]), 3)

    seg = {
        "start": start,
        "end": end,
        "text": text,
        "words": words,
    }

    speaker = majority_speaker_from_words(words) or source_segment.get("speaker")
    if speaker:
        seg["speaker"] = speaker

    raw_lang = source_segment.get("detected_language")
    raw_prob = safe_float(source_segment.get("detected_language_probability"))
    lang, prob = normalize_language(
        lang=raw_lang,
        prob=raw_prob,
        text=text,
        default_language=default_language,
        english_min_prob=english_min_prob,
        keep_other_languages=keep_other_languages,
        other_lang_min_prob=other_lang_min_prob,
    )
    seg["detected_language"] = lang
    seg["detected_language_probability"] = round(prob, 4) if prob is not None else None
    return seg


def split_segment(
    segment: dict[str, Any],
    max_segment_sec: float,
    target_segment_sec: float,
    min_segment_sec: float,
    default_language: str,
    english_min_prob: float,
    keep_other_languages: bool,
    other_lang_min_prob: float,
) -> list[dict[str, Any]]:
    words = cleanup_words(segment.get("words") or [])
    if not words:
        fallback = deepcopy(segment)
        start = safe_float(fallback.get("start"))
        end = safe_float(fallback.get("end"))
        if start is None or end is None or end < start:
            return []
        fallback["start"] = round(start, 3)
        fallback["end"] = round(end, 3)
        fallback["text"] = (fallback.get("text") or "").strip()
        fallback["words"] = []
        if "speaker" not in fallback:
            fallback["speaker"] = None
        raw_lang = fallback.get("detected_language")
        raw_prob = safe_float(fallback.get("detected_language_probability"))
        lang, prob = normalize_language(
            raw_lang,
            raw_prob,
            fallback["text"],
            default_language,
            english_min_prob,
            keep_other_languages,
            other_lang_min_prob,
        )
        fallback["detected_language"] = lang
        fallback["detected_language_probability"] = round(prob, 4) if prob is not None else None
        return [fallback]

    fill_word_speakers(words, segment.get("speaker"))

    total_dur = words[-1]["end"] - words[0]["start"]
    if total_dur <= max_segment_sec:
        return [
            build_segment_from_words(
                words,
                segment,
                default_language,
                english_min_prob,
                keep_other_languages,
                other_lang_min_prob,
            )
        ]

    chunks: list[list[dict[str, Any]]] = []
    i = 0
    n = len(words)

    while i < n:
        start_t = words[i]["start"]
        hard_end_t = start_t + max_segment_sec
        target_end_t = start_t + target_segment_sec

        last_within = None
        punct_best = None
        punct_best_score = None
        near_best = None
        near_best_score = None

        k = i
        while k < n and words[k]["end"] <= hard_end_t + 1e-6:
            last_within = k
            dur = words[k]["end"] - start_t
            if dur >= min_segment_sec:
                score = abs(words[k]["end"] - target_end_t)
                if near_best is None or score < near_best_score:
                    near_best = k
                    near_best_score = score
                if is_sentence_end_token(words[k]["word"]):
                    if punct_best is None or score < punct_best_score:
                        punct_best = k
                        punct_best_score = score
            k += 1

        if punct_best is not None:
            split_idx = punct_best
        elif near_best is not None:
            split_idx = near_best
        elif last_within is not None:
            split_idx = last_within
        else:
            split_idx = i

        if split_idx < i:
            split_idx = i
        chunk = words[i : split_idx + 1]
        chunks.append(chunk)
        i = split_idx + 1

    # Merge tiny tail if possible.
    if len(chunks) >= 2:
        last = chunks[-1]
        prev = chunks[-2]
        last_dur = last[-1]["end"] - last[0]["start"]
        merged_dur = last[-1]["end"] - prev[0]["start"]
        if last_dur < (min_segment_sec * 0.6) and merged_dur <= (max_segment_sec * 1.15):
            chunks[-2] = prev + last
            chunks.pop()

    out: list[dict[str, Any]] = []
    for cw in chunks:
        if not cw:
            continue
        out.append(
            build_segment_from_words(
                cw,
                segment,
                default_language,
                english_min_prob,
                keep_other_languages,
                other_lang_min_prob,
            )
        )
    return out


def fill_missing_segment_speakers(segments: list[dict[str, Any]]) -> int:
    filled = 0

    # First pass: use own words majority.
    for seg in segments:
        if seg.get("speaker"):
            continue
        spk = majority_speaker_from_words(seg.get("words") or [])
        if spk:
            seg["speaker"] = spk
            filled += 1

    # Second pass: nearest neighbor segment speaker.
    for i, seg in enumerate(segments):
        if seg.get("speaker"):
            continue
        prev = None
        for j in range(i - 1, -1, -1):
            if segments[j].get("speaker"):
                prev = segments[j]["speaker"]
                break
        nxt = None
        for j in range(i + 1, len(segments)):
            if segments[j].get("speaker"):
                nxt = segments[j]["speaker"]
                break
        if prev:
            seg["speaker"] = prev
            filled += 1
        elif nxt:
            seg["speaker"] = nxt
            filled += 1
    return filled


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_postprocessed{input_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else default_output_path(input_path)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected dict-style WhisperX JSON with top-level `segments`.")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Missing `segments` list in input JSON.")

    new_segments: list[dict[str, Any]] = []
    split_count = 0
    word_speaker_filled = 0
    language_forced = 0

    for seg in segments:
        seg_words = cleanup_words(seg.get("words") or [])
        word_speaker_filled += fill_word_speakers(seg_words, seg.get("speaker"))
        seg_copy = deepcopy(seg)
        seg_copy["words"] = seg_words

        split_out = split_segment(
            segment=seg_copy,
            max_segment_sec=args.max_segment_sec,
            target_segment_sec=args.target_segment_sec,
            min_segment_sec=args.min_segment_sec,
            default_language=args.default_language,
            english_min_prob=args.english_min_prob,
            keep_other_languages=args.keep_other_languages,
            other_lang_min_prob=args.other_lang_min_prob,
        )
        if len(split_out) > 1:
            split_count += 1
        for s in split_out:
            raw_lang = (seg.get("detected_language") or "").strip().lower() if seg.get("detected_language") else None
            if raw_lang and s.get("detected_language") != raw_lang:
                language_forced += 1
        new_segments.extend(split_out)

    segment_speaker_filled = fill_missing_segment_speakers(new_segments)

    # Second pass: after segment-level speaker is finalized, fill any remaining
    # missing word speakers from the segment speaker.
    word_speaker_filled_second_pass = 0
    for seg in new_segments:
        words = seg.get("words") or []
        if not words:
            continue
        word_speaker_filled_second_pass += fill_word_speakers(words, seg.get("speaker"))

    if args.sort_output:
        new_segments.sort(key=lambda s: (safe_float(s.get("start")) or 0.0, safe_float(s.get("end")) or 0.0))

    new_word_segments: list[dict[str, Any]] = []
    for seg in new_segments:
        for w in seg.get("words") or []:
            new_word_segments.append(deepcopy(w))
    if args.sort_output:
        new_word_segments.sort(key=lambda w: (safe_float(w.get("start")) or 0.0, safe_float(w.get("end")) or 0.0))

    out_data = deepcopy(data)
    out_data["segments"] = new_segments
    out_data["word_segments"] = new_word_segments

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Segments: {len(segments)} -> {len(new_segments)}")
    print(f"[INFO] Word segments: {len(data.get('word_segments', []))} -> {len(new_word_segments)}")
    print(f"[INFO] Split source segments (>max): {split_count}")
    print(f"[INFO] Filled missing word speaker labels: {word_speaker_filled}")
    print(f"[INFO] Filled missing word speaker labels (second pass): {word_speaker_filled_second_pass}")
    print(f"[INFO] Filled missing segment speaker labels: {segment_speaker_filled}")
    print(f"[INFO] Language labels forced to default: {language_forced}")


if __name__ == "__main__":
    main()
