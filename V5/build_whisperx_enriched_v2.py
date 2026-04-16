#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ENTITY_EDGE_CHARS = " \t\r\n`'\".,;:!?()[]{}<>\u00ab\u00bb\u2018\u2019\u201c\u201d\u2039\u203a"
ENTITY_STOPWORDS = {
    # English function words / pronouns
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "their",
    "our",
    "its",
    # German function words / pronouns
    "und",
    "oder",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einer",
    "einem",
    "einen",
    "zu",
    "im",
    "am",
    "mit",
    "von",
    "ist",
    "sind",
    "war",
    "waren",
    "ich",
    "du",
    "er",
    "sie",
    "wir",
    "ihr",
    "es",
    "man",
    "mich",
    "dich",
    "mir",
    "sich",
}

# User-requested types for single-token stopword filtering.
STOPWORD_FILTER_TYPES = {"PER", "ORG", "LOC", "WORK_OF_ART"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Build CineMinds WhisperX enriched v2 JSON.")
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", default=None)
    p.add_argument("--output_suffix", default="_enriched_v2")
    p.add_argument("--glossary_tags_json", default=None)
    p.add_argument("--entity_fields", default="entities_v3,entities,named_entities")
    p.add_argument("--max_keywords", type=int, default=8)
    p.add_argument("--max_tags_per_chunk", type=int, default=40)
    p.add_argument(
        "--ner_taxonomy_json",
        default=str(root / "schemas" / "cineminds_ner_taxonomy.json"),
    )
    return p.parse_args()


def ep(path_str: str) -> Path:
    return Path(os.path.expanduser(path_str)).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object: {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ns(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def nk(text: Any) -> str:
    return re.sub(r"\W+", "", ns(text).casefold())


def clean_entity_text(text: Any) -> str:
    t = ns(text)
    if not t:
        return ""
    start = 0
    end = len(t)
    while start < end and t[start] in ENTITY_EDGE_CHARS:
        start += 1
    while end > start and t[end - 1] in ENTITY_EDGE_CHARS:
        end -= 1
    return t[start:end].strip()


def trim_entity_span(
    seg_text: str,
    start_char: int,
    end_char: int,
    fallback_text: str,
) -> tuple[str, int, int]:
    try:
        s = int(start_char)
        e = int(end_char)
    except Exception:
        return clean_entity_text(fallback_text), -1, -1
    if s < 0 or e <= s or e > len(seg_text):
        return clean_entity_text(fallback_text), -1, -1
    while s < e and seg_text[s] in ENTITY_EDGE_CHARS:
        s += 1
    while e > s and seg_text[e - 1] in ENTITY_EDGE_CHARS:
        e -= 1
    return clean_entity_text(seg_text[s:e]), s, e


def canonical_entity_key(ntype: str, text: str) -> str:
    return f"{ntype}|{nk(clean_entity_text(text))}"


def choose_preferred_subtype(current: str, candidate: str, default_subtype: str) -> str:
    cur = ns(current) or default_subtype
    cand = ns(candidate) or default_subtype
    if cur == default_subtype and cand != default_subtype:
        return cand
    return cur


def choose_preferred_text(current: str, candidate: str) -> str:
    cur = clean_entity_text(current)
    cand = clean_entity_text(candidate)
    if not cur:
        return cand
    if not cand:
        return cur
    cur_words = len(cur.split())
    cand_words = len(cand.split())
    if cand_words > cur_words:
        return cand
    if cand_words == cur_words and len(cand) > len(cur):
        return cand
    return cur


def text_from_word_indices(word_rows: list[dict[str, Any]], indices: list[int]) -> str:
    if not indices:
        return ""
    return clean_entity_text(" ".join([ns(word_rows[j].get("form", "")) for j in indices]).strip())


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    t = ns(text)
    if not t:
        return []
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for part in SENT_SPLIT_RE.split(t):
        s = ns(part)
        if not s:
            continue
        pos = t.find(s, cursor)
        if pos < 0:
            pos = cursor
        out.append((s, pos, pos + len(s)))
        cursor = pos + len(s)
    return out or [(t, 0, len(t))]


def char_to_time(seg: dict[str, Any], char_pos: int) -> float:
    text = ns(seg.get("text", ""))
    st = float(seg.get("start", 0.0) or 0.0)
    ed = float(seg.get("end", st) or st)
    if ed < st:
        ed = st
    if not text or ed <= st:
        return round(st, 3)
    ratio = max(0.0, min(1.0, char_pos / max(1, len(text))))
    return round(st + (ed - st) * ratio, 3)


def align_words(seg_text: str, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = ns(seg_text)
    out: list[dict[str, Any]] = []
    cursor = 0
    for i, w in enumerate(words):
        form = ns(w.get("word", w.get("text", "")))
        key = form.strip("`'\".,;:!?()[]{}<>") or form
        s = t.find(key, cursor) if key else -1
        if s < 0 and key:
            s = t.find(key)
        if s < 0:
            s = cursor
            e = min(len(t), s + len(key))
        else:
            e = s + len(key)
        cursor = max(cursor, e)
        out.append(
            {
                "index": i,
                "form": form,
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker"),
                "char_start": s,
                "char_end": e,
                "ner_type": None,
                "ner_subtype": None,
                "ner_id": None,
                "_score": -1.0,
            }
        )
    return out


def norm_label(label: Any, tax: dict[str, Any]) -> str:
    raw = ns(label).upper()
    cmap = tax.get("canonical_type_map", {})
    if raw in cmap:
        return str(cmap[raw])
    return raw or "MISC"


def infer_subtype(ntype: str, ent_text: str, seg_text: str, tax: dict[str, Any]) -> str:
    default = str(tax.get("default_subtype", "unspecified"))
    sub = tax.get("types", {}).get(ntype, {}).get("subtypes", {})
    if not isinstance(sub, dict):
        return default
    haystack = f"{ent_text} {seg_text}".casefold()
    for sname, cfg in sub.items():
        kws = cfg.get("keywords", []) if isinstance(cfg, dict) else []
        if any(ns(k).casefold() in haystack for k in kws):
            return str(sname)
    return default


def entity_tokens(text: str) -> list[str]:
    parts = re.split(r"\s+", ns(text))
    return [nk(p) for p in parts if nk(p)]


def should_drop_single_token_stopword_entity(entity_text: str, ntype: str) -> bool:
    if ntype not in STOPWORD_FILTER_TYPES:
        return False
    toks = entity_tokens(entity_text)
    if len(toks) != 1:
        return False
    return toks[0] in ENTITY_STOPWORDS


def parse_entities(seg: dict[str, Any], fields: list[str], tax: dict[str, Any]) -> list[dict[str, Any]]:
    seg_text = ns(seg.get("text", ""))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for field_name in fields:
        arr = seg.get(field_name)
        if not isinstance(arr, list):
            continue
        for e in arr:
            if not isinstance(e, dict):
                continue
            ntype = norm_label(e.get("label", e.get("type", "MISC")), tax)
            raw_text = ns(e.get("text", e.get("entity", e.get("name", ""))))
            if not raw_text:
                continue
            text = clean_entity_text(raw_text)
            if not text:
                continue
            if should_drop_single_token_stopword_entity(text, ntype):
                continue
            s = e.get("start_pos", e.get("start_char"))
            t = e.get("end_pos", e.get("end_char"))
            text_from_span, span_start, span_end = trim_entity_span(seg_text, s, t, text)
            if span_start < 0:
                s = seg_text.find(text)
                t = s + len(text) if s >= 0 else -1
                text_from_span, span_start, span_end = trim_entity_span(seg_text, s, t, text)
            if span_start < 0 or span_end <= span_start or not text_from_span:
                continue
            subtype = ns(e.get("subtype", e.get("ner_subtype", ""))) or infer_subtype(
                ntype, text_from_span, seg_text, tax
            )
            key = (nk(text_from_span), ntype, int(span_start), int(span_end))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": text_from_span,
                    "type": ntype,
                    "subtype": subtype,
                    "start_char": span_start,
                    "end_char": span_end,
                }
            )
    out.sort(key=lambda x: (x["start_char"], x["end_char"]))
    return out


def collect_chunk_keywords(segments: list[dict[str, Any]], max_keywords: int) -> dict[str, list[str]]:
    by_chunk: dict[str, Counter] = {}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        cid = str(seg.get("chunk_id", f"segment_{i}"))
        counter = by_chunk.setdefault(cid, Counter())
        tw = seg.get("topic_words", [])
        if isinstance(tw, list):
            for w in tw:
                token = ns(w)
                if token:
                    counter[token] += 1
    out: dict[str, list[str]] = {}
    for cid, counter in by_chunk.items():
        out[cid] = [w for w, _ in counter.most_common(max_keywords)]
    return out


def topic_name(keywords: list[str]) -> str:
    if not keywords:
        return "General Discussion"
    joined = " ".join(keywords).casefold()
    if any(x in joined for x in ["camera", "shot", "frame", "angle", "lens"]):
        return "Cinematography"
    if any(x in joined for x in ["edit", "montage", "cut"]):
        return "Editing"
    if any(x in joined for x in ["sound", "voice", "music", "audio"]):
        return "Sound Design"
    if any(x in joined for x in ["actor", "acting", "schauspiel"]):
        return "Acting Practice"
    return " / ".join([ns(k).title() for k in keywords[:3]])


def load_tags(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = load_json(path)
    arr = data.get("tags", [])
    if not isinstance(arr, list):
        return []
    out: list[dict[str, Any]] = []
    for t in arr:
        if not isinstance(t, dict):
            continue
        label = ns(t.get("label", ""))
        cat = ns(t.get("category", "")).lower()
        if label and cat in {"term", "field", "register"}:
            out.append(
                {
                    **t,
                    "tag_id": t.get("tag_id"),
                    "label": label,
                    "category": cat,
                    "_label_lower": label.casefold(),
                }
            )
    return out


def add_tag_hit(tag: dict[str, Any], hits: list[dict[str, Any]], seen: set[tuple[str, str]]) -> None:
    label = ns(tag.get("label", ""))
    category = ns(tag.get("category", "")).lower()
    if not label or category not in {"term", "field", "register"}:
        return
    key = (category, label.casefold())
    if key in seen:
        return
    seen.add(key)
    hits.append({k: v for k, v in tag.items() if not str(k).startswith("_")})


def match_chunk_tags(chunk_text: str, tags: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    text = ns(chunk_text).casefold()
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for tag in tags:
        label = ns(tag.get("label", ""))
        if len(label) < 3:
            continue
        pat = re.compile(rf"(?<!\w){re.escape(label.casefold())}(?!\w)")
        if not pat.search(text):
            continue
        add_tag_hit(tag, hits, seen)
        if tag.get("category") != "term":
            continue
        field = ns(tag.get("field", ""))
        register = ns(tag.get("register", ""))
        exp_field = ns(tag.get("exp_field", ""))
        if field:
            add_tag_hit(
                {
                    "tag_id": f"{tag.get('term_id') or tag.get('tag_id')}_field",
                    "label": field,
                    "category": "field",
                    "field": field,
                    "exp_field": exp_field,
                },
                hits,
                seen,
            )
        if register:
            add_tag_hit(
                {
                    "tag_id": f"{tag.get('term_id') or tag.get('tag_id')}_register",
                    "label": register,
                    "category": "register",
                    "register": register,
                },
                hits,
                seen,
            )
    hits.sort(key=lambda x: (x.get("category", ""), ns(x.get("label", "")).casefold()))
    return hits[:top_k]


def build_segment_entity_cache(
    segments: list[dict[str, Any]],
    fields: list[str],
    tax: dict[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    default_subtype = str(tax.get("default_subtype", "unspecified"))
    segment_entities: dict[int, list[dict[str, Any]]] = {}
    entity_profiles: dict[str, dict[str, Any]] = {}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        entities = parse_entities(seg, fields, tax)
        segment_entities[i] = entities
        for ent in entities:
            key = canonical_entity_key(ent["type"], ent["text"])
            if not key.endswith("|"):
                profile = entity_profiles.setdefault(
                    key,
                    {
                        "type": ent["type"],
                        "subtype": default_subtype,
                        "text": clean_entity_text(ent["text"]),
                    },
                )
                profile["subtype"] = choose_preferred_subtype(
                    str(profile.get("subtype", default_subtype)),
                    str(ent.get("subtype", default_subtype)),
                    default_subtype,
                )
                profile["text"] = choose_preferred_text(str(profile.get("text", "")), ent["text"])
    return segment_entities, entity_profiles


def main() -> None:
    args = parse_args()
    in_path = ep(args.input_json)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path = ep(args.output_json) if args.output_json else in_path.with_name(
        f"{in_path.stem}{args.output_suffix}{in_path.suffix}"
    )
    tax_path = ep(args.ner_taxonomy_json)
    if not tax_path.exists():
        raise FileNotFoundError(tax_path)
    tax = load_json(tax_path)
    src = load_json(in_path)
    segs = src.get("segments", [])
    if not isinstance(segs, list):
        raise ValueError("Missing top-level segments list.")

    tag_path = ep(args.glossary_tags_json) if args.glossary_tags_json else None
    tag_dict = load_tags(tag_path)
    chunk_keywords = collect_chunk_keywords(segs, args.max_keywords)
    chunk_texts: dict[str, str] = {}
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            continue
        cid = str(seg.get("chunk_id", f"segment_{i}"))
        chunk_texts[cid] = (chunk_texts.get(cid, "") + " " + ns(seg.get("text", ""))).strip()
    chunk_tags = {
        cid: match_chunk_tags(text, tag_dict, args.max_tags_per_chunk)
        for cid, text in chunk_texts.items()
    }

    sentence_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    entity_map: dict[str, str] = {}
    entity_next = 1
    sent_next = 1
    fields = [ns(x) for x in args.entity_fields.split(",") if ns(x)]
    parsed_entities_by_segment, entity_profiles = build_segment_entity_cache(segs, fields, tax)

    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            continue
        text = ns(seg.get("text", ""))
        if not text:
            continue
        cid = seg.get("chunk_id", f"segment_{i}")
        ckey = str(cid)
        kws = chunk_keywords.get(ckey, [])
        tname = topic_name(kws)
        tags = chunk_tags.get(ckey, [])

        words_raw = seg.get("words", [])
        words = [w for w in words_raw if isinstance(w, dict)]
        word_rows = align_words(text, words)
        entities = parsed_entities_by_segment.get(i, [])

        for ent in entities:
            source_key = canonical_entity_key(ent["type"], ent["text"])
            profile = entity_profiles.get(
                source_key,
                {
                    "type": ent["type"],
                    "subtype": ent["subtype"],
                    "text": clean_entity_text(ent["text"]),
                },
            )
            touched = [
                j
                for j, w in enumerate(word_rows)
                if not (w["char_end"] <= ent["start_char"] or w["char_start"] >= ent["end_char"])
            ]
            if not touched:
                continue
            aligned_text = text_from_word_indices(word_rows, touched)
            key = canonical_entity_key(profile["type"], aligned_text or ent["text"])
            nid = entity_map.get(key)
            if nid is None:
                existing_nid = entity_map.get(source_key)
                if existing_nid is not None:
                    nid = existing_nid
                    entity_map[key] = nid
                else:
                    nid = f"ner_{entity_next:04d}"
                    entity_map[key] = nid
                    entity_map[source_key] = nid
                    entity_next += 1
            entity_map[source_key] = nid
            entity_map[key] = nid
            for j in touched:
                word_rows[j]["ner_type"] = profile["type"]
                word_rows[j]["ner_subtype"] = profile["subtype"]
                word_rows[j]["ner_id"] = nid

        for stext, s_char, e_char in split_sentences(text):
            sidx = [
                j
                for j, w in enumerate(word_rows)
                if not (w["char_end"] <= s_char or w["char_start"] >= e_char)
            ]
            s_words = [
                {
                    "form": word_rows[j]["form"],
                    "start": word_rows[j]["start"],
                    "end": word_rows[j]["end"],
                    "speaker": word_rows[j]["speaker"],
                    "ner_type": word_rows[j]["ner_type"],
                    "ner_subtype": word_rows[j]["ner_subtype"],
                    "ner_id": word_rows[j]["ner_id"],
                }
                for j in sidx
            ]
            sid = f"sent_{sent_next:06d}"
            sent_next += 1
            spk_candidates = [ns(w.get("speaker")) for w in s_words if ns(w.get("speaker"))]
            spk = ns(seg.get("speaker")) or (
                Counter(spk_candidates).most_common(1)[0][0] if spk_candidates else None
            )
            row = {
                "sentence_id": sid,
                "chunk_id": cid,
                "segment_index": i,
                "text": stext,
                "start": char_to_time(seg, s_char),
                "end": char_to_time(seg, e_char),
                "speaker": spk,
                "bert_keywords": kws,
                "bert_topic_name": tname,
                "tags": tags,
                "words": s_words,
            }
            sentence_rows.append(row)

            run_id = None
            run_words: list[dict[str, Any]] = []
            for w in s_words + [{"ner_id": None}]:
                if w.get("ner_id") and w.get("ner_id") == run_id:
                    run_words.append(w)
                    continue
                if run_id and run_words:
                    entity_text = clean_entity_text(
                        " ".join([ns(x.get("form", "")) for x in run_words]).strip()
                    )
                    if not entity_text:
                        entity_text = " ".join([ns(x.get("form", "")) for x in run_words]).strip()
                    entity_rows.append(
                        {
                            "id": run_id,
                            "text": entity_text,
                            "type": run_words[0].get("ner_type"),
                            "subtype": run_words[0].get("ner_subtype"),
                            "start": min(float(x.get("start") or row["start"]) for x in run_words),
                            "end": max(float(x.get("end") or row["end"]) for x in run_words),
                            "chunk_id": cid,
                            "sentence_id": sid,
                            "segment_index": i,
                        }
                    )
                run_id = w.get("ner_id")
                run_words = [w] if run_id else []

    dedup_entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float, str]] = set()
    for e in entity_rows:
        k = (str(e["id"]), str(e["sentence_id"]), float(e["start"]), float(e["end"]), nk(e["text"]))
        if k in seen:
            continue
        seen.add(k)
        dedup_entities.append(e)

    out = {
        "format_version": "cineminds_whisperx_enriched_v2",
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_file": str(in_path),
        "transcript": {"segments": segs, "word_segments": src.get("word_segments", [])},
        "chunk_tags": [{"chunk_id": k, "tags": v} for k, v in chunk_tags.items()],
        "sentences": sentence_rows,
        "entities": dedup_entities,
        "meta": {
            "task": "build_whisperx_enriched_v2",
            "sentence_count": len(sentence_rows),
            "entity_mention_count": len(dedup_entities),
            "chunk_count": len(chunk_texts),
            "entity_fields": fields,
            "has_glossary_tags": bool(tag_dict),
        },
    }
    save_json(out_path, out)
    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Sentences: {len(sentence_rows)}")
    print(f"[INFO] Entity mentions: {len(dedup_entities)}")


if __name__ == "__main__":
    main()
