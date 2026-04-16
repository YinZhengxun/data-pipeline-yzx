#!/usr/bin/env python3
"""
CineMinds LLM NER v3 (hardened baseline).

Compared with v2:
- Better surface-form filtering (reduced false drops on valid punctuated entities).
- Occurrence-aware span alignment for repeated mentions in a segment.
- Explicit handling of unknown labels (skip instead of defaulting to MISC).
- Label-aware score filtering + score-sorted truncation.
- Isolated write field by default (`entities_v3`) to avoid mixing with prior runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError("Missing dependency: openai. Install with `pip install openai`.") from exc


DEFAULT_INPUT_JSON = "~/cineminds/martin_topics_v2/final_whisper_style_enriched_rechunked.json"
DEFAULT_OUTPUT_SUFFIX = "_llm_ner_v3"
DEFAULT_MODEL = "MiniMax-M2.7"

# Fallbacks for server convenience.
HARDCODED_BASE_URL = "https://api.minimax.io/v1"
HARDCODED_API_KEY = ""

ALLOWED_LABELS = {"PER", "ORG", "LOC", "WORK", "EVT", "DATE", "MISC"}
LABEL_ALIAS = {
    "PER": "PER",
    "PERSON": "PER",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "PLACE": "LOC",
    "GPE": "LOC",
    "WORK": "WORK",
    "WORK_OF_ART": "WORK",
    "TITLE": "WORK",
    "FILM": "WORK",
    "MOVIE": "WORK",
    "SERIES": "WORK",
    "EVT": "EVT",
    "EVENT": "EVT",
    "DATE": "DATE",
    "TIME": "DATE",
    "YEAR": "DATE",
    "MISC": "MISC",
    "PRODUCT": "MISC",
    "NORP": "MISC",
}

NER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["segment_entities"],
    "properties": {
        "segment_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segment_local_index", "entities"],
                "properties": {
                    "segment_local_index": {"type": "integer", "minimum": 0},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "label", "score"],
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string"},
                                "score": {"type": "number"},
                                "occurrence": {"type": "integer", "minimum": 1},
                            },
                        },
                    },
                },
            },
        }
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM NER v3 for whisper-style JSON (recall-focused + safer postprocessing)."
    )
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_suffix", default=DEFAULT_OUTPUT_SUFFIX)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--base_url_env", default="OPENAI_BASE_URL")

    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_backoff_sec", type=float, default=2.0)
    parser.add_argument("--checkpoint_every", type=int, default=10)

    parser.add_argument("--context_window", type=int, default=1, help="Neighbor segments used as context")
    parser.add_argument("--max_segment_chars", type=int, default=500)
    parser.add_argument("--max_entities_per_segment", type=int, default=10)
    parser.add_argument(
        "--entity_field",
        default="entities_v3",
        help="Segment field to write v3 entities into (default: entities_v3)",
    )

    parser.add_argument("--min_score", type=float, default=0.50)
    parser.add_argument("--min_score_misc", type=float, default=0.60)
    parser.add_argument("--min_score_date", type=float, default=0.40)
    parser.add_argument(
        "--drop_low_confidence",
        action="store_true",
        help="Drop entities below label-aware thresholds",
    )

    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument(
        "--merge_with_existing",
        action="store_true",
        help="Merge with existing list in --entity_field instead of replacing it",
    )
    parser.add_argument("--no_schema_guard", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def expand_path(path_str: str) -> Path:
    return Path(os.path.expanduser(path_str)).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def derive_output_path(input_path: Path, suffix: str) -> Path:
    suffix = suffix or DEFAULT_OUTPUT_SUFFIX
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def chunk_sort_key(chunk_id: Any) -> tuple[int, Any]:
    try:
        return (0, int(str(chunk_id)))
    except (TypeError, ValueError):
        return (1, str(chunk_id))


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return str(args.api_key).strip()
    env_value = os.getenv(args.api_key_env, "").strip()
    if env_value:
        return env_value
    if HARDCODED_API_KEY.strip():
        return HARDCODED_API_KEY.strip()
    raise EnvironmentError(
        f"Missing API key. Use --api_key, set {args.api_key_env}, or set HARDCODED_API_KEY."
    )


def resolve_base_url(args: argparse.Namespace) -> str | None:
    if args.base_url:
        return str(args.base_url).strip()
    env_value = os.getenv(args.base_url_env, "").strip()
    if env_value:
        return env_value
    if HARDCODED_BASE_URL.strip():
        return HARDCODED_BASE_URL.strip()
    return None


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate_text(text: str, max_chars: int) -> str:
    t = normalize_ws(text)
    if len(t) <= max_chars:
        return t
    cut = t[: max_chars - 3].rstrip()
    ws = cut.rfind(" ")
    if ws > max_chars * 0.65:
        cut = cut[:ws].rstrip()
    return cut + "..."


def build_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Input JSON must contain a non-empty 'segments' list")

    grouped: dict[str, dict[str, Any]] = {}
    for seg_idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        chunk_id = seg.get("chunk_id")
        if chunk_id is None:
            chunk_id = f"segment_{seg_idx}"
        key = str(chunk_id)

        item = grouped.get(key)
        if item is None:
            item = {
                "chunk_id": chunk_id,
                "segment_refs": [],      # list[(global_idx, local_idx)]
                "segment_payload": [],   # list[dict]
            }
            grouped[key] = item

        local_idx = len(item["segment_payload"])
        text = normalize_ws(str(seg.get("text", "") or ""))
        item["segment_refs"].append((seg_idx, local_idx))
        item["segment_payload"].append(
            {
                "segment_local_index": local_idx,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": seg.get("speaker"),
                "text": text,
            }
        )

    out: list[dict[str, Any]] = []
    for k in sorted(grouped.keys(), key=lambda x: chunk_sort_key(grouped[x]["chunk_id"])):
        out.append(grouped[k])
    return out


def build_messages(
    chunk: dict[str, Any],
    context_window: int,
    max_segment_chars: int,
) -> list[dict[str, str]]:
    segs = chunk["segment_payload"]
    payload: list[dict[str, Any]] = []

    for i, row in enumerate(segs):
        prev_items: list[str] = []
        next_items: list[str] = []
        for j in range(max(0, i - context_window), i):
            prev_items.append(truncate_text(str(segs[j]["text"]), max(80, max_segment_chars // 2)))
        for j in range(i + 1, min(len(segs), i + 1 + context_window)):
            next_items.append(truncate_text(str(segs[j]["text"]), max(80, max_segment_chars // 2)))

        payload.append(
            {
                "segment_local_index": row["segment_local_index"],
                "start": row.get("start"),
                "end": row.get("end"),
                "speaker": row.get("speaker"),
                "text": truncate_text(str(row.get("text", "")), max_segment_chars),
                "context_prev": prev_items,
                "context_next": next_items,
            }
        )

    system_prompt = (
        "You are a NER annotator for German/Swiss German interview ASR transcripts in film studies.\n"
        "Goal: maximize useful recall while staying grounded.\n"
        "Rules:\n"
        "1) You may use neighboring context to disambiguate.\n"
        "2) Output entity text must match surface text in the target segment; "
        "case-only differences are acceptable.\n"
        "3) If a mention is plausible but somewhat noisy, keep it with medium confidence (0.50-0.75).\n"
        "4) Label set: PER, ORG, LOC, WORK, EVT, DATE, MISC.\n"
        "5) Extract DATE only when specific and semantically relevant.\n"
        "6) If same entity appears multiple times in one segment, include occurrence (1-based).\n"
        "7) Return JSON only."
    )

    user_prompt = (
        f"chunk_id: {chunk['chunk_id']}\n"
        "Return JSON schema:\n"
        "{\n"
        '  "segment_entities": [\n'
        "    {\n"
        '      "segment_local_index": 0,\n'
        '      "entities": [\n'
        '        {"text":"Johnny Depp","label":"PER","score":0.97,"occurrence":1},\n'
        '        {"text":"Pirates of the Caribbean","label":"WORK","score":0.95}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Segments with context:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def extract_text_from_completion(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    out.append(t)
        return "\n".join(out).strip()
    return ""


def extract_first_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw_text)):
        ch = raw_text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : i + 1]
    return ""


def parse_json_response(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    cand = extract_first_json_object(text)
    if not cand:
        raise ValueError("Could not parse response JSON")
    obj = json.loads(cand)
    if not isinstance(obj, dict):
        raise ValueError("Top-level response must be an object")
    return obj


def normalize_label(label: Any) -> str:
    k = str(label or "").strip().upper()
    if k in LABEL_ALIAS:
        mapped = LABEL_ALIAS[k]
        return mapped if mapped in ALLOWED_LABELS else ""
    if k in ALLOWED_LABELS:
        return k
    return ""


def normalize_score(value: Any) -> float:
    try:
        s = float(value)
    except (TypeError, ValueError):
        s = 0.70
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return round(s, 4)


def clean_entity_text(text: str) -> str:
    t = normalize_ws(text)
    t = t.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return t


def find_all_spans(
    segment_text: str,
    entity_text: str,
    case_sensitive: bool,
) -> list[tuple[int, int]]:
    if not entity_text:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE
    return [(m.start(), m.end()) for m in re.finditer(re.escape(entity_text), segment_text, flags=flags)]


def select_span(
    segment_text: str,
    entity_text: str,
    occurrence: int | None,
    used_spans: set[tuple[int, int]],
) -> tuple[int, int]:
    if not entity_text:
        return (-1, -1)

    exact = find_all_spans(segment_text, entity_text, case_sensitive=True)
    fallback = find_all_spans(segment_text, entity_text, case_sensitive=False) if not exact else []
    candidates = exact if exact else fallback
    if not candidates:
        return (-1, -1)

    if isinstance(occurrence, int) and occurrence >= 1:
        idx = occurrence - 1
        if idx < len(candidates):
            span = candidates[idx]
            if span not in used_spans:
                return span

    for span in candidates:
        if span not in used_spans:
            return span

    return candidates[0]


def label_min_score(label: str, args: argparse.Namespace) -> float:
    if label == "MISC":
        return args.min_score_misc
    if label == "DATE":
        return args.min_score_date
    return args.min_score


def looks_bad_surface(text: str, label: str) -> bool:
    if not text:
        return True

    t = text.strip()
    if len(t) <= 1:
        return True

    if label == "DATE" and re.fullmatch(r"\d{3,4}", t):
        return False

    # More specific DATE variants like 12.03.1989 / 12-03-89 / 2024/05.
    if label == "DATE" and re.fullmatch(r"\d{1,4}([./-]\d{1,2}){0,2}", t):
        return False

    # Non-date pure numbers are usually noise.
    if label != "DATE" and re.fullmatch(r"\d+", t):
        return True

    # Reject strings made only of symbols.
    if re.fullmatch(r"[_\W]+", t):
        return True

    # Require at least one letter for non-DATE labels.
    if label != "DATE" and not re.search(r"[^\W\d_]", t):
        return True

    # Allow common punctuation in names/titles; reject heavy odd-symbol noise only.
    odd = len(re.findall(r"[^\w\s\-'.,:&/]", t))
    if odd >= 2 and (odd / max(1, len(t))) > 0.15:
        return True

    return False


def dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for e in entities:
        key = (e.get("text"), e.get("label"), e.get("start_pos"), e.get("end_pos"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def empty_chunk_entities(chunk: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {int(row["segment_local_index"]): [] for row in chunk["segment_payload"]}


def normalize_annotation(
    parsed: dict[str, Any],
    chunk: dict[str, Any],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    out = empty_chunk_entities(chunk)

    segment_payload = chunk["segment_payload"]
    segment_map: dict[int, dict[str, Any]] = {
        int(row["segment_local_index"]): row for row in segment_payload
    }

    rows = parsed.get("segment_entities")
    if rows is None:
        rows = parsed.get("segments")
    if not isinstance(rows, list):
        return out

    for row in rows:
        if not isinstance(row, dict):
            continue

        local_idx = row.get("segment_local_index")
        if local_idx is None:
            local_idx = row.get("segment_index", row.get("local_index"))
        try:
            local_idx = int(local_idx)
        except (TypeError, ValueError):
            continue
        if local_idx not in segment_map:
            continue

        seg_text = str(segment_map[local_idx].get("text", "") or "")
        ents = row.get("entities")
        if not isinstance(ents, list):
            continue

        normalized_entities: list[dict[str, Any]] = []
        used_spans: set[tuple[int, int]] = set()
        for ent in ents:
            if not isinstance(ent, dict):
                continue
            raw_text = ent.get("text", ent.get("entity", ent.get("name")))
            text = clean_entity_text(str(raw_text or ""))
            if not text:
                continue

            label = normalize_label(ent.get("label"))
            if not label:
                continue
            score = normalize_score(ent.get("score"))
            occurrence_raw = ent.get("occurrence")
            occurrence: int | None
            if occurrence_raw is None:
                occurrence = None
            else:
                try:
                    occurrence = int(occurrence_raw)
                except (TypeError, ValueError):
                    occurrence = None

            start_pos, end_pos = select_span(
                segment_text=seg_text,
                entity_text=text,
                occurrence=occurrence,
                used_spans=used_spans,
            )
            if start_pos < 0:
                continue

            if looks_bad_surface(text, label):
                continue

            if args.drop_low_confidence and score < label_min_score(label, args):
                continue

            used_spans.add((start_pos, end_pos))
            normalized_entities.append(
                {
                    "text": text,
                    "label": label,
                    "score": score,
                    "start_pos": int(start_pos),
                    "end_pos": int(end_pos),
                }
            )

        normalized_entities = dedupe_entities(normalized_entities)
        normalized_entities = sorted(
            normalized_entities,
            key=lambda x: (-float(x.get("score", 0.0)), int(x.get("start_pos", 0))),
        )
        out[local_idx] = normalized_entities[: max(1, args.max_entities_per_segment)]

    return out


def request_chunk_ner(
    client: OpenAI,
    chunk: dict[str, Any],
    args: argparse.Namespace,
    use_schema_guard: bool,
) -> dict[int, list[dict[str, Any]]]:
    messages = build_messages(
        chunk=chunk,
        context_window=max(0, args.context_window),
        max_segment_chars=max(120, args.max_segment_chars),
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
    }

    if use_schema_guard:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "segment_ner_response_v3",
                "strict": True,
                "schema": NER_RESPONSE_SCHEMA,
            },
        }

    completion = client.chat.completions.create(**kwargs)
    raw = extract_text_from_completion(completion.choices[0].message.content)
    parsed = parse_json_response(raw)
    return normalize_annotation(parsed=parsed, chunk=chunk, args=args)


def request_chunk_ner_with_fallback(
    client: OpenAI,
    chunk: dict[str, Any],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    if not args.no_schema_guard:
        try:
            return request_chunk_ner(client=client, chunk=chunk, args=args, use_schema_guard=True)
        except Exception:
            print(
                f"[WARN] chunk {chunk.get('chunk_id')}: schema-guard call failed, "
                "retrying without API schema guard"
            )
    return request_chunk_ner(client=client, chunk=chunk, args=args, use_schema_guard=False)


def annotate_chunk_with_retry(
    client: OpenAI,
    chunk: dict[str, Any],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    last_error: Exception | None = None
    for attempt in range(1, max(1, args.max_retries) + 1):
        try:
            return request_chunk_ner_with_fallback(client=client, chunk=chunk, args=args)
        except Exception as exc:
            last_error = exc
            if attempt < max(1, args.max_retries):
                sleep_sec = max(0.0, args.retry_backoff_sec) * (2 ** (attempt - 1))
                print(
                    f"[WARN] chunk {chunk.get('chunk_id')} failed on attempt "
                    f"{attempt}/{args.max_retries}: {exc}. Retry in {sleep_sec:.1f}s"
                )
                time.sleep(sleep_sec)
    raise RuntimeError(
        f"Chunk {chunk.get('chunk_id')} failed after {args.max_retries} attempts: {last_error}"
    )


def apply_chunk_entities_to_segments(
    data: dict[str, Any],
    chunk: dict[str, Any],
    local_entities: dict[int, list[dict[str, Any]]],
    entity_field: str,
    merge_with_existing: bool,
) -> None:
    segments = data["segments"]
    target_field = entity_field or "entities_v3"
    for global_idx, local_idx in chunk["segment_refs"]:
        seg = segments[global_idx]
        incoming = local_entities.get(local_idx, [])
        if merge_with_existing and isinstance(seg.get(target_field), list):
            existing = [e for e in seg[target_field] if isinstance(e, dict)]
            seg[target_field] = dedupe_entities(existing + incoming)
        else:
            seg[target_field] = incoming


def main() -> None:
    args = parse_args()

    input_path = expand_path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_path = expand_path(args.output_json) if args.output_json else derive_output_path(
        input_path, args.output_suffix
    )

    resume_mode = output_path.exists() and not args.overwrite_existing
    if resume_mode:
        data = load_json(output_path)
        print(f"[INFO] Resume from existing output: {output_path}")
    else:
        data = load_json(input_path)

    chunks = build_chunks(data)
    selected = chunks[args.start_chunk :] if args.start_chunk > 0 else chunks
    if args.max_chunks is not None:
        selected = selected[: args.max_chunks]

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Total chunks: {len(chunks)}")
    print(f"[INFO] Selected chunks: {len(selected)}")
    print(f"[INFO] Entity field: {args.entity_field}")
    if args.entity_field == "entities" and args.merge_with_existing:
        print(
            "[WARN] You are merging into 'entities'. This mixes sources "
            "(legacy + v3) in one field."
        )

    if args.dry_run:
        return

    api_key = resolve_api_key(args)
    base_url = resolve_base_url(args)
    client = OpenAI(api_key=api_key, base_url=base_url)

    processed_ids: set[str] = set(str(x) for x in data.get("llm_ner_v3_processed_chunk_ids", []))
    failures: list[dict[str, Any]] = list(data.get("llm_ner_v3_failures", []))
    failed_ids = {str(f.get("chunk_id")) for f in failures if isinstance(f, dict)}

    processed_this_run = 0
    skipped_existing = 0
    failed_this_run = 0

    for i, chunk in enumerate(selected, start=1):
        chunk_id = str(chunk.get("chunk_id"))

        if (not args.overwrite_existing) and chunk_id in processed_ids:
            if args.retry_failed and chunk_id in failed_ids:
                pass
            else:
                skipped_existing += 1
                continue

        print(f"[INFO] Processing chunk {i}/{len(selected)} (chunk_id={chunk_id})")

        try:
            local_entities = annotate_chunk_with_retry(client=client, chunk=chunk, args=args)
            apply_chunk_entities_to_segments(
                data=data,
                chunk=chunk,
                local_entities=local_entities,
                entity_field=args.entity_field,
                merge_with_existing=args.merge_with_existing,
            )
            processed_ids.add(chunk_id)
            failures = [
                f
                for f in failures
                if not (isinstance(f, dict) and str(f.get("chunk_id")) == chunk_id)
            ]
            processed_this_run += 1
        except Exception as exc:
            failed_this_run += 1
            failures.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "error": str(exc),
                    "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                }
            )
            print(f"[ERROR] chunk {chunk_id}: {exc}")

        if processed_this_run > 0 and args.checkpoint_every > 0:
            if processed_this_run % args.checkpoint_every == 0:
                data["llm_ner_v3_processed_chunk_ids"] = sorted(processed_ids, key=chunk_sort_key)
                data["llm_ner_v3_failures"] = failures
                write_json(output_path, data)
                print(f"[INFO] Checkpoint written after {processed_this_run} processed chunks")

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    data["llm_ner_v3_processed_chunk_ids"] = sorted(processed_ids, key=chunk_sort_key)
    data["llm_ner_v3_failures"] = failures
    data["llm_ner_v3_meta"] = {
        "task": "cineminds_llm_ner_v3",
        "model": args.model,
        "generated_at_utc": now,
        "input_json": str(input_path),
        "output_json": str(output_path),
        "base_url": base_url,
        "labels": sorted(ALLOWED_LABELS),
        "total_chunks": len(chunks),
        "selected_chunks": len(selected),
        "processed_this_run": processed_this_run,
        "skipped_existing": skipped_existing,
        "failed_this_run": failed_this_run,
        "failed_total": len(failures),
        "context_window": args.context_window,
        "max_segment_chars": args.max_segment_chars,
        "drop_low_confidence": bool(args.drop_low_confidence),
        "min_score": args.min_score,
        "min_score_misc": args.min_score_misc,
        "min_score_date": args.min_score_date,
        "entity_field": args.entity_field,
        "merge_with_existing": bool(args.merge_with_existing),
    }

    source_info = data.get("chunk_level_annotation_source")
    if not isinstance(source_info, dict):
        source_info = {}
    source_info["llm_ner_v3"] = {
        "task": "segment_level_entities",
        "model": args.model,
        "labels": sorted(ALLOWED_LABELS),
        "entity_field": args.entity_field,
    }
    data["chunk_level_annotation_source"] = source_info

    write_json(output_path, data)

    print("[INFO] Done")
    print(f"[INFO] Processed this run: {processed_this_run}")
    print(f"[INFO] Skipped existing: {skipped_existing}")
    print(f"[INFO] Failed this run: {failed_this_run}")
    print(f"[INFO] Failed total (accumulated): {len(failures)}")
    print(f"[INFO] Output saved: {output_path}")


if __name__ == "__main__":
    main()
