#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError("Missing dependency: openai. Install with `pip install openai`.") from exc


DEFAULT_MODEL = "MiniMax-M2.7"
DEFAULT_OUTPUT_SUFFIX = "_subtyped"
HARDCODED_BASE_URL = "https://api.minimax.io/v1"
HARDCODED_API_KEY = ""
TARGET_TYPES = ("PER", "ORG", "WORK_OF_ART", "EVT")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Classify entity subtypes on CineMinds enriched WhisperX JSON."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_suffix", default=DEFAULT_OUTPUT_SUFFIX)
    parser.add_argument(
        "--ner_taxonomy_json",
        default=str(root / "schemas" / "cineminds_ner_taxonomy.json"),
    )
    parser.add_argument(
        "--entity_knowledge_json",
        default=str(root / "schemas" / "entity_knowledge_hints.json"),
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--base_url_env", default="OPENAI_BASE_URL")

    parser.add_argument("--context_window", type=int, default=1)
    parser.add_argument("--max_contexts", type=int, default=5)
    parser.add_argument("--max_context_chars", type=int, default=320)
    parser.add_argument("--max_entities", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_backoff_sec", type=float, default=2.0)
    parser.add_argument("--min_confidence", type=float, default=0.55)
    parser.add_argument("--reclassify_existing", action="store_true")
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


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_text_key(text: Any) -> str:
    return re.sub(r"\W+", "", normalize_ws(text).casefold())


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3].rstrip()
    ws = cut.rfind(" ")
    if ws > max_chars * 0.65:
        cut = cut[:ws].rstrip()
    return cut + "..."


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


def subtype_options_for_type(taxonomy: dict[str, Any], entity_type: str) -> dict[str, dict[str, Any]]:
    type_cfg = taxonomy.get("types", {}).get(entity_type, {})
    subtypes = type_cfg.get("subtypes", {})
    if not isinstance(subtypes, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in subtypes.items():
        if not isinstance(cfg, dict):
            cfg = {}
        out[str(name)] = cfg
    return out


def load_entity_knowledge_hints(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = load_json(path)
    arr = data.get("hints", [])
    if not isinstance(arr, list):
        return []
    out: list[dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        norm_aliases = [normalize_text_key(x) for x in aliases if normalize_text_key(x)]
        if not norm_aliases:
            continue
        out.append(
            {
                **item,
                "_aliases_norm": norm_aliases,
            }
        )
    return out


def matching_entity_knowledge(entity_text: str, hints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = normalize_text_key(entity_text)
    if not key:
        return []
    return [hint for hint in hints if key in hint.get("_aliases_norm", [])]


def build_sentence_index(sentences: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[int]]]:
    by_id: dict[str, int] = {}
    by_chunk: dict[str, list[int]] = {}
    for idx, sent in enumerate(sentences):
        sid = normalize_ws(sent.get("sentence_id"))
        if sid:
            by_id[sid] = idx
        chunk_id = str(sent.get("chunk_id", ""))
        by_chunk.setdefault(chunk_id, []).append(idx)
    return by_id, by_chunk


def gather_contexts(
    sentence_ids: list[str],
    sentences: list[dict[str, Any]],
    sent_id_to_index: dict[str, int],
    by_chunk: dict[str, list[int]],
    context_window: int,
    max_contexts: int,
    max_context_chars: int,
) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()
    for sid in sentence_ids:
        idx = sent_id_to_index.get(sid)
        if idx is None:
            continue
        sent = sentences[idx]
        chunk_id = str(sent.get("chunk_id", ""))
        chunk_indices = by_chunk.get(chunk_id, [])
        try:
            pos = chunk_indices.index(idx)
        except ValueError:
            pos = -1
        if pos >= 0:
            window_indices = chunk_indices[max(0, pos - context_window) : pos + context_window + 1]
        else:
            window_indices = [idx]
        parts = [normalize_ws(sentences[i].get("text", "")) for i in window_indices if normalize_ws(sentences[i].get("text", ""))]
        joined = truncate_text(" ".join(parts), max_context_chars)
        key = normalize_text_key(joined)
        if not key or key in seen:
            continue
        seen.add(key)
        contexts.append(joined)
        if len(contexts) >= max_contexts:
            break
    return contexts


def choose_display_text(mentions: list[dict[str, Any]]) -> str:
    best = ""
    for mention in mentions:
        text = normalize_ws(mention.get("text", ""))
        if not text:
            continue
        if len(text.split()) > len(best.split()):
            best = text
            continue
        if len(text.split()) == len(best.split()) and len(text) > len(best):
            best = text
    return best


def build_jobs(
    data: dict[str, Any],
    taxonomy: dict[str, Any],
    knowledge_hints: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sentences = data.get("sentences", [])
    entities = data.get("entities", [])
    if not isinstance(sentences, list) or not isinstance(entities, list):
        raise ValueError("Input JSON must contain 'sentences' and 'entities' lists.")

    sent_id_to_index, by_chunk = build_sentence_index(sentences)
    grouped: dict[str, dict[str, Any]] = {}
    default_subtype = str(taxonomy.get("default_subtype", "unspecified"))

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_id = normalize_ws(entity.get("id"))
        entity_type = normalize_ws(entity.get("type")).upper()
        if not entity_id or entity_type not in TARGET_TYPES:
            continue
        subtypes = subtype_options_for_type(taxonomy, entity_type)
        if not subtypes:
            continue
        current_subtype = normalize_ws(entity.get("subtype")) or default_subtype
        if (not args.reclassify_existing) and current_subtype != default_subtype:
            continue
        job = grouped.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "mentions": [],
                "sentence_ids": [],
                "current_subtypes": Counter(),
            },
        )
        job["mentions"].append(entity)
        sid = normalize_ws(entity.get("sentence_id"))
        if sid:
            job["sentence_ids"].append(sid)
        job["current_subtypes"][current_subtype] += 1

    jobs: list[dict[str, Any]] = []
    for job in grouped.values():
        entity_type = job["entity_type"]
        subtype_cfg = subtype_options_for_type(taxonomy, entity_type)
        contexts = gather_contexts(
            sentence_ids=job["sentence_ids"],
            sentences=sentences,
            sent_id_to_index=sent_id_to_index,
            by_chunk=by_chunk,
            context_window=max(0, args.context_window),
            max_contexts=max(1, args.max_contexts),
            max_context_chars=max(120, args.max_context_chars),
        )
        job["entity_text"] = choose_display_text(job["mentions"])
        job["contexts"] = contexts
        job["subtype_options"] = subtype_cfg
        job["knowledge_hints"] = matching_entity_knowledge(job["entity_text"], knowledge_hints)
        jobs.append(job)

    jobs.sort(key=lambda x: (x["entity_type"], x["entity_text"].casefold(), x["entity_id"]))
    if args.max_entities is not None:
        jobs = jobs[: max(0, args.max_entities)]
    return jobs


def build_messages(job: dict[str, Any], taxonomy: dict[str, Any]) -> list[dict[str, str]]:
    entity_type = job["entity_type"]
    type_desc = normalize_ws(taxonomy.get("types", {}).get(entity_type, {}).get("description", ""))
    subtype_lines = []
    for name, cfg in job["subtype_options"].items():
        keywords = cfg.get("keywords", []) if isinstance(cfg, dict) else []
        subtype_lines.append(
            {
                "subtype": name,
                "keywords": [normalize_ws(k) for k in keywords if normalize_ws(k)],
            }
        )

    payload = {
        "entity_id": job["entity_id"],
        "entity_text": job["entity_text"],
        "entity_type": entity_type,
        "entity_type_description": type_desc,
        "current_subtype": job["current_subtypes"].most_common(1)[0][0] if job["current_subtypes"] else "unspecified",
        "allowed_subtypes": subtype_lines,
        "contexts": job["contexts"],
        "knowledge_hints": [
            {
                "canonical_name": normalize_ws(h.get("canonical_name", "")),
                "preferred_type": normalize_ws(h.get("preferred_type", "")).upper(),
                "preferred_subtype": normalize_ws(h.get("preferred_subtype", "")),
                "description": normalize_ws(h.get("description", "")),
            }
            for h in job.get("knowledge_hints", [])
        ],
    }

    system = (
        "You are classifying film-domain named entity subtypes in a transcript.\n"
        "Your job is not to decide whether the entity is film-related. That is already established.\n"
        "Your job is to choose the most plausible subtype within the allowed list.\n"
        "Use transcript context first, but you may also use widely known public film knowledge for famous people, works, organizations, and events.\n"
        "Do not default to 'unspecified' just because the profession is not stated literally in the sentence.\n"
        "For famous film people such as actors, directors, producers, writers, or cinematographers, assign the best supported subtype.\n"
        "If a person is known for multiple film roles, choose the primary subtype that best fits the current mention context.\n"
        "Examples: Marlon Brando -> actor; Robert De Niro -> actor; Johnny Depp -> actor; Jean-Luc Godard -> director; Clint Eastwood in an acting discussion -> actor.\n"
        "Only return 'unspecified' when the subtype is genuinely unclear even after using context and common film knowledge.\n"
        "Use only the allowed subtype names or 'unspecified'.\n"
        "Return JSON only."
    )
    user = (
        "Classify the entity subtype using the provided transcript contexts.\n"
        "Important rules:\n"
        "1. Prefer a specific subtype over 'unspecified' when a famous film entity is reasonably identifiable.\n"
        "2. If the context is about performance or acting examples, prefer 'actor' for film performers.\n"
        "3. If the context is about directing, regie, or auteurship, prefer 'director'.\n"
        "4. If the entity is a title and is clearly a film example, prefer 'film'.\n"
        "5. Use 'unspecified' only when none of the allowed subtypes can be defended.\n"
        "Output format:\n"
        "{\n"
        '  "entity_id": "...",\n'
        '  "subtype": "one allowed subtype or unspecified",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short phrase citing the cue or world-knowledge basis"\n'
        "}\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    raw = normalize_ws(text)
    if not raw:
        raise ValueError("Empty model response")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not locate JSON object in response: {raw[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    return parsed


def classify_with_knowledge_hint(job: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any] | None:
    contexts_text = " ".join([normalize_ws(x) for x in job.get("contexts", [])]).casefold()
    for hint in job.get("knowledge_hints", []):
        preferred_type = normalize_ws(hint.get("preferred_type", "")).upper()
        preferred_subtype = normalize_ws(hint.get("preferred_subtype", ""))
        if preferred_type not in TARGET_TYPES:
            continue
        allowed_subtypes = set(subtype_options_for_type(taxonomy, preferred_type).keys())
        if preferred_subtype not in allowed_subtypes:
            continue
        keywords = [normalize_ws(x).casefold() for x in hint.get("context_keywords", []) if normalize_ws(x)]
        require_context = bool(hint.get("require_context_keywords", False))
        if require_context and keywords and not any(k in contexts_text for k in keywords):
            continue
        confidence = hint.get("confidence", 0.98 if not require_context else 0.9)
        return {
            "entity_id": job["entity_id"],
            "entity_text": job["entity_text"],
            "entity_type": preferred_type,
            "subtype": preferred_subtype,
            "confidence": normalize_confidence(confidence),
            "evidence": normalize_ws(
                hint.get("description", f"knowledge hint: {hint.get('canonical_name', job['entity_text'])}")
            ),
            "source": "knowledge_hint",
        }
    return None


def normalize_subtype_choice(choice: Any, allowed: set[str], default_subtype: str) -> str:
    value = normalize_ws(choice)
    if not value:
        return default_subtype
    if value in allowed:
        return value
    lowered = value.casefold()
    for item in allowed:
        if item.casefold() == lowered:
            return item
    return default_subtype


def normalize_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def classify_job(
    client: OpenAI,
    job: dict[str, Any],
    taxonomy: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    allowed = set(job["subtype_options"].keys()) | {"unspecified"}
    allowed_types = {job["entity_type"]}
    for hint in job.get("knowledge_hints", []):
        preferred_type = normalize_ws(hint.get("preferred_type", "")).upper()
        if preferred_type in TARGET_TYPES:
            allowed_types.add(preferred_type)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entity_id", "entity_type", "subtype", "confidence", "evidence"],
        "properties": {
            "entity_id": {"type": "string"},
            "entity_type": {"type": "string", "enum": sorted(allowed_types)},
            "subtype": {"type": "string", "enum": sorted(allowed)},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
        },
    }
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(job, taxonomy),
        "temperature": args.temperature,
    }
    if not args.no_schema_guard:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_subtype_response",
                "strict": True,
                "schema": schema,
            },
        }

    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content or ""
    parsed = extract_json_object(content)
    default_subtype = str(taxonomy.get("default_subtype", "unspecified"))
    return {
        "entity_id": job["entity_id"],
        "entity_text": job["entity_text"],
        "entity_type": normalize_ws(parsed.get("entity_type", job["entity_type"])).upper() or job["entity_type"],
        "subtype": normalize_subtype_choice(parsed.get("subtype"), allowed, default_subtype),
        "confidence": normalize_confidence(parsed.get("confidence")),
        "evidence": normalize_ws(parsed.get("evidence", "")),
        "source": "llm",
    }


def apply_predictions(
    data: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    min_confidence: float,
) -> dict[str, int]:
    entities = data.get("entities", [])
    sentences = data.get("sentences", [])
    if not isinstance(entities, list) or not isinstance(sentences, list):
        raise ValueError("Input JSON must contain 'entities' and 'sentences' lists.")

    applied = 0
    skipped_low_conf = 0
    unresolved = 0
    changed_ids: dict[str, tuple[str, str]] = {}

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_id = normalize_ws(entity.get("id"))
        pred = predictions.get(entity_id)
        if not pred:
            continue
        subtype = normalize_ws(pred.get("subtype"))
        confidence = normalize_confidence(pred.get("confidence"))
        if subtype == "unspecified":
            unresolved += 1
            continue
        if confidence < min_confidence:
            skipped_low_conf += 1
            continue
        entity_type = normalize_ws(pred.get("entity_type", entity.get("type", ""))).upper() or normalize_ws(entity.get("type", "")).upper()
        entity["type"] = entity_type
        entity["subtype"] = subtype
        changed_ids[entity_id] = (entity_type, subtype)
        applied += 1

    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        words = sentence.get("words", [])
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            entity_id = normalize_ws(word.get("ner_id"))
            if entity_id in changed_ids:
                word["ner_type"] = changed_ids[entity_id][0]
                word["ner_subtype"] = changed_ids[entity_id][1]

    return {
        "applied_mentions": applied,
        "updated_entity_ids": len(changed_ids),
        "skipped_low_confidence": skipped_low_conf,
        "kept_unspecified": unresolved,
    }


def main() -> None:
    args = parse_args()
    input_path = expand_path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    output_path = (
        expand_path(args.output_json)
        if args.output_json
        else input_path.with_name(f"{input_path.stem}{args.output_suffix}{input_path.suffix}")
    )
    taxonomy_path = expand_path(args.ner_taxonomy_json)
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy JSON not found: {taxonomy_path}")
    knowledge_path = expand_path(args.entity_knowledge_json)

    data = load_json(input_path)
    taxonomy = load_json(taxonomy_path)
    knowledge_hints = load_entity_knowledge_hints(knowledge_path)
    jobs = build_jobs(data, taxonomy, knowledge_hints, args)
    if args.dry_run:
        print(f"[INFO] Input: {input_path}")
        print(f"[INFO] Dry run only")
        print(f"[INFO] Candidate entities: {len(jobs)}")
        return

    api_key = resolve_api_key(args)
    base_url = resolve_base_url(args)
    client = OpenAI(api_key=api_key, base_url=base_url)

    predictions: dict[str, dict[str, Any]] = {}
    for idx, job in enumerate(jobs, start=1):
        rule_pred = classify_with_knowledge_hint(job, taxonomy)
        if rule_pred is not None:
            predictions[job["entity_id"]] = rule_pred
            print(
                f"[INFO] [{idx}/{len(jobs)}] {job['entity_id']} {job['entity_text']} -> "
                f"{rule_pred['subtype']} ({rule_pred['confidence']:.2f}) [knowledge_hint]"
            )
            continue
        attempt = 0
        while True:
            try:
                pred = classify_job(client, job, taxonomy, args)
                predictions[job["entity_id"]] = pred
                print(
                    f"[INFO] [{idx}/{len(jobs)}] {job['entity_id']} {job['entity_text']} -> "
                    f"{pred['subtype']} ({pred['confidence']:.2f})"
                )
                break
            except Exception as exc:
                attempt += 1
                if attempt >= max(1, args.max_retries):
                    raise RuntimeError(
                        f"Subtype classification failed for {job['entity_id']} after {attempt} attempts"
                    ) from exc
                time.sleep(max(0.0, args.retry_backoff_sec) * attempt)

    stats = apply_predictions(data, predictions, args.min_confidence)

    meta = data.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["entity_subtype_classification"] = {
            "task": "llm_entity_subtype_on_enriched_v2",
            "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "model": args.model,
            "candidate_entity_ids": len(jobs),
            "predictions": len(predictions),
            "min_confidence": args.min_confidence,
            "knowledge_hints_loaded": len(knowledge_hints),
            **stats,
        }

    write_json(output_path, data)
    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Candidate entities: {len(jobs)}")
    print(f"[INFO] Predictions: {len(predictions)}")
    print(f"[INFO] Updated entity ids: {stats['updated_entity_ids']}")
    print(f"[INFO] Updated mentions: {stats['applied_mentions']}")


if __name__ == "__main__":
    main()
