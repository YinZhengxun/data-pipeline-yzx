#!/usr/bin/env python3
"""
Extract a clean glossary table and tag dictionary from the CineMinds Excel file.

Key behaviors:
- Handles "main term row + continuation rows" format.
- Keeps German/English term, register/field metadata, and optional example rows.
- Produces term/register/field tags for chunk-level tagging.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColumnMap:
    term_de: str
    term_en: str
    register: str
    field: str
    exp_field: str
    source: str
    description_de: str
    description_en: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CineMinds glossary tags from Excel.")
    parser.add_argument(
        "--input_xlsx",
        required=True,
        help="Path to glossary Excel file (for example Glossary_BordwellThompson.xlsx).",
    )
    parser.add_argument("--sheet", default=0, help="Sheet name or index. Default: first sheet.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: <input_dir>/glossary_exports",
    )
    return parser.parse_args()


def normalize_header(text: Any) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "", raw)
    return raw


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def resolve_columns(df: pd.DataFrame) -> ColumnMap:
    normalized = {normalize_header(col): col for col in df.columns}

    def pick(*aliases: str) -> str:
        for alias in aliases:
            key = normalize_header(alias)
            if key in normalized:
                return str(normalized[key])
        return ""

    col_map = ColumnMap(
        term_de=pick("Term_DE", "TermDE", "Term DE"),
        term_en=pick("Term_EN", "TermEN", "Term EN"),
        register=pick("Register"),
        field=pick("Field"),
        exp_field=pick("exp Field", "exp_field", "expField"),
        source=pick("Source"),
        description_de=pick("Description German", "Description_German"),
        description_en=pick("Description English", "Description_English"),
    )

    missing = [
        name
        for name, value in col_map.__dict__.items()
        if name in {"term_de", "term_en"} and not value
    ]
    if missing:
        raise ValueError(f"Required glossary columns not found: {', '.join(missing)}")

    return col_map


def split_variants(text: str) -> list[str]:
    if not text:
        return []
    raw_parts = re.split(r"[;/]|(?<!\w)/(?!\w)", text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        item = clean_text(part)
        key = item.casefold()
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned


def extract_records(df: pd.DataFrame, col_map: ColumnMap) -> list[dict[str, Any]]:
    core_cols = {
        col_map.term_de,
        col_map.term_en,
        col_map.register,
        col_map.field,
        col_map.exp_field,
        col_map.source,
        col_map.description_de,
        col_map.description_en,
    }
    extra_cols = [str(c) for c in df.columns if str(c) not in core_cols]

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row_idx, row in df.iterrows():
        term_de = clean_text(row.get(col_map.term_de, "")) if col_map.term_de else ""
        term_en = clean_text(row.get(col_map.term_en, "")) if col_map.term_en else ""
        register = clean_text(row.get(col_map.register, "")) if col_map.register else ""
        field = clean_text(row.get(col_map.field, "")) if col_map.field else ""
        exp_field = clean_text(row.get(col_map.exp_field, "")) if col_map.exp_field else ""
        source = clean_text(row.get(col_map.source, "")) if col_map.source else ""
        desc_de = clean_text(row.get(col_map.description_de, "")) if col_map.description_de else ""
        desc_en = clean_text(row.get(col_map.description_en, "")) if col_map.description_en else ""

        is_main_row = bool(term_de or term_en)
        if is_main_row:
            current = {
                "row_start": int(row_idx) + 2,
                "term_id": "",
                "term_de": term_de,
                "term_en": term_en,
                "term_de_variants": split_variants(term_de),
                "term_en_variants": split_variants(term_en),
                "register": register,
                "field": field,
                "exp_field": exp_field,
                "source": source,
                "description_de": desc_de,
                "description_en": desc_en,
                "example_rows": [],
            }
            records.append(current)
        elif current is None:
            continue
        else:
            if register and not current.get("register"):
                current["register"] = register
            if field and not current.get("field"):
                current["field"] = field
            if exp_field and not current.get("exp_field"):
                current["exp_field"] = exp_field
            if source and not current.get("source"):
                current["source"] = source
            if desc_de and not current.get("description_de"):
                current["description_de"] = desc_de
            if desc_en and not current.get("description_en"):
                current["description_en"] = desc_en

        assert current is not None
        extras: dict[str, str] = {}
        for col in extra_cols:
            value = clean_text(row.get(col, ""))
            if value:
                extras[col] = value
        if extras:
            current["example_rows"].append(
                {
                    "row_index": int(row_idx) + 2,
                    "data": extras,
                }
            )

    for idx, record in enumerate(records, start=1):
        record["term_id"] = f"term_{idx:04d}"

    return records


def build_tag_dictionary(records: list[dict[str, Any]]) -> dict[str, Any]:
    tags: list[dict[str, Any]] = []
    seen_tag_keys: set[tuple[str, str, str, str]] = set()
    next_id = 1

    def add_tag(tag: dict[str, Any]) -> None:
        nonlocal next_id
        key = (
            str(tag.get("category", "")).casefold(),
            str(tag.get("label", "")).casefold(),
            str(tag.get("register", "")).casefold(),
            str(tag.get("field", "")).casefold(),
        )
        if not key[1] or key in seen_tag_keys:
            return
        seen_tag_keys.add(key)
        tag["tag_id"] = f"tag_{next_id:04d}"
        next_id += 1
        tags.append(tag)

    for item in records:
        variants = item.get("term_en_variants", []) + item.get("term_de_variants", [])
        variants_seen: set[str] = set()
        for variant in variants:
            k = str(variant).casefold()
            if k in variants_seen:
                continue
            variants_seen.add(k)
            add_tag(
                {
                    "category": "term",
                    "label": variant,
                    "term_id": item.get("term_id"),
                    "term_en": item.get("term_en", ""),
                    "term_de": item.get("term_de", ""),
                    "register": item.get("register", ""),
                    "field": item.get("field", ""),
                    "exp_field": item.get("exp_field", ""),
                    "source": item.get("source", ""),
                }
            )

        if item.get("field"):
            add_tag(
                {
                    "category": "field",
                    "label": item.get("field"),
                    "field": item.get("field", ""),
                    "exp_field": item.get("exp_field", ""),
                }
            )

        if item.get("register"):
            add_tag(
                {
                    "category": "register",
                    "label": item.get("register"),
                    "register": item.get("register", ""),
                }
            )

    by_category: dict[str, list[str]] = {"term": [], "field": [], "register": []}
    for tag in tags:
        category = str(tag.get("category", "")).lower()
        if category in by_category:
            by_category[category].append(str(tag.get("label", "")))
    for category in by_category:
        by_category[category] = sorted({x for x in by_category[category] if x}, key=str.casefold)

    return {
        "version": "1.0.0",
        "tags": tags,
        "vocabulary": by_category,
        "stats": {
            "tag_count": len(tags),
            "term_count": len(by_category["term"]),
            "field_count": len(by_category["field"]),
            "register_count": len(by_category["register"]),
        },
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_xlsx).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent / "glossary_exports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, sheet_name=args.sheet, dtype=str).fillna("")
    col_map = resolve_columns(df)
    records = extract_records(df, col_map)
    tag_dict = build_tag_dictionary(records)

    records_json = output_dir / "glossary_records.json"
    tags_json = output_dir / "glossary_tag_dictionary.json"
    vocab_json = output_dir / "glossary_tag_vocabulary.json"
    tags_csv = output_dir / "glossary_tag_dictionary.csv"

    save_json(records_json, {"records": records, "count": len(records)})
    save_json(tags_json, tag_dict)
    save_json(vocab_json, tag_dict["vocabulary"])

    pd.DataFrame(tag_dict["tags"]).to_csv(tags_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Main glossary records: {len(records)}")
    print(f"[INFO] Tags generated: {tag_dict['stats']['tag_count']}")
    print(f"[INFO] Files:")
    print(f"  - {records_json}")
    print(f"  - {tags_json}")
    print(f"  - {vocab_json}")
    print(f"  - {tags_csv}")


if __name__ == "__main__":
    main()
