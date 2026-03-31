import os
import re
import glob
import json
import argparse
from copy import deepcopy

CHUNK_RE = re.compile(r'chunk_(\d+)_(\d+)_(\d+)_whisperx(?:_cleaned)?\.json$')

def parse_chunk_info(path):
    name = os.path.basename(path)
    m = CHUNK_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse chunk info from filename: {name}")
    chunk_idx = int(m.group(1))
    chunk_start = float(m.group(2))
    chunk_end = float(m.group(3))
    duration = chunk_end - chunk_start
    return chunk_idx, chunk_start, chunk_end, duration

def is_num(x):
    return isinstance(x, (int, float))

def round3(x):
    return round(float(x), 3)

def clip_to_range(start, end, lo, hi):
    start = max(lo, float(start))
    end = min(hi, float(end))
    return start, end

def prefix_speaker(speaker, chunk_idx):
    if speaker is None:
        return None
    return f"CHUNK_{chunk_idx:04d}_{speaker}"

def clean_word_list(words, chunk_duration):
    out = []
    removed = 0
    clipped = 0
    for w in words or []:
        start = w.get("start")
        end = w.get("end")
        word = (w.get("word") or "").strip()
        if not is_num(start) or not is_num(end) or not word:
            removed += 1
            continue
        if start >= chunk_duration:
            removed += 1
            continue
        orig_end = end
        start, end = clip_to_range(start, end, 0.0, chunk_duration)
        if end <= start:
            removed += 1
            continue
        ww = deepcopy(w)
        ww["start"] = round3(start)
        ww["end"] = round3(end)
        out.append(ww)
        if orig_end > chunk_duration:
            clipped += 1
    return out, removed, clipped

def clean_segments(segments, chunk_duration):
    out = []
    removed = 0
    clipped = 0
    for seg in segments or []:
        start = seg.get("start")
        end = seg.get("end")
        text = (seg.get("text") or "").strip()

        if not is_num(start) or not is_num(end) or not text:
            removed += 1
            continue

        if start >= chunk_duration:
            removed += 1
            continue

        orig_end = end
        start, end = clip_to_range(start, end, 0.0, chunk_duration)
        if end <= start:
            removed += 1
            continue

        new_seg = deepcopy(seg)
        new_seg["start"] = round3(start)
        new_seg["end"] = round3(end)

        if "words" in new_seg:
            words, _, _ = clean_word_list(new_seg.get("words"), chunk_duration)
            new_seg["words"] = words

        out.append(new_seg)
        if orig_end > chunk_duration:
            clipped += 1
    return out, removed, clipped

def clean_word_segments(word_segments, chunk_duration):
    out = []
    removed = 0
    clipped = 0
    for w in word_segments or []:
        start = w.get("start")
        end = w.get("end")
        word = (w.get("word") or "").strip()
        if not is_num(start) or not is_num(end) or not word:
            removed += 1
            continue
        if start >= chunk_duration:
            removed += 1
            continue
        orig_end = end
        start, end = clip_to_range(start, end, 0.0, chunk_duration)
        if end <= start:
            removed += 1
            continue
        ww = deepcopy(w)
        ww["start"] = round3(start)
        ww["end"] = round3(end)
        out.append(ww)
        if orig_end > chunk_duration:
            clipped += 1
    return out, removed, clipped

def clean_one_file(infile, outdir):
    chunk_idx, chunk_start, chunk_end, duration = parse_chunk_info(infile)
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = deepcopy(data)

    segments, seg_removed, seg_clipped = clean_segments(data.get("segments", []), duration)
    word_segments, ws_removed, ws_clipped = clean_word_segments(data.get("word_segments", []), duration)

    cleaned["segments"] = segments
    cleaned["word_segments"] = word_segments
    cleaned["cleaning_metadata"] = {
        "source_file": infile,
        "chunk_index": chunk_idx,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "chunk_duration": duration,
        "segments_removed": seg_removed,
        "segments_clipped": seg_clipped,
        "word_segments_removed": ws_removed,
        "word_segments_clipped": ws_clipped,
    }

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(infile).replace(".json", "_cleaned.json"))
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return outfile, cleaned["cleaning_metadata"]

def trim_words_for_overlap(words, overlap):
    kept = []
    for w in words or []:
        start = w.get("start")
        end = w.get("end")
        if not is_num(start) or not is_num(end):
            continue
        if end <= overlap:
            continue
        ww = deepcopy(w)
        if start < overlap < end:
            ww["start"] = round3(overlap)
        else:
            ww["start"] = round3(start)
        ww["end"] = round3(end)
        kept.append(ww)
    return kept

def trim_segments_for_overlap(segments, overlap):
    kept = []
    removed = 0
    clipped = 0
    for seg in segments or []:
        start = seg.get("start")
        end = seg.get("end")
        if not is_num(start) or not is_num(end):
            removed += 1
            continue
        if end <= overlap:
            removed += 1
            continue

        new_seg = deepcopy(seg)
        if start < overlap < end:
            new_seg["start"] = round3(overlap)
            clipped += 1
            if "words" in new_seg:
                new_seg["words"] = trim_words_for_overlap(new_seg.get("words"), overlap)
        else:
            new_seg["start"] = round3(start)
            if "words" in new_seg:
                new_seg["words"] = trim_words_for_overlap(new_seg.get("words"), overlap)

        new_seg["end"] = round3(end)
        kept.append(new_seg)
    return kept, removed, clipped

def trim_word_segments_for_overlap(word_segments, overlap):
    kept = []
    removed = 0
    clipped = 0
    for w in word_segments or []:
        start = w.get("start")
        end = w.get("end")
        if not is_num(start) or not is_num(end):
            removed += 1
            continue
        if end <= overlap:
            removed += 1
            continue
        ww = deepcopy(w)
        if start < overlap < end:
            ww["start"] = round3(overlap)
            clipped += 1
        else:
            ww["start"] = round3(start)
        ww["end"] = round3(end)
        kept.append(ww)
    return kept, removed, clipped

def combine_files(cleaned_files, combined_out, overlap):
    all_segments = []
    all_word_segments = []
    metadata = []

    for i, path in enumerate(sorted(cleaned_files)):
        chunk_idx, chunk_start, chunk_end, duration = parse_chunk_info(path.replace("_cleaned.json", ".json"))

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        word_segments = data.get("word_segments", [])

        overlap_removed = 0
        overlap_clipped = 0
        ws_overlap_removed = 0
        ws_overlap_clipped = 0

        if i > 0 and overlap > 0:
            segments, overlap_removed, overlap_clipped = trim_segments_for_overlap(segments, overlap)
            word_segments, ws_overlap_removed, ws_overlap_clipped = trim_word_segments_for_overlap(word_segments, overlap)

        metadata.append({
            **data.get("cleaning_metadata", {}),
            "combine_overlap_removed_segments": overlap_removed,
            "combine_overlap_clipped_segments": overlap_clipped,
            "combine_overlap_removed_word_segments": ws_overlap_removed,
            "combine_overlap_clipped_word_segments": ws_overlap_clipped,
        })

        for seg in segments:
            new_seg = deepcopy(seg)
            new_seg["start"] = round3(seg["start"] + chunk_start)
            new_seg["end"] = round3(seg["end"] + chunk_start)
            new_seg["speaker"] = prefix_speaker(seg.get("speaker"), chunk_idx)

            if "words" in new_seg:
                new_words = []
                for w in new_seg["words"]:
                    ww = deepcopy(w)
                    if is_num(ww.get("start")):
                        ww["start"] = round3(ww["start"] + chunk_start)
                    if is_num(ww.get("end")):
                        ww["end"] = round3(ww["end"] + chunk_start)
                    ww["speaker"] = prefix_speaker(ww.get("speaker"), chunk_idx)
                    new_words.append(ww)
                new_seg["words"] = new_words

            all_segments.append(new_seg)

        for w in word_segments:
            ww = deepcopy(w)
            ww["start"] = round3(w["start"] + chunk_start)
            ww["end"] = round3(w["end"] + chunk_start)
            ww["speaker"] = prefix_speaker(w.get("speaker"), chunk_idx)
            all_word_segments.append(ww)

    all_segments.sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
    all_word_segments.sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))

    combined = {
        "segments": all_segments,
        "word_segments": all_word_segments,
        "combine_metadata": {
            "num_files": len(cleaned_files),
            "overlap_seconds": overlap,
            "note": "Speakers are chunk-local and prefixed with CHUNK_xxxx_ to avoid false cross-chunk speaker merging."
        },
        "source_cleaning_metadata": metadata
    }

    os.makedirs(os.path.dirname(combined_out), exist_ok=True)
    with open(combined_out, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--combined_out", required=True)
    parser.add_argument("--overlap", type=float, default=0.0, help="Chunk overlap in seconds, e.g. 15")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    print(f"[INFO] Found {len(files)} files")
    cleaned_files = []
    for infile in files:
        outfile, meta = clean_one_file(infile, args.clean_dir)
        cleaned_files.append(outfile)
        print(
            f"[CLEANED] {os.path.basename(infile)} -> {os.path.basename(outfile)} | "
            f"removed segments={meta['segments_removed']}, clipped segments={meta['segments_clipped']}, "
            f"removed word_segments={meta['word_segments_removed']}, clipped word_segments={meta['word_segments_clipped']}"
        )

    combine_files(cleaned_files, args.combined_out, args.overlap)
    print(f"[DONE] Combined JSON written to: {args.combined_out}")

if __name__ == "__main__":
    main()
