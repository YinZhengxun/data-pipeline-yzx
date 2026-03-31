# V3 Refactoring Report

## A. V3 Restructured Directory Structure

```
V3/
├── pipeline_v3_unified.py          # ★ Main unified pipeline entry point
├── run_v3_pipeline.sh              # ★ Shell launcher
├── config.yaml                     # ★ Default configuration
│
├── requirements.txt                # Python dependencies
│
├── steps/                          # Pipeline step implementations (self-contained)
│   ├── 01_extract_wav.py           # Stage 1: audio extraction
│   ├── 02_split_wav.py             # Stage 2: chunking
│   ├── 03_chunk_asr.py             # Stage 3: ASR per chunk
│   ├── 04_clean_combine.py         # Stage 4+5: clean & merge
│   ├── 05_bertopic.py              # Stage 6: topic modeling
│   ├── 06_flair_ner.py            # Stage 7: NER
│   └── 07_merge.py                # Stage 8: merge back to whisper JSON
│
├── scripts/                        # Purpose-specific standalone scripts
│   ├── process_single_file_pipeline_AG.py   # ASR pipeline for single file
│   ├── asr_whisperx_AG.py                    # Core ASR+diarization+alignment
│   ├── split_wav_chunks.py                   # WAV chunking utility
│   ├── clean_and_combine_whisperx_overlap.py  # Hallucination cleaning + merging
│   ├── bertopic_from_combined_json.py        # Topic modeling
│   ├── flair_ner_on_chunked_json.py          # NER on chunks
│   └── merge_chunk_topics_ner_back_to_whisper_json.py  # Merge annotations
│
├── utils/                          # Shared utilities
│   └── (paths.py, config.py)       # Future: path resolution, config helpers
│
├── legacy/                         # Old/obsolete files kept for reference
│   ├── process_single_file_pipeline.py      # Original V1 (full video ASR - OOM risk)
│   ├── asr_whisperx.py                       # Original ASR module
│   ├── asr_whisperx_AG.py.bak               # Backup
│   ├── annotate_whisperx_json.py            # Alternative annotation approach
│   ├── qwen_topic_detector.py               # Qwen topic detector (unused in V3)
│   └── (other V1 files)
│
└── V3_PIPELINE_GUIDE.md            # This file
```

**Files removed from V3 root:**
- `run_chunks.sh` — hardcoded paths, superseded by unified pipeline
- `process_single_file_pipeline_AG.py` (duplicate) — moved to scripts/

---

## B. Unified Pipeline Design

### Entry Point
```bash
python pipeline_v3_unified.py --episode_path /path/to/video.mp4 --work_dir output_v3
```

### 8 Stages (run in order)

| # | Stage | Key Params | Output |
|---|-------|------------|--------|
| 1 | **Extract WAV** | 16kHz mono | `work_dir/wav/{name}.wav` |
| 2 | **Split WAV** | chunk_sec=600, overlap_sec=15 | `work_dir/chunks/chunks_manifest.json` |
| 3 | **Chunk ASR** | asr_model, gpu_index, hf_token | `work_dir/chunk_json_raw/chunk_*_whisperx.json` |
| 4 | **Clean chunks** | Hallucination removal | `work_dir/chunk_json_clean/` |
| 5 | **Combine** | overlap handling | `work_dir/combined/{name}_combined_whisperx.json` |
| 6 | **BERTopic** | chunk_size=8, german_stopwords=true | `work_dir/topics/chunked_transcript_with_topics.json` |
| 7 | **Flair NER** | ner_model=flair/ner-german-large | `work_dir/topics/chunked_transcript_with_topics_ner.json` |
| 8 | **Merge** | Annotations → whisper JSON | `work_dir/final/{name}_final_whisperx_topic_ner.json` |

### Skip Options
- `--skip_topics` → Stops after stage 5 (combined JSON only)
- `--skip_ner` → Skips stage 7 (BERTopic only, then merge)

### Resume Behavior
- All stages auto-check if output exists
- Use `--no_resume` to force re-run

---

## C. Key Improvements Over Previous Versions

### 1. Fixed `--skip_topics` / `--skip_ner` Logic
Previous code incorrectly required `--skip_ner` when using `--skip_topics`. Now:
- `--skip_topics` stops cleanly after combined JSON
- `--skip_ner` skips NER but still runs BERTopic + merge

### 2. German Stopwords Now Configurable
```bash
python pipeline_v3_unified.py ... --use_german_stopwords   # default
python pipeline_v3_unified.py ... --no_german_stopwords
```
Added `--use_german_stopwords` / `--no_german_stopwords` flags to both pipeline and bertopic script.

### 3. Robust Merge with Mismatch Handling
Previous merge script would fail hard on chunk count mismatch. Now:
- Warns and truncates to minimum count
- Allows pipeline to continue despite minor mismatches

### 4. Environment Auto-Detection
Pipeline tries to auto-detect Python paths:
- ASR: looks for `data-env/bin/python`
- NLP: looks for `topic-env/bin/python`
Override with `--asr_python` and `--nlp_python`

### 5. No Hardcoded User Paths
All `/home/zhyin/...` paths removed from pipeline. Paths now derived from `--work_dir`.

### 6. Config File Support
```bash
python pipeline_v3_unified.py --config config.yaml --episode_path video.mp4
```
Config values fill in defaults; CLI arguments override config.

---

## D. Environment Unification

### Current Situation
- **ASR阶段**: `data-env` (torch, whisperx, pyannote)
- **NLP阶段**: `topic-env` (bertopic, flair)

### Recommended Solution
Keep environments **separate** because:
1. BERTopic + flair have complex ML dependencies that may conflict with whisperx versions
2. torch version conflicts between environments
3. Simpler to maintain two envs than debug dependency hell

### V3 Mechanism
Pipeline handles dual-environment calls internally:
- ASR stages use `asr_python` (or auto-detected `data-env`)
- NLP stages (BERTopic, NER, merge) use `nlp_python` (or auto-detected `topic-env`)
- No manual environment switching needed

### If You Must Merge
```bash
# Create unified env from data-env, then add:
pip install bertopic flair sentence-transformers umap-learn hdbscan
# Risk: version conflicts with whisperx dependencies
```

---

## E. Default Parameters (Validated)

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `chunk_sec` | 600 | 10 min — prevents ASR OOM |
| `overlap_sec` | 15 | 15s overlap for continuity |
| `topic_chunk_size` | 8 | User-validated: better than 5 |
| `use_german_stopwords` | true | User-validated for German content |
| `embedding_model` | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual support |
| `asr_model` | openai/whisper-small | Memory-safe for long videos |
| `ner_model` | flair/ner-german-large | Best for German NER |
| `skip_alignment` | false | Keep alignment for word timestamps |

---

## F. Risk Points &待确认事项

### 1. Chunk Filename Assumption
The pipeline glob pattern `chunk_*_whisperx.json` assumes chunk filenames from `split_wav_chunks.py` always start with `chunk_`. This is correct but fragile — if input WAVs happen to be named with `chunk_` prefix already, collisions possible. **Recommendation**: Use output subdir for chunk JSONs to avoid name collisions.

### 2. `process_single_file_pipeline_AG.py` Always Exports to `json/` Subdirectory
The ASR module creates `output_dir/json/{base_name}_whisperx.json`. The pipeline handles this by looking in `out_dir/json/`. Correct, but if `out_dir` already exists with different content, could have collisions.

### 3. Per-Segment Language Detection Loads Whisper "base" Model
Inside `asr_whisperx_AG.py`, after ASR, it loads a second whisper model for per-segment language detection. This doubles GPU memory usage. On memory-constrained GPUs, consider `--skip_alignment` to reduce peak memory.

### 4. JSON Schema Inconsistency
- `combined_whisperx.json` has top-level `segments`, `word_segments`, `combine_metadata`
- `chunked_transcript_with_topics.json` has `{"chunks": [...]}`
- `final_whisperx_topic_ner.json` is back to whisper-style `{"segments": [...]}`

The merge step converts back to whisper-style, which is good for compatibility, but the intermediate formats are different.

### 5. Unknown: topic-env Location
The pipeline tries to auto-detect `topic-env` at multiple paths. If not found, falls back to inheriting Python. User should verify `topic-env` exists or set `--nlp_python` explicitly.

### 6. Overlap Trimming in `clean_and_combine_whisperx_overlap.py`
The overlap trimming (`trim_segments_for_overlap`) removes any segment where `end <= overlap`. This is correct for removing duplicate content in overlap regions, but could accidentally remove very short segments that happen to fall in the overlap zone.

### 7. Hallucination Cleaning Limited
`clean_one_file()` does word/segment clipping and boundary cleanup, but the hallucination detection (e.g., detecting repeated template sentences like "Dies ist ein deutsches Video") is **not** implemented — it's mentioned in the problem description as needed, but the actual `clean_one_file()` function only does boundary trimming. Real hallucination detection would require pattern matching or ML-based approach.

---

## G. Testing on GPU

Once GPU is available, test with:

```bash
cd /path/to/data-pipeline-main/V3

# Test with 3 chunks only (safe)
python pipeline_v3_unified.py \
  --episode_path /path/to/test_video.mp4 \
  --work_dir test_output \
  --max_chunks 3 \
  --hf_token YOUR_HF_TOKEN \
  --language de

# Full run
python pipeline_v3_unified.py \
  --episode_path /path/to/long_video.mp4 \
  --work_dir long_video_output \
  --hf_token YOUR_HF_TOKEN \
  --language de

# Skip topics (ASR only)
python pipeline_v3_unified.py \
  --episode_path /path/to/video.mp4 \
  --work_dir asr_only \
  --skip_topics \
  --hf_token YOUR_HF_TOKEN
```
