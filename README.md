# V3 Unified Audio/Video Pipeline

This repository contains a reproducible V3 pipeline for long audio/video processing:

1. WAV extraction (16k mono)
2. Chunking with overlap
3. Chunk-level ASR + diarization + alignment
4. Chunk cleaning and transcript merge
5. BERTopic topic modeling
6. Flair NER
7. Merge topic/NER back into final whisper-style JSON

The main entrypoint is `pipeline_v3_unified.py`.

## Repository Structure

- `pipeline_v3_unified.py`: one-click end-to-end pipeline
- `flair_ner_on_chunked_json.py`: NER on topic chunks
- `bertopic_from_combined_json.py`: topic modeling
- `merge_chunk_topics_ner_back_to_whisper_json.py`: merge topic/NER back to final JSON
- `requirements.txt`: install all dependencies (single environment)
- `requirements_asr.txt`: ASR environment dependencies
- `requirements_nlp.txt`: BERTopic/NER environment dependencies
- `run_v3_pipeline.sh`: shell launcher

## Requirements

- Python 3.10+
- `ffmpeg` installed and available in PATH (required by `pydub`)
- Optional but recommended: NVIDIA GPU + CUDA for ASR/NER speed

## Installation

### Option A: Single environment (quick start)

```bash
python -m venv v3-env
source v3-env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv v3-env
.\v3-env\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### Option B: Two environments (recommended for stability)

ASR environment:

```bash
python -m venv data-env
source data-env/bin/activate
pip install -U pip
pip install -r requirements_asr.txt
```

NLP environment (BERTopic + NER):

```bash
python -m venv topic-env
source topic-env/bin/activate
pip install -U pip
pip install -r requirements_nlp.txt
```

## Quick Run

```bash
python pipeline_v3_unified.py \
  --episode_path /path/to/input.mp4 \
  --work_dir /path/to/output_v3 \
  --hf_token <your_hf_token> \
  --language de \
  --gpu_index 0
```

With separate Python environments:

```bash
python pipeline_v3_unified.py \
  --episode_path /path/to/input.mp4 \
  --work_dir /path/to/output_v3 \
  --hf_token <your_hf_token> \
  --asr_python /abs/path/to/data-env/bin/python \
  --nlp_python /abs/path/to/topic-env/bin/python
```

## Useful Flags

- `--skip_topics`: stop after combined transcript (skip BERTopic + NER + merge)
- `--skip_ner`: run BERTopic but skip NER, then merge
- `--max_chunks N`: debug with first N chunks
- `--no_resume`: force rerun even if outputs exist
- `--ner_force_cpu`: force NER on CPU

## Output Files

Main outputs under `work_dir`:

- `combined/<name>_combined_whisperx.json`
- `topics/chunked_transcript_with_topics.json`
- `topics/chunked_transcript_with_topics_ner.json`
- `final/<name>_final_whisperx_topic_ner.json`
- `pipeline_summary.json`

## NER-Related Fixes (current version)

- NER stage now correctly receives `--gpu_index` from the unified pipeline.
- NER script validates GPU index and falls back safely if index is invalid.
- Merge step now keeps consistent default fields (`topic_id`, `topic_words`, `entities`) for all segments.
- Remaining uncovered segment count is now computed correctly.

## GitHub Submission Checklist

1. Confirm `config.yaml` does not contain private tokens.
2. Keep large generated outputs out of git (`V3_data`, `__pycache__`, etc.).
3. Include this README and requirements files so collaborators can reproduce.
4. Commit and push:

```bash
git init
git add .
git commit -m "Add V3 unified pipeline with BERTopic + NER docs"
git branch -M main
git remote add origin <your_repo_url>
git push -u origin main
```
