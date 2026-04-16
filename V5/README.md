# CineMinds V5 Handoff

## Purpose

This `V5` folder is a cleaned handoff bundle for the next teammate.

It contains:

- the current scripts that were actually useful for the new full-file pipeline
- the schemas and glossary resources needed by those scripts
- the best stage outputs from the current pipeline
- a concise explanation of what is finished, what is only partial, and where to continue next

This bundle is meant to reduce confusion from older experimental files in `V3` and from the earlier chunk-based pipeline.

## Directory Layout

```text
V5/
  README.md
  run_swiss_asr_fullfile.py
  process_single_file_pipeline_swiss.py
  asr_whisperx_AG.py
  postprocess_fullfile_asr.py
  bertopic_from_combined_json.py
  merge_chunk_topics_ner_back_to_whisper_json.py
  llm_ner_on_whisper_json_v3.py
  build_whisperx_enriched_v2.py
  llm_entity_subtype_on_enriched_v2.py
  extract_glossary_tags.py
  requirements.txt
  requirements_nlp.txt
  schemas/
    cineminds_ner_taxonomy.json
    entity_knowledge_hints.json
    cineminds_whisperx_enriched_v2.schema.json
    cineminds_chunk_annotation.schema.json
  resources/
    glossary/
      Glossary_BordwellThompson.xlsx
      glossary_records.json
      glossary_tag_dictionary.json
      glossary_tag_dictionary.csv
      glossary_tag_vocabulary.json
  results/
    01_asr_input/
      Martin_Rapold_16k_mono_whisperx_postprocessed.json
    02_topics_ner/
      Martin_Rapold_16k_mono_topics_merged_llm_ner_v3.json
    03_final/
      final_whisper_style_enriched_v2_fullfile_fix2.json
      final_whisper_style_enriched_v2_fullfile_subtyped_v3.json
```

## Best Files By Stage

These are the files that should be treated as the current best artifacts.

### 1. Best ASR transcript input

Use this as the clean transcript input for downstream topic / NER work:

- `results/01_asr_input/Martin_Rapold_16k_mono_whisperx_postprocessed.json`

Why this one:

- it comes from the new full-file Swiss German ASR path
- it was post-processed after ASR
- long segments were split more reasonably
- missing speaker labels were filled
- language labels were normalized

### 2. Best topic + NER intermediate

Use this if you want to start from the chunk/topic/NER stage instead of rerunning ASR:

- `results/02_topics_ner/Martin_Rapold_16k_mono_topics_merged_llm_ner_v3.json`

Why this one:

- it is based on the cleaner full-file postprocessed transcript
- compared with the older rechunked pipeline, it had fewer low-confidence entities and less `MISC` noise

### 3. Best final enriched output before subtype refinement

Use this if you want to inspect or rerun only the subtype step:

- `results/03_final/final_whisper_style_enriched_v2_fullfile_fix2.json`

Why this one:

- it is the best `build_whisperx_enriched_v2.py` output before the subtype refinement pass
- it already contains sentence-level tags, word-level NER writeback, and multi-token shared IDs

### 4. Best final enriched output overall

Use this as the current main result:

- `results/03_final/final_whisper_style_enriched_v2_fullfile_subtyped_v3.json`

Why this one:

- it is the current best final output
- it includes the subtype refinement pass on top of `fix2`
- it is the best file for inspection, demo, and next-step continuation

## What Changed Compared With The Older Pipeline

The most important change is that the input side was rebuilt.

The old path relied heavily on chunk-based / rechunked transcript inputs. That created several downstream problems:

- chunk-boundary artifacts
- missing speaker labels
- noisier segment structure
- worse downstream word-level alignment
- more NER noise

The new path switched to:

1. full-file Swiss German ASR
2. post-processing after ASR
3. then topic / NER / glossary tagging / enriched build

This made the downstream files cleaner and easier to work with.

Observed improvements on the new input side:

- segment count became more stable for downstream use
- missing segment speaker labels dropped to `0`
- missing word speaker labels dropped to `0`
- topic + NER intermediate quality improved
- final enriched output had fewer obviously bad word-level entity spans

## Quality Snapshot Of The Current Best Files

These are the main reasons the included files were selected as the current best versions.

### Input-side improvement

Compared with the older rechunked-style input:

- segment-level speaker gaps were eliminated
- word-level speaker gaps were eliminated
- downstream transcript structure became less noisy

### Topic + NER intermediate improvement

Compared with the older rechunked LLM NER intermediate:

- entity confidence improved
- low-confidence entities decreased substantially
- `WORK` mentions increased
- `MISC` noise decreased

### Final enriched output improvement

Compared with the older final rechunked enriched output:

- word-level NER writeback became cleaner
- punctuation and stopword-related entity noise was reduced
- `field/register/term` tag structure is now present instead of mostly `term` only
- subtype refinement reduced the number of `unspecified` target entities

Current caveat:

- glossary tagging quality is structurally improved, but still largely lexical rather than fully semantic
- some boundary and taxonomy cases remain unresolved

## Pipeline Overview

The pipeline now has five practical stages.

### Stage A. Full-file Swiss German ASR

Files:

- `run_swiss_asr_fullfile.py`
- `process_single_file_pipeline_swiss.py`
- `asr_whisperx_AG.py`

Purpose:

- run Swiss German ASR on the full file instead of externally splitting the audio into large chunks
- keep output compatible with the WhisperX-style JSON structure already used downstream

Recommended environment:

```bash
source ~/cineminds/data-env/bin/activate
```

Typical command pattern:

```bash
python V5/run_swiss_asr_fullfile.py \
  --episode_path /path/to/input.wav \
  --out_dir /path/to/output_dir \
  --gpu_index 1 \
  --language de \
  --asr_batch_size 1 \
  --asr_chunk_length_s 20 \
  --asr_stride_length_s 3 \
  --hf_token "$HF_TOKEN"
```

Notes:

- this path was introduced to reduce the problems caused by external chunk splitting
- the ASR model used in this project was `Flurin17/whisper-large-v3-turbo-swiss-german`
- diarization still depends on the WhisperX / pyannote environment and may require environment fixes on a fresh server

### Stage B. Post-process the full-file ASR JSON

File:

- `postprocess_fullfile_asr.py`

Purpose:

- split overly long segments using word-level timestamps and punctuation
- fill missing speaker labels
- normalize language labels

Recommended environment:

```bash
source ~/cineminds/data-env/bin/activate
```

Typical command pattern:

```bash
python V5/postprocess_fullfile_asr.py \
  --input_json /path/to/Martin_Rapold_16k_mono_whisperx.json \
  --output_json /path/to/Martin_Rapold_16k_mono_whisperx_postprocessed.json \
  --max_segment_sec 20 \
  --target_segment_sec 12 \
  --min_segment_sec 4 \
  --default_language de \
  --sort_output
```

Current best output of this stage:

- `results/01_asr_input/Martin_Rapold_16k_mono_whisperx_postprocessed.json`

### Stage C. Topic modeling and merge-back

Files:

- `bertopic_from_combined_json.py`
- `merge_chunk_topics_ner_back_to_whisper_json.py`

Purpose:

- chunk the transcript for topic modeling
- run BERTopic on chunk text
- write `chunk_id`, `topic_id`, and `topic_words` back to the whisper-style JSON

Recommended environment:

```bash
source ~/cineminds/topic-env/bin/activate
```

Important note:

- this topic stage is inherited from the earlier codebase
- the current handoff includes the best ready-to-use topic+NER intermediate, so rerunning topic modeling is optional unless you explicitly want to change the topic stage
- if you do rerun topic modeling, confirm the chunk size you want to use and keep it consistent between BERTopic chunking and merge-back

Current best ready-to-use downstream file after topic + NER:

- `results/02_topics_ner/Martin_Rapold_16k_mono_topics_merged_llm_ner_v3.json`

### Stage D. LLM NER v3

File:

- `llm_ner_on_whisper_json_v3.py`

Purpose:

- run the current hardened LLM NER pass
- write entities into `entities_v3`

Recommended environment:

```bash
source ~/cineminds/topic-env/bin/activate
```

Typical command pattern:

```bash
python V5/llm_ner_on_whisper_json_v3.py \
  --input_json /path/to/topic_merged.json \
  --output_json /path/to/topic_merged_llm_ner_v3.json \
  --model MiniMax-M2.7 \
  --base_url https://api.minimax.io/v1 \
  --api_key "$MINIMAX_API_KEY" \
  --overwrite_existing
```

Notes:

- this script expects an OpenAI-compatible API interface
- in this project it was used with MiniMax via `base_url` + API key

### Stage E. Build final enriched WhisperX JSON

File:

- `build_whisperx_enriched_v2.py`

Purpose:

- create the final enriched JSON structure
- add sentence-level fields
- write word-level `ner_type`, `ner_subtype`, `ner_id`
- keep multi-token entities under one shared ID
- attach chunk tags and repeat them at sentence level

Recommended environment:

```bash
source ~/cineminds/topic-env/bin/activate
```

Typical command pattern:

```bash
python V5/build_whisperx_enriched_v2.py \
  --input_json /path/to/Martin_Rapold_16k_mono_topics_merged_llm_ner_v3.json \
  --entity_fields entities_v3 \
  --glossary_tags_json V5/resources/glossary/glossary_tag_dictionary.json \
  --ner_taxonomy_json V5/schemas/cineminds_ner_taxonomy.json \
  --output_json /path/to/final_whisper_style_enriched_v2_fullfile_fix2.json
```

Current best output of this stage:

- `results/03_final/final_whisper_style_enriched_v2_fullfile_fix2.json`

### Stage F. Subtype refinement

File:

- `llm_entity_subtype_on_enriched_v2.py`

Purpose:

- refine `ner_subtype` at the entity level
- optionally correct some `entity_type` values when supported by knowledge hints
- write refined subtype results back into both top-level entities and word-level annotations

Recommended environment:

```bash
source ~/cineminds/topic-env/bin/activate
```

Typical command pattern:

```bash
python V5/llm_entity_subtype_on_enriched_v2.py \
  --input_json /path/to/final_whisper_style_enriched_v2_fullfile_fix2.json \
  --output_json /path/to/final_whisper_style_enriched_v2_fullfile_subtyped_v3.json \
  --ner_taxonomy_json V5/schemas/cineminds_ner_taxonomy.json \
  --entity_knowledge_json V5/schemas/entity_knowledge_hints.json \
  --model MiniMax-M2.7 \
  --base_url https://api.minimax.io/v1 \
  --api_key "$MINIMAX_API_KEY" \
  --min_confidence 0.45
```

Current best output of this stage:

- `results/03_final/final_whisper_style_enriched_v2_fullfile_subtyped_v3.json`

## Glossary Resources

Files:

- `extract_glossary_tags.py`
- `resources/glossary/Glossary_BordwellThompson.xlsx`
- `resources/glossary/glossary_records.json`
- `resources/glossary/glossary_tag_dictionary.json`
- `resources/glossary/glossary_tag_dictionary.csv`
- `resources/glossary/glossary_tag_vocabulary.json`

Purpose:

- extract a clean glossary table from the Excel file
- build a machine-readable tag dictionary
- keep a CSV export for manual inspection

Typical command pattern:

```bash
source ~/cineminds/data-env/bin/activate

python V5/extract_glossary_tags.py \
  --input_xlsx V5/resources/glossary/Glossary_BordwellThompson.xlsx \
  --output_dir /path/to/glossary_exports
```

What each file is for:

- `glossary_records.json`: clean structured glossary records
- `glossary_tag_dictionary.json`: main machine-readable tag dictionary used downstream
- `glossary_tag_dictionary.csv`: spreadsheet-style export for manual checking
- `glossary_tag_vocabulary.json`: lightweight vocabulary grouped by `term`, `field`, `register`

## What Is Already Working

The following are already in place in the current bundle:

- full-file ASR path for Swiss German
- ASR postprocessing
- glossary extraction into structured resources
- final enriched JSON structure
- sentence-level fields and word-level fields
- shared `ner_id` for multi-token entities
- glossary tags repeated from chunk level to sentence level
- subtype refinement pass with taxonomy + knowledge hints

## What Is Still Partial

These are the main known limitations and the likely next-step work.

### 1. Glossary tagging is still mostly lexical

Current status:

- useful as a first pass
- structure is in place
- but it is not yet a true LLM semantic chunk-tagging stage

Main issue:

- high-frequency common words can still trigger false-positive tags

Examples previously observed:

- `natürlich`
- `fast`
- `schnell`
- `warm`

Recommended next step:

- replace pure lexical matching with `candidate retrieval + LLM semantic decision`

### 2. Taxonomy coverage is still limited

Current status:

- the current taxonomy works for many people, organizations, and titles
- but some film-domain or culture-domain cases still do not fit cleanly

Examples:

- `Dogma`
- `Papa Moll`
- `Monty Python`
- `Method Acting`

Recommended next step:

- expand taxonomy only after deciding whether these belong to NER or glossary/concept tagging

### 3. Some word-level boundaries are improved but not perfect

Current status:

- much better than the old rechunked final file
- still some residual punctuation-boundary cases remain

Examples:

- quoted titles
- split names such as `Daniel D.` / `Lewis`

## Recommended Starting Points For The Next Teammate

There are three practical ways to continue work.

### Option 1. Continue from the current best final output

Start here if the goal is analysis, error review, subtype refinement, or documentation:

- `results/03_final/final_whisper_style_enriched_v2_fullfile_subtyped_v3.json`

### Option 2. Continue from the final output before subtype

Start here if the goal is to redesign subtype logic:

- `results/03_final/final_whisper_style_enriched_v2_fullfile_fix2.json`

### Option 3. Continue from the cleaned transcript input

Start here if the goal is to rerun topic modeling or NER from a cleaner base:

- `results/01_asr_input/Martin_Rapold_16k_mono_whisperx_postprocessed.json`

## Practical Notes

- `data-env` is for ASR and transcript postprocessing
- `topic-env` is for topic modeling, NER, final build, and subtype refinement
- `llm_ner_on_whisper_json_v3.py` and `llm_entity_subtype_on_enriched_v2.py` both assume an OpenAI-compatible API interface
- in this project, MiniMax was used through that interface
- the copied scripts in `V5` are the scripts that matter for the current best pipeline, not the full archive of all historical experiments
