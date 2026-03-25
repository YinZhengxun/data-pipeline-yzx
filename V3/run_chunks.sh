#!/bin/bash

for f in /home/zhyin/cineminds/data/martin_chunks/*.wav; do
  base=$(basename "$f" .wav)

  echo "[RUN] $base"

  out_dir="/home/zhyin/cineminds/martin_chunk_outputs/output_${base}"

  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python process_single_file_pipeline_AG.py \
    --episode_path "$f" \
    --out_dir "$out_dir" \
    --gpu_index 1 \
    --asr_backend transformers \
    --asr_model openai/whisper-small \
    --language de \
    --hf_token "token"

  cp "$out_dir"/json/*_whisperx.json /home/zhyin/cineminds/martin_raw_json/

done
