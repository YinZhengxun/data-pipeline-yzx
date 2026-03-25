#!/bin/bash

set -euo pipefail

# ============================================================================
# Configuration (can be overridden by command line or config.yaml)
# ============================================================================

# Default values
DATA_DIR=${1:-"/home/arfarh/DAVA/Conservatives_s1.mp4"}  # Can be file or directory
LANG=${2:-"en"}
NUM_SPEAKERS=${3:-""}
OUT_DIR=${4:-"/home/arfarh/DAVA/out_my_pipeline"}
HF_TOKEN=${5:-""}
GPU_INDEX=${6:-"6"}
NER_MODEL=${7:-"bert"}  # "flair" or "bert"
DO_TOPICS=${8:-"True"}  # "true" or empty to enable/disable topic modeling
ASR_BACKEND=${9:-"openai"}  # "transformers" | "openai" | "faster-whisper"
ASR_MODEL=${10:-"openai/whisper-large-v3"}  # e.g., "distil-whisper/distil-large-v3.5" or "Systran/faster-whisper-large-v3"
STAGE=${11:-"1"}  # Stage to start (1: manifest, 2: processing)
STOP_STAGE=${12:-"2"}  # Stage to stop

cd "$(dirname "$0")"

# Get DAVA Python path (try from config, fallback to system python3)
DAVA_PYTHON_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['dava_python_path'])" 2>/dev/null || echo "python3")

# ============================================================================
# Determine if DATA_DIR is a file or directory
# ============================================================================

if [ -f "$DATA_DIR" ]; then
    # Single file mode
    SINGLE_FILE_MODE=true
    EPISODE_PATH="$DATA_DIR"
    DATA_DIR=$(dirname "$DATA_DIR")
    CSV_FILE="${OUT_DIR}/.temp_single_file_manifest.csv"
    echo "$EPISODE_PATH" > "$CSV_FILE"
    echo "Single file mode: $EPISODE_PATH"
elif [ -d "$DATA_DIR" ]; then
    # Directory mode - will generate manifest
    SINGLE_FILE_MODE=false
    CSV_FILE="${OUT_DIR}/manifest.csv"
    echo "Directory mode: $DATA_DIR"
else
    echo "ERROR: $DATA_DIR is neither a file nor a directory"
    exit 1
fi

# ============================================================================
# Print Configuration
# ============================================================================

echo "=========================================="
echo "DAVA Complete Pipeline (with NER & Topics)"
echo "=========================================="
echo "Data source:         $DATA_DIR"
echo "Output directory:   $OUT_DIR"
echo "Language:           $LANG"
echo "GPU index:          $GPU_INDEX"
echo "ASR backend:        $ASR_BACKEND"
echo "ASR model:          $ASR_MODEL"
echo "NER model:          $NER_MODEL"
echo "Topic modeling:     $([ -n "$DO_TOPICS" ] && echo "Enabled" || echo "Disabled")"
echo "Stage:              $STAGE"
echo "Stop stage:         $STOP_STAGE"
echo ""

# ============================================================================
# Validate NER model choice
# ============================================================================

if [ "$NER_MODEL" != "flair" ] && [ "$NER_MODEL" != "bert" ]; then
    echo "Invalid NER_MODEL: $NER_MODEL (must be 'flair' or 'bert')"
    echo "Using default: bert"
    NER_MODEL="bert"
fi

# ============================================================================
# Set up cuDNN for faster-whisper if needed
# ============================================================================

if [ "$ASR_BACKEND" = "faster-whisper" ]; then
    echo "Setting up cuDNN for faster-whisper..."
    CUDNN_PATH=$(python3 -c "
try:
    import nvidia.cudnn
    import os
    if hasattr(nvidia.cudnn, '__path__'):
        base_path = nvidia.cudnn.__path__[0]
        lib_path = os.path.join(base_path, 'lib')
        print(lib_path if os.path.exists(lib_path) else base_path)
    else:
        print('')
except Exception:
    print('')
" 2>/dev/null)
    
    if [ -z "$CUDNN_PATH" ]; then
        echo "WARNING: Could not find nvidia.cudnn library path"
        echo "faster-whisper may require cuDNN. Consider installing:"
        echo "  pip install nvidia-cudnn-cu12"
    else
        echo "Found cuDNN at: $CUDNN_PATH"
        export LD_LIBRARY_PATH="$CUDNN_PATH:${LD_LIBRARY_PATH:-}"
    fi
    
    # Set environment variables for faster-whisper
    export FAST_WHISPER_COMPUTE_TYPE=${FAST_WHISPER_COMPUTE_TYPE:-"int8_float16"}
    export FAST_WHISPER_BEAM=${FAST_WHISPER_BEAM:-"1"}
    echo "Compute type: $FAST_WHISPER_COMPUTE_TYPE"
    echo "Beam size: $FAST_WHISPER_BEAM"
    echo ""
fi

# ============================================================================
# Stage 1: Generate Manifest (only if directory mode)
# ============================================================================

if [ "$SINGLE_FILE_MODE" = false ] && [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
    echo "=========================================="
    echo "Stage 1: Generate Manifest"
    echo "=========================================="
    
    # Check if generate_mp4_manifest.py exists
    if [ ! -f "generate_mp4_manifest.py" ]; then
        echo "ERROR: generate_mp4_manifest.py not found"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Create temporary manifests for each supported format
    temp_dir=$(dirname "$CSV_FILE")
    mkdir -p "$temp_dir"
    
    temp_mp4="${temp_dir}/.temp_mp4_manifest.csv"
    temp_mkv="${temp_dir}/.temp_mkv_manifest.csv"
    temp_webm="${temp_dir}/.temp_webm_manifest.csv"
    temp_m4a="${temp_dir}/.temp_m4a_manifest.csv"
    
    # Generate manifests for each format
    echo "Generating manifest for supported formats (mp4, mkv, webm, m4a)..."
    
    "$DAVA_PYTHON_PATH" generate_mp4_manifest.py --dir "$DATA_DIR" --ext "mp4" --output "$temp_mp4" 2>/dev/null || touch "$temp_mp4"
    "$DAVA_PYTHON_PATH" generate_mp4_manifest.py --dir "$DATA_DIR" --ext "mkv" --output "$temp_mkv" 2>/dev/null || touch "$temp_mkv"
    "$DAVA_PYTHON_PATH" generate_mp4_manifest.py --dir "$DATA_DIR" --ext "webm" --output "$temp_webm" 2>/dev/null || touch "$temp_webm"
    "$DAVA_PYTHON_PATH" generate_mp4_manifest.py --dir "$DATA_DIR" --ext "m4a" --output "$temp_m4a" 2>/dev/null || touch "$temp_m4a"
    
    # Combine all manifests (remove duplicates and empty files)
    cat "$temp_mp4" "$temp_mkv" "$temp_webm" "$temp_m4a" 2>/dev/null | grep -v '^$' | sort -u > "$CSV_FILE" || true
    
    # Clean up temporary files
    rm -f "$temp_mp4" "$temp_mkv" "$temp_webm" "$temp_m4a"
    
    if [ ! -f "$CSV_FILE" ] || [ ! -s "$CSV_FILE" ]; then
        echo "ERROR: No files found in $DATA_DIR (supported formats: mp4, mkv, webm, m4a)"
        exit 1
    fi
    
    file_count=$(wc -l < "$CSV_FILE" | tr -d ' ')
    echo "✓ Manifest generated: $CSV_FILE ($file_count files)"
    echo ""
fi

# ============================================================================
# Stage 2: Process Files
# ============================================================================

if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
    echo "=========================================="
    echo "Stage 2: Process Files"
    echo "=========================================="
    
    # Validate CSV file exists
    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: CSV file not found: $CSV_FILE"
        echo "Please run Stage 1 first to generate the manifest."
        exit 1
    fi
    
    # Create main output directory
    mkdir -p "$OUT_DIR"
    
    # Read files from manifest
    file_list=()
    while IFS= read -r line || [ -n "$line" ]; do
        file_path=$(echo "$line" | tr -d '\r\n' | xargs)
        if [ -n "$file_path" ] && [ -f "$file_path" ]; then
            file_list+=("$file_path")
        fi
    done < "$CSV_FILE"
    
    total_files=${#file_list[@]}
    
    if [ $total_files -eq 0 ]; then
        echo "ERROR: No valid files found in manifest"
        exit 1
    fi
    
    echo "Found $total_files files to process"
    echo ""
    
    # Process each file
    for i in "${!file_list[@]}"; do
        file_path="${file_list[$i]}"
        file_basename=$(basename "$file_path")
        file_name="${file_basename%.*}"  # Remove extension
        
        echo "=========================================="
        echo "[$((i+1))/$total_files] Processing: $file_basename"
        echo "=========================================="
        
        # Create per-file output directory
        file_out_dir="${OUT_DIR}/${file_name}"
        mkdir -p "$file_out_dir"
        
        # Build command for process_single_file_pipeline.py
        CMD="$DAVA_PYTHON_PATH process_single_file_pipeline.py"
        CMD="$CMD --episode_path \"$file_path\""
        CMD="$CMD --language \"$LANG\""
        CMD="$CMD --out_dir \"$file_out_dir\""
        CMD="$CMD --ner_model $NER_MODEL"
        CMD="$CMD --asr_backend $ASR_BACKEND"
        CMD="$CMD --asr_model \"$ASR_MODEL\""
        CMD="$CMD --gpu_index $GPU_INDEX"
        
        if [ -n "$HF_TOKEN" ]; then
            CMD="$CMD --hf_token \"$HF_TOKEN\""
        fi
        
        if [ -n "$NUM_SPEAKERS" ]; then
            CMD="$CMD --num_speakers $NUM_SPEAKERS"
        fi
        
        if [ -n "$DO_TOPICS" ]; then
            CMD="$CMD --do_topics"
        fi
        
        # Run the pipeline
        echo "Running: $CMD"
        echo ""
        eval $CMD
        
        # Reorganize output: move files from subdirectories to per-file directory
        echo ""
        echo "Reorganizing output files..."
        
        moved_count=0
        
        # Move JSON files
        if [ -d "${file_out_dir}/json" ]; then
            json_files=$(find "${file_out_dir}/json" -type f -name "*.json" 2>/dev/null | wc -l)
            if [ "$json_files" -gt 0 ]; then
                find "${file_out_dir}/json" -type f -name "*.json" -exec mv {} "$file_out_dir/" \; 2>/dev/null
                moved_count=$((moved_count + json_files))
            fi
            rmdir "${file_out_dir}/json" 2>/dev/null || true
        fi
        
        # Move TXT files
        if [ -d "${file_out_dir}/txt" ]; then
            txt_files=$(find "${file_out_dir}/txt" -type f -name "*.txt" 2>/dev/null | wc -l)
            if [ "$txt_files" -gt 0 ]; then
                find "${file_out_dir}/txt" -type f -name "*.txt" -exec mv {} "$file_out_dir/" \; 2>/dev/null
                moved_count=$((moved_count + txt_files))
            fi
            rmdir "${file_out_dir}/txt" 2>/dev/null || true
        fi
        
        # Move CoNLL-U files
        if [ -d "${file_out_dir}/conllu" ]; then
            conllu_files=$(find "${file_out_dir}/conllu" -type f -name "*.conllu" 2>/dev/null | wc -l)
            if [ "$conllu_files" -gt 0 ]; then
                find "${file_out_dir}/conllu" -type f -name "*.conllu" -exec mv {} "$file_out_dir/" \; 2>/dev/null
                moved_count=$((moved_count + conllu_files))
            fi
            rmdir "${file_out_dir}/conllu" 2>/dev/null || true
        fi
        
        # Move NER files
        if [ -d "${file_out_dir}/ner" ]; then
            ner_files=$(find "${file_out_dir}/ner" -type f -name "*.tsv" 2>/dev/null | wc -l)
            if [ "$ner_files" -gt 0 ]; then
                find "${file_out_dir}/ner" -type f -name "*.tsv" -exec mv {} "$file_out_dir/" \; 2>/dev/null
                moved_count=$((moved_count + ner_files))
            fi
            rmdir "${file_out_dir}/ner" 2>/dev/null || true
        fi
        
        # Topics directory stays as subdirectory (contains multiple files)
        if [ -d "${file_out_dir}/topics" ]; then
            echo "  Topics: ${file_out_dir}/topics/ (kept as subdirectory)"
        fi
        
        echo "✓ Output organized: $moved_count files moved to $file_out_dir"
        echo ""
    done
    
    echo "=========================================="
    echo "Pipeline complete!"
    echo "=========================================="
    echo "Output directory: $OUT_DIR"
    echo "  Each file has its own folder with all outputs"
    echo ""
fi

# Clean up temporary manifest if single file mode
if [ "$SINGLE_FILE_MODE" = true ] && [ -f "$CSV_FILE" ]; then
    rm -f "$CSV_FILE"
fi
