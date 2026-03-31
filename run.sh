#!/bin/bash


#set -euo pipefail

# ============================================================================
# Activate DAVA Environment (First Task)
# ============================================================================

cd "$(dirname "$0")"

# Get DAVA Python path from config (try system python3, fallback to hardcoded)
DAVA_PYTHON_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['dava_python_path'])" 2>/dev/null || echo "/home/arfarh/DAVA/dava-env/bin/python")

# Activate DAVA environment
export PATH="$(dirname "$(dirname "$DAVA_PYTHON_PATH")")/bin:$PATH"

# ============================================================================
# Default Configuration
# ============================================================================

config_file="config.yaml"

# ============================================================================
# Load Configuration from YAML
# ============================================================================

# Load config and export all variables
eval "$("$DAVA_PYTHON_PATH" << 'PYTHON_EOF'
import yaml
import sys

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print('ERROR: Config file is empty', file=sys.stderr)
        sys.exit(1)
    
    for key, value in config.items():
        if value is None:
            value = ''
        elif isinstance(value, bool):
            value = 'true' if value else 'false'
        else:
            value = str(value).replace('"', '\\"')
        print(f'export {key}="{value}"')
except Exception as e:
    print(f'ERROR: Failed to load config: {e}', file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)"

# ============================================================================
# Validation
# ============================================================================

# Disable topic detection if Qwen Python not found
#[ "$enable_topic_detection" = "true" ] && [ ! -f "$qwen_python_path" ] && enable_topic_detection=false

# ============================================================================
# Print Configuration
# ============================================================================

echo "=========================================="
echo "DAVA Batch Processing Pipeline"
echo "=========================================="
echo "Config file:          $config_file"
echo "Stage:               ${stage:-1}"
echo "Stop stage:          ${stop_stage:-2}"

# Set defaults if not loaded from config
stage=${stage:-1}
stop_stage=${stop_stage:-2}
echo ""

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "[STAGE 1] Manifest Generation:"
    echo "  Data directory:      $data_dir"
    echo "  File extension:      $file_extension"
    echo "  Output manifest:     $csv_file"
    echo ""
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "[STAGE 2] Processing Pipeline:"
    echo "  CSV file:            $csv_file"
    echo "  Output directory:   $out_dir"
    echo "  GPU index:          $gpu_index"
    echo "  ASR backend:        $asr_backend"
    echo "  ASR model:          $asr_model"
    echo "  Batch size:         $batch_size"
    echo "  Language detection: $lang_detection_model"
    [ -n "$lang" ] && echo "  Language:            $lang" || echo "  Language:            auto-detect"
    [ -n "$num_speakers" ] && echo "  Number of speakers: $num_speakers" || echo "  Number of speakers: auto-detect"
    
    if [ "$enable_topic_detection" = "true" ]; then
        echo "  Topic detection:    ENABLED"
        echo "    Model:            $topic_model_id"
        echo "    Verbose:          $topic_verbose"
    else
        echo "  Topic detection:    DISABLED"
    fi
    echo ""
fi

# ============================================================================
# Setup Functions
# ============================================================================

create_output_directories() {
    echo "Creating output directories..."
    mkdir -p "$out_dir"
    mkdir -p "$out_dir/wav"
    echo ""
}

# ============================================================================
# Stage 1: Generate Manifest
# ============================================================================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "=========================================="
    echo "Stage 1: Generate Manifest"
    echo "=========================================="
    
    # Check if generate_mp4_manifest.py exists
    if [ ! -f "generate_mp4_manifest.py" ]; then
        echo "ERROR: generate_mp4_manifest.py not found"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$data_dir" ]; then
        echo "ERROR: Data directory not found: $data_dir"
        exit 1
    fi
    
    echo "Generating manifest for $file_extension files in $data_dir..."
    "$DAVA_PYTHON_PATH" generate_mp4_manifest.py \
        --dir "$data_dir" \
        --ext "$file_extension" \
        --output "$csv_file"
    
    if [ ! -f "$csv_file" ]; then
        echo "ERROR: Failed to generate manifest file"
        exit 1
    fi
    
    file_count=$(wc -l < "$csv_file" | tr -d ' ')
    echo "✓ Manifest generated: $csv_file ($file_count files)"
    echo ""
fi

# ============================================================================
# Stage 2: Process Files
# ============================================================================

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "=========================================="
    echo "Stage 2: Process Files"
    echo "=========================================="
    
    # Validate CSV file exists
    if [ ! -f "$csv_file" ]; then
        echo "ERROR: CSV file not found: $csv_file"
        echo "Please run Stage 1 first to generate the manifest."
        exit 1
    fi
    
    create_output_directories
    
    echo "Starting batch processing..."
    echo ""
    
    # Build command
    cmd=(
        "$DAVA_PYTHON_PATH" process_batch_gpu.py
        --csv_file "$csv_file"
        --out_dir "$out_dir"
        --batch_size "$batch_size"
        --lang "$lang"
        --num_speakers "$num_speakers"
        --hf_token "$hf_token"
        --gpu_index "$gpu_index"
        --asr_backend "$asr_backend"
        --asr_model "$asr_model"
        --lang_detection_model "$lang_detection_model"
    )
    
    # Add topic detection flags if enabled
    if [ "$enable_topic_detection" = "true" ]; then
        cmd+=(--enable_topic_detection)
        cmd+=(--topic_model_id "$topic_model_id")
        cmd+=(--qwen_python_path "$qwen_python_path")
        [ "$topic_verbose" = "true" ] && cmd+=(--topic_verbose)
    fi
    
    # Execute
    "${cmd[@]}"

fi

#exit 0

