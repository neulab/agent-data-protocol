#!/bin/bash
#SBATCH --job-name=dataset_download
#SBATCH --output=/home/%u/logs/output_%j.out
#SBATCH --error=/home/%u/logs/error_%j.err
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

# ============================================
# SLURM Script for Dataset Download and Conversion
# ============================================
# This script downloads and processes 8 datasets:
#   android_in_the_wild, androidcontrol, llava_plus, omniact,
#   weblinx, wonderbread, go-browse-wa, openhands
#
# PRE-REQUISITES (run on login node before submitting):
#   1. Install git-lfs to ~/.local/bin
#   2. Set up GCS credentials: gcloud auth application-default login --no-launch-browser
#   3. Create logs directory: mkdir -p logs

# ============================================
# Configuration
# ============================================
# REPO_DIR is the current working directory (where sbatch was run from)
# Can be overridden by setting REPO_DIR environment variable before running
REPO_DIR="${REPO_DIR:-$(pwd)}"
DATA_DIR=/data/user_data/josephl4
DATASETS_DIR=$DATA_DIR/datasets
LOGS_DIR=$REPO_DIR/logs
mkdir -p "$LOGS_DIR"

echo "========================================="
echo "Dataset Download and Conversion Script"
echo "Started at: $(date)"
echo "========================================="
echo "REPO_DIR: $REPO_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "DATASETS_DIR: $DATASETS_DIR"
echo "========================================="

# ============================================
# Strict Validation - Error out if requirements not met
# ============================================
echo ""
echo "Validating environment..."

# Check that /data exists and is accessible
if [ ! -d "/data" ]; then
    echo "ERROR: /data directory does not exist." >&2
    echo "This script must be run on a compute node where /data is mounted." >&2
    echo "If running via SLURM: sbatch $SCRIPT_PATH" >&2
    echo "If running locally for testing, /data must be available." >&2
    exit 1
fi

# Check that DATA_DIR is writable
if ! mkdir -p "$DATA_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create or access DATA_DIR: $DATA_DIR" >&2
    echo "Check that /data/user_data/josephl4 is accessible." >&2
    exit 1
fi

# Check that REPO_DIR exists and contains expected files
if [ ! -f "$REPO_DIR/datasets/llava_plus/extract_raw.py" ]; then
    echo "ERROR: REPO_DIR does not appear to be the agent-data-collection repository." >&2
    echo "Expected to find: $REPO_DIR/datasets/llava_plus/extract_raw.py" >&2
    echo "REPO_DIR: $REPO_DIR" >&2
    echo "Make sure to run 'sbatch slurm_download_datasets.sh' from the repo directory," >&2
    echo "or set REPO_DIR environment variable: REPO_DIR=/path/to/repo sbatch ..." >&2
    exit 1
fi

# Check that virtual environment exists
if [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $REPO_DIR/.venv" >&2
    echo "Please create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

echo "Validation passed."

# ============================================
# Activate Virtual Environment
# ============================================
echo ""
echo "Activating virtual environment..."
source "$REPO_DIR/.venv/bin/activate"

# Verify pip is available
if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found after activating virtual environment." >&2
    exit 1
fi
echo "Virtual environment activated. Python: $(which python)"

# ============================================
# Environment Variables
# ============================================
# HuggingFace
export HF_HOME=$DATA_DIR/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

# GCS credentials (set up on login node beforehand)
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json

# Git LFS (installed to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Python imports
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

# SSL certificates for GCS
export CURL_CA_BUNDLE=$(python -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")

echo "Environment variables set:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  PATH includes: $HOME/.local/bin"

# ============================================
# Create directories
# ============================================
echo ""
echo "Creating directories..."
mkdir -p $DATASETS_DIR
mkdir -p $DATA_DIR/.hf_cache

# ============================================
# Install Python dependencies
# ============================================
echo ""
echo "Installing Python dependencies..."
pip install tensorflow certifi huggingface_hub "datasets>=2.14.0" "pyarrow<21.0.0" "fsspec==2023.9.2" markdown tqdm protobuf pillow

# Install browsergym-core for openhands processing
echo "Installing browsergym-core..."
if ! pip install browsergym-core; then
    echo "ERROR: browsergym-core installation failed." >&2
    exit 1
fi

# Install android_env_utils from repo
echo "Installing android_env_utils..."
if ! pip install -e $REPO_DIR/datasets/androidcontrol/android_env_utils/; then
    echo "ERROR: android_env_utils installation failed." >&2
    exit 1
fi

# Install Playwright Python package (for utility functions only - no browser needed for data conversion)
echo "Installing Playwright Python package..."
if ! pip install playwright; then
    echo "ERROR: Playwright installation failed." >&2
    exit 1
fi
# Note: We skip 'playwright install chromium --with-deps' as it requires sudo.
# Browser binaries aren't needed for data conversion, only the Python utilities.

# ============================================
# Create symlinks from repo to data directory
# ============================================
echo ""
echo "Setting up symlinks..."

# For each dataset, create symlink for screenshots/data storage
for dataset in android_in_the_wild androidcontrol llava_plus omniact weblinx wonderbread go-browse-wa openhands; do
    if ! mkdir -p "$DATASETS_DIR/$dataset/screenshots"; then
        echo "ERROR: Failed to create directory $DATASETS_DIR/$dataset/screenshots" >&2
        exit 1
    fi
    # Remove existing symlink or directory if it exists
    if [ -L "$REPO_DIR/datasets/$dataset/screenshots" ]; then
        rm "$REPO_DIR/datasets/$dataset/screenshots"
    fi
    # Create symlink
    if ! ln -sf "$DATASETS_DIR/$dataset/screenshots" "$REPO_DIR/datasets/$dataset/screenshots"; then
        echo "ERROR: Failed to create symlink for $dataset/screenshots" >&2
        exit 1
    fi
    echo "  Linked: $dataset/screenshots -> $DATASETS_DIR/$dataset/screenshots"
done

# ============================================
# Dataset Processing Function
# ============================================
process_dataset() {
    local name=$1
    local is_web=$2
    local api_env=$3
    local extra_args="${@:4}"
    local error_log="$LOGS_DIR/${name}_error.log"

    echo ""
    echo "========================================="
    echo "Processing: $name"
    echo "Started at: $(date)"
    echo "========================================="

    cd $REPO_DIR
    mkdir -p "$DATASETS_DIR/$name/full_sft"

    # Clear previous error log
    > "$error_log"

    # RAW extraction
    echo "[$name] Extracting raw data..."
    if ! python datasets/$name/extract_raw.py $extra_args > "$DATASETS_DIR/$name/full_raw.jsonl" 2>> "$error_log"; then
        echo "[$name] ERROR in extract_raw.py" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
        return 1
    fi
    local raw_lines=$(wc -l < "$DATASETS_DIR/$name/full_raw.jsonl" 2>/dev/null || echo 0)
    echo "[$name] Raw extraction complete. Lines: $raw_lines"
    if [ "$raw_lines" -eq 0 ]; then
        echo "[$name] WARNING: No raw data extracted!" >&2
        echo "No raw data extracted" >> "$error_log"
        return 1
    fi

    # STD conversion
    echo "[$name] Converting to standardized format..."
    if ! cat "$DATASETS_DIR/$name/full_raw.jsonl" | python datasets/$name/raw_to_standardized.py > "$DATASETS_DIR/$name/full_std.jsonl" 2>> "$error_log"; then
        echo "[$name] ERROR in raw_to_standardized.py" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
        return 1
    fi
    local std_lines=$(wc -l < "$DATASETS_DIR/$name/full_std.jsonl" 2>/dev/null || echo 0)
    echo "[$name] Standardization complete. Lines: $std_lines"
    if [ "$std_lines" -eq 0 ]; then
        echo "[$name] WARNING: No standardized data produced!" >&2
        echo "No standardized data produced" >> "$error_log"
        return 1
    fi

    # SFT conversion (openhands)
    export MY_DATASET=$name
    echo "[$name] Converting to SFT format (openhands)..."
    cat "$DATASETS_DIR/$name/full_std.jsonl" | python agents/openhands/std_to_sft.py --is_web=$is_web --api_env=$api_env > "$DATASETS_DIR/$name/full_sft/full_sft_openhands.jsonl" 2>> "$error_log"
    local sft_lines=$(wc -l < "$DATASETS_DIR/$name/full_sft/full_sft_openhands.jsonl" 2>/dev/null || echo 0)
    echo "[$name] OpenHands SFT complete. Lines: $sft_lines"
    if [ "$sft_lines" -eq 0 ]; then
        echo "[$name] WARNING: SFT produced 0 lines!" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
    fi

    echo "[$name] Complete at $(date)!"
    return 0
}

# ============================================
# Download weblinx (requires git lfs)
# ============================================
echo ""
echo "========================================="
echo "Downloading weblinx dataset (git lfs)..."
echo "========================================="

# Verify git-lfs is available
if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs not found." >&2
    echo "See SETUP_PREREQUISITES.md for installing git-lfs to ~/.local/bin" >&2
    exit 1
fi

WEBLINX_READY=false
cd "$DATASETS_DIR"
# rm -rf weblinx
if [ ! -d "weblinx/WebLINX-full" ]; then
    if ! mkdir -p weblinx; then
        echo "ERROR: Failed to create weblinx directory" >&2
        exit 1
    fi
    cd weblinx
    echo "Cloning WebLINX-full repository..."
    if ! git clone --progress https://huggingface.co/datasets/McGill-NLP/WebLINX-full; then
        echo "ERROR: git clone failed for weblinx" >&2
        exit 1
    fi
    cd WebLINX-full
    echo "Pulling LFS files (excluding large files)..."
    if ! git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png"; then
        echo "ERROR: git lfs pull failed for weblinx" >&2
        exit 1
    fi
    echo "WebLINX download complete!"
    WEBLINX_READY=true
else
    echo "WebLINX already downloaded, skipping..."
    WEBLINX_READY=true
fi

if [ "$WEBLINX_READY" = true ]; then
    # Create symlink in repo
    # Handle existing path: remove if symlink, warn if directory
    if [ -L "$REPO_DIR/datasets/weblinx/WebLINX-full" ]; then
        rm "$REPO_DIR/datasets/weblinx/WebLINX-full"
    elif [ -d "$REPO_DIR/datasets/weblinx/WebLINX-full" ]; then
        echo "WARNING: $REPO_DIR/datasets/weblinx/WebLINX-full is a directory, not a symlink." >&2
        echo "Skipping symlink creation. Remove manually if this is not the data location." >&2
    fi
    if [ ! -d "$REPO_DIR/datasets/weblinx/WebLINX-full" ]; then
        if ! ln -sf "$DATASETS_DIR/weblinx/WebLINX-full" "$REPO_DIR/datasets/weblinx/WebLINX-full"; then
            echo "ERROR: Failed to create WebLINX symlink" >&2
            exit 1
        fi
        echo "Symlink created: $REPO_DIR/datasets/weblinx/WebLINX-full -> $DATASETS_DIR/weblinx/WebLINX-full"
    fi
fi

# Return to repo directory
cd "$REPO_DIR"

# ============================================
# Process datasets in order
# ============================================
# Start with simpler HuggingFace datasets, then more complex ones

echo ""
echo "========================================="
echo "Starting dataset processing..."
echo "========================================="

# Track successes and failures
FAILED_DATASETS=""
SUCCEEDED_DATASETS=""

# Simple HuggingFace datasets first
for dataset_info in "llava_plus|no|execute_bash" "omniact|no|execute_bash" "go-browse-wa|yes|browser" "openhands|no|execute_bash"; do
    IFS='|' read -r name is_web api_env <<< "$dataset_info"
    if process_dataset "$name" "$is_web" "$api_env"; then
        SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS $name"
    else
        FAILED_DATASETS="$FAILED_DATASETS $name"
    fi
done

# wget-based dataset
export WONDERBREAD_ROOT=$DATASETS_DIR/wonderbread
if process_dataset "wonderbread" "yes" "browser"; then
    SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS wonderbread"
else
    FAILED_DATASETS="$FAILED_DATASETS wonderbread"
fi

# weblinx dataset
if process_dataset "weblinx" "yes" "browser"; then
    SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS weblinx"
else
    FAILED_DATASETS="$FAILED_DATASETS weblinx"
fi

# GCS-based datasets (require credentials)
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "ERROR: GCS credentials not found at $GOOGLE_APPLICATION_CREDENTIALS" >&2
    echo "Required for android_in_the_wild and androidcontrol datasets." >&2
    echo "Set up credentials with: gcloud auth application-default login --no-launch-browser" >&2
    exit 1
fi

echo ""
echo "GCS credentials found, processing android datasets..."

if process_dataset "android_in_the_wild" "no" "execute_bash" "--limit=0" "--output-dir=$DATASETS_DIR/android_in_the_wild/screenshots"; then
    SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS android_in_the_wild"
else
    FAILED_DATASETS="$FAILED_DATASETS android_in_the_wild"
fi

export DATASET_OUTPUT_DIR=$DATASETS_DIR/androidcontrol/screenshots
if process_dataset "androidcontrol" "no" "execute_bash"; then
    SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS androidcontrol"
else
    FAILED_DATASETS="$FAILED_DATASETS androidcontrol"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "========================================="
echo "Processing Complete!"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Succeeded:$SUCCEEDED_DATASETS"
echo "Failed:$FAILED_DATASETS"
echo ""
echo "Output files are in: $DATASETS_DIR/<name>/"
echo "  - full_raw.jsonl"
echo "  - full_std.jsonl"
echo "  - full_sft/full_sft_openhands.jsonl"
echo ""
echo "Logs are in: $LOGS_DIR/"
echo "========================================="
