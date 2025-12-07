#!/bin/bash
#SBATCH --job-name=dataset_download
#SBATCH --output=/home/%u/logs/sbatch/output_%j.out
#SBATCH --error=/home/%u/logs/sbatch/error_%j.err
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
#   3. Create logs directory: mkdir -p ~/logs/sbatch
#
# USAGE:
#   1. Update <FILL_IN_PARTITION> with your partition name
#   2. sbatch slurm_download_datasets.sh
# ============================================

# NOTE: Not using 'set -e' so that failures in one dataset don't stop others

# ============================================
# Configuration
# ============================================
DATA_DIR=/data/user_data/josephl4
REPO_DIR=$HOME/agent-data-collection
DATASETS_DIR=$DATA_DIR/datasets

echo "========================================="
echo "Dataset Download and Conversion Script"
echo "Started at: $(date)"
echo "========================================="
echo "DATA_DIR: $DATA_DIR"
echo "REPO_DIR: $REPO_DIR"
echo "DATASETS_DIR: $DATASETS_DIR"
echo "========================================="

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
pip install --user tensorflow certifi huggingface_hub datasets markdown tqdm protobuf pillow

# Install browsergym-core for openhands processing
echo "Installing browsergym-core..."
pip install --user browsergym-core || echo "Warning: browsergym-core installation failed, openhands may not work"

# Install android_env_utils from repo
echo "Installing android_env_utils..."
pip install --user -e $REPO_DIR/datasets/androidcontrol/android_env_utils/ || echo "Warning: android_env_utils installation failed"

# Try to install Playwright (may fail without root, but utils may still work)
echo "Attempting to install Playwright..."
pip install --user playwright || true
playwright install chromium --with-deps 2>/dev/null || echo "Note: Playwright browser installation skipped (may not be needed)"

# ============================================
# Create symlinks from repo to data directory
# ============================================
echo ""
echo "Setting up symlinks..."

# For each dataset, create symlink for screenshots/data storage
for dataset in android_in_the_wild androidcontrol llava_plus omniact weblinx wonderbread go-browse-wa openhands; do
    mkdir -p $DATASETS_DIR/$dataset/screenshots
    # Remove existing symlink or directory if it exists
    if [ -L "$REPO_DIR/datasets/$dataset/screenshots" ]; then
        rm "$REPO_DIR/datasets/$dataset/screenshots"
    fi
    # Create symlink
    ln -sf $DATASETS_DIR/$dataset/screenshots $REPO_DIR/datasets/$dataset/screenshots 2>/dev/null || true
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
    local error_log="$REPO_DIR/datasets/$name/error.log"

    echo ""
    echo "========================================="
    echo "Processing: $name"
    echo "Started at: $(date)"
    echo "========================================="

    cd $REPO_DIR
    mkdir -p datasets/$name/full_sft

    # Clear previous error log
    > "$error_log"

    # RAW extraction
    echo "[$name] Extracting raw data..."
    if ! python datasets/$name/extract_raw.py $extra_args > datasets/$name/full_raw.jsonl 2>> "$error_log"; then
        echo "[$name] ERROR in extract_raw.py" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
        return 1
    fi
    local raw_lines=$(wc -l < datasets/$name/full_raw.jsonl 2>/dev/null || echo 0)
    echo "[$name] Raw extraction complete. Lines: $raw_lines"
    if [ "$raw_lines" -eq 0 ]; then
        echo "[$name] WARNING: No raw data extracted!" >&2
        echo "No raw data extracted" >> "$error_log"
        return 1
    fi

    # STD conversion
    echo "[$name] Converting to standardized format..."
    if ! cat datasets/$name/full_raw.jsonl | python datasets/$name/raw_to_standardized.py > datasets/$name/full_std.jsonl 2>> "$error_log"; then
        echo "[$name] ERROR in raw_to_standardized.py" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
        return 1
    fi
    local std_lines=$(wc -l < datasets/$name/full_std.jsonl 2>/dev/null || echo 0)
    echo "[$name] Standardization complete. Lines: $std_lines"
    if [ "$std_lines" -eq 0 ]; then
        echo "[$name] WARNING: No standardized data produced!" >&2
        echo "No standardized data produced" >> "$error_log"
        return 1
    fi

    # SFT conversion (openhands) - continue even if this fails
    echo "[$name] Converting to SFT format (openhands)..."
    if ! cat datasets/$name/full_std.jsonl | python agents/openhands/std_to_sft.py --is_web=$is_web --api_env=$api_env > datasets/$name/full_sft/full_sft_openhands.jsonl 2>> "$error_log"; then
        echo "[$name] Warning: openhands SFT conversion had issues" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
    fi
    echo "[$name] OpenHands SFT complete. Lines: $(wc -l < datasets/$name/full_sft/full_sft_openhands.jsonl 2>/dev/null || echo 0)"

    # SFT conversion (agentlab) - continue even if this fails
    echo "[$name] Converting to SFT format (agentlab)..."
    if ! cat datasets/$name/full_std.jsonl | python agents/agentlab/std_to_sft.py > datasets/$name/full_sft/full_sft_agentlab.jsonl 2>> "$error_log"; then
        echo "[$name] Warning: agentlab SFT conversion had issues" >&2
        echo "[$name] === Error log ===" >&2
        cat "$error_log" >&2
        echo "[$name] === End error log ===" >&2
    fi
    echo "[$name] AgentLab SFT complete. Lines: $(wc -l < datasets/$name/full_sft/full_sft_agentlab.jsonl 2>/dev/null || echo 0)"

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

WEBLINX_READY=false

# Verify git-lfs is available
if ! command -v git-lfs &> /dev/null; then
    echo "WARNING: git-lfs not found. weblinx dataset will be skipped." >&2
    echo "See SETUP_PREREQUISITES.md for installing git-lfs to ~/.local/bin" >&2
else
    cd $DATASETS_DIR
    if [ ! -d "weblinx/WebLINX-full" ]; then
        mkdir -p weblinx
        cd weblinx
        echo "Cloning WebLINX-full repository..."
        if git clone https://huggingface.co/datasets/McGill-NLP/WebLINX-full 2>&1; then
            cd WebLINX-full
            echo "Pulling LFS files (excluding large files)..."
            if git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png" 2>&1; then
                echo "WebLINX download complete!"
                WEBLINX_READY=true
            else
                echo "WARNING: git lfs pull failed for weblinx" >&2
            fi
        else
            echo "WARNING: git clone failed for weblinx" >&2
        fi
    else
        echo "WebLINX already downloaded, skipping..."
        WEBLINX_READY=true
    fi

    if [ "$WEBLINX_READY" = true ]; then
        # Create symlink in repo
        ln -sf $DATASETS_DIR/weblinx/WebLINX-full $REPO_DIR/datasets/weblinx/WebLINX-full
        echo "Symlink created: $REPO_DIR/datasets/weblinx/WebLINX-full -> $DATASETS_DIR/weblinx/WebLINX-full"
    fi
fi

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
if process_dataset "wonderbread" "yes" "browser"; then
    SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS wonderbread"
else
    FAILED_DATASETS="$FAILED_DATASETS wonderbread"
fi

# Pre-downloaded weblinx (only if download succeeded)
if [ "$WEBLINX_READY" = true ]; then
    if process_dataset "weblinx" "yes" "browser"; then
        SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS weblinx"
    else
        FAILED_DATASETS="$FAILED_DATASETS weblinx"
    fi
else
    echo ""
    echo "[weblinx] Skipping - download was not successful"
    FAILED_DATASETS="$FAILED_DATASETS weblinx"
fi

# GCS-based datasets (require credentials)
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo ""
    echo "GCS credentials found, processing android datasets..."

    if process_dataset "android_in_the_wild" "no" "execute_bash" "--limit=0"; then
        SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS android_in_the_wild"
    else
        FAILED_DATASETS="$FAILED_DATASETS android_in_the_wild"
    fi

    if process_dataset "androidcontrol" "no" "execute_bash"; then
        SUCCEEDED_DATASETS="$SUCCEEDED_DATASETS androidcontrol"
    else
        FAILED_DATASETS="$FAILED_DATASETS androidcontrol"
    fi
else
    echo ""
    echo "WARNING: GCS credentials not found at $GOOGLE_APPLICATION_CREDENTIALS"
    echo "Skipping android_in_the_wild and androidcontrol datasets."
    echo "To process these, set up credentials with: gcloud auth application-default login --no-launch-browser"
    FAILED_DATASETS="$FAILED_DATASETS android_in_the_wild androidcontrol"
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
echo "Output files are in: $REPO_DIR/datasets/<name>/"
echo "  - full_raw.jsonl"
echo "  - full_std.jsonl"
echo "  - full_sft/full_sft_openhands.jsonl"
echo "  - full_sft/full_sft_agentlab.jsonl"
echo "========================================="
