#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vllm-%j.log
#SBATCH --error=vllm-%j.err

# Activate your vLLM environment
source .venv/bin/activate

# Set up HuggingFace cache paths (as per your cluster's guide)
export HF_HOME=/data/user_data/${USER}/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1  # Use cached model if available

# Create personal HF directory if it doesn't exist
mkdir -p ${HF_HOME}

# Get compute node hostname for SSH tunneling
COMPUTE_NODE=$(hostname -f)
PORT=8000
REMOTE_PORT=9127

echo "========================================"
echo "vLLM Server Starting"
echo "========================================"
echo "Time: $(date)"
echo "Node: ${COMPUTE_NODE}"
echo "Port: ${PORT}"
echo ""
echo "To connect from your local machine:"
echo "----------------------------------------"
echo "1. Open a new terminal on your laptop"
echo "2. Run this SSH tunnel command:"
echo ""
echo "   ssh -L 8000:${COMPUTE_NODE}:8000 ${USER}@YOUR_LOGIN_NODE.edu"
echo ""
echo "3. Keep that terminal open"
echo "4. Access the API at: http://localhost:8000"
echo ""
echo "Test with:"
echo "   curl http://localhost:8000/v1/models"
echo "========================================"

# Run vLLM server
#export VLLM_LOGGING_CONFIG_PATH=/home/josephl4/agent-data-protocol/vllm_logging_config.json
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port ${PORT} \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --enable-log-outputs \
    --enable-log-requests \
    --tool-call-parser hermes &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"
sleep 30

echo "Creating reverse tunnel to expose service..."
ssh -R ${EXPOSED_PORT}:localhost:${PORT} \
    -o ServerAliveInterval=60 \
    -o StrictHostKeyChecking=no \
    josephl4@login.babel.cs.cmu.edu &

SSH_PID=$!

echo "========================================"
echo "vLLM API is available at:"
echo "http://login.babel.cs.cmu.edu:${EXPOSED_PORT}"
echo "========================================"
echo "Test with:"
echo "curl http://login.babel.cs.cmu.edu:${EXPOSED_PORT}/v1/models"
echo "========================================"

# Wait for vLLM to finish
wait $VLLM_PID


echo "vLLM server stopped at $(date)"
