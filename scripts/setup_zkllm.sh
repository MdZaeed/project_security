#!/usr/bin/env bash
# Setup script for zkLLM on A100 servers.
# Run from the repo root:  bash scripts/setup_zkllm.sh
#
# Assumes system CUDA (12.x) is already loaded (e.g. module load cuda/12.8).
# Uses system nvcc rather than installing CUDA via conda.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZKLLM_DIR="$SCRIPT_DIR/../zkllm"
ENV_NAME="zkllm-env"

# Verify system nvcc is available
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Load the CUDA module first, e.g.: module load cuda/12.8"
    exit 1
fi
echo "Using system nvcc: $(nvcc --version | head -1)"

echo ""
echo "=== [1/3] Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.11 -y

echo ""
echo "=== [2/3] Installing Python dependencies ==="
conda run -n "$ENV_NAME" pip install \
    torch torchvision torchaudio \
    "transformers==4.36.2" \
    datasets \
    huggingface_hub

echo ""
echo "=== [3/3] Building CUDA code (targeting sm_80 for A100) ==="
cd "$ZKLLM_DIR"

# Apply compatibility patches (kept in parent repo since we can't push to upstream)
cp "$SCRIPT_DIR/../zkllm-patches/llama-self-attn.py" "$ZKLLM_DIR/llama-self-attn.py"
# Patch GPU architecture for A100 (upstream targets sm_86)
sed -i 's/ARCH := sm_86/ARCH := sm_80/' Makefile
# Use system nvcc instead of conda's
sed -i 's|NVCC := $(CONDA_PREFIX)/bin/nvcc|NVCC := nvcc|' Makefile
# Use system CUDA headers/libs; keep conda prefix for Python-related includes
sed -i 's|INCLUDES := -I$(CONDA_PREFIX)/include|INCLUDES := -I$(CONDA_PREFIX)/include -I$(CUDA_HOME)/include|' Makefile
sed -i 's|LIBS := -L$(CONDA_PREFIX)/lib|LIBS := -L$(CONDA_PREFIX)/lib -L$(CUDA_HOME)/lib64|' Makefile
make clean && make all -j"$(nproc)"

echo ""
echo "=== Setup complete ==="
echo "Activate with:        conda activate $ENV_NAME"
echo "Download model with:  python zkllm/download-models.py meta-llama/Llama-2-7b-hf <HF_TOKEN>"
