#!/usr/bin/env bash
# Run the full zkLLM proof pipeline for LLaMA-2 across all layers.
# Run from the zkllm/ directory:
#   bash ../scripts/run_zkllm_proof.sh
#
# Usage:
#   bash ../scripts/run_zkllm_proof.sh [model_size] [num_layers] [seq_len]
#
# Defaults: model_size=7, num_layers=32, seq_len=2048

set -euo pipefail

MODEL_SIZE=${1:-7}
NUM_LAYERS=${2:-32}
SEQ_LEN=${3:-2048}

echo "=== zkLLM Proof Pipeline ==="
echo "Model: LLaMA-2-${MODEL_SIZE}b | Layers: ${NUM_LAYERS} | Seq len: ${SEQ_LEN}"
echo ""

# Layer 0 starts from a randomly generated input (created by rmsnorm if missing)
CURRENT_INPUT="layer_input.bin"

for layer in $(seq 0 $((NUM_LAYERS - 1))); do
    echo "--- Layer ${layer} / $((NUM_LAYERS - 1)) ---"

    echo "[${layer}] RMSNorm (input)"
    python llama-rmsnorm.py $MODEL_SIZE $layer input $SEQ_LEN \
        --input_file "$CURRENT_INPUT" \
        --output_file attn_input.bin

    echo "[${layer}] Self-Attention"
    python llama-self-attn.py $MODEL_SIZE $layer $SEQ_LEN \
        --input_file attn_input.bin \
        --output_file attn_output.bin

    echo "[${layer}] Skip connection (post-attn)"
    python llama-skip-connection.py \
        --block_input_file "$CURRENT_INPUT" \
        --block_output_file attn_output.bin \
        --output_file post_attn_norm_input.bin

    echo "[${layer}] RMSNorm (post-attention)"
    python llama-rmsnorm.py $MODEL_SIZE $layer post_attention $SEQ_LEN \
        --input_file post_attn_norm_input.bin \
        --output_file ffn_input.bin

    echo "[${layer}] FFN"
    python llama-ffn.py $MODEL_SIZE $layer $SEQ_LEN \
        --input_file ffn_input.bin \
        --output_file ffn_output.bin

    echo "[${layer}] Skip connection (post-FFN)"
    python llama-skip-connection.py \
        --block_input_file post_attn_norm_input.bin \
        --block_output_file ffn_output.bin \
        --output_file layer_output.bin

    echo "[${layer}] Done."
    echo ""

    # Output of this layer is input to the next
    cp layer_output.bin "$CURRENT_INPUT"
done

echo "=== All ${NUM_LAYERS} layers proved successfully ==="
