#!/usr/bin/env bash
# Run 5 zkLLM configs sequentially on GPU 0.
# Run from ~/opal/src/: nohup bash ../scripts/run_zkllm_batch.sh > ../zkllm_batch.log 2>&1 &

set -euo pipefail

CONFIGS=(
    accuracy_pc+rl
    sobol_ia
    sobol_pc+ia
    shmembench_rl
    shmembench_pc+rl+ia
)

for config in "${CONFIGS[@]}"; do
    echo "=== Starting $config at $(date) ==="
    python opal_cli_codex.py \
        --config "config/${config}_zkllm.yaml" \
        --zkllm_dir ../zkllm \
        --zkllm_proof_dir ../zkllm_proofs
    echo "=== Finished $config at $(date) ==="
done

echo "=== All configs complete at $(date) ==="
