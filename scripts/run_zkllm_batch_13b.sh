#!/usr/bin/env bash
# Run 6 zkLLM 13B configs sequentially on GPU 0.
# Run from ~/opal/src/: nohup bash ../scripts/run_zkllm_batch_13b.sh > ../zkllm_batch_13b.log 2>&1 &

set -euo pipefail

CONFIGS=(
    accuracy_ia_zkllm_13b
    accuracy_pc+rl_zkllm_13b
    sobol_ia_zkllm_13b
    sobol_pc+ia_zkllm_13b
    shmembench_rl_zkllm_13b
    shmembench_pc+rl+ia_zkllm_13b
)

for config in "${CONFIGS[@]}"; do
    echo "=== Starting $config at $(date) ==="
    python opal_cli_codex.py \
        --config "config/${config}.yaml" \
        --zkllm_dir ../zkllm \
        --zkllm_proof_dir ../zkllm_proofs_13b
    echo "=== Finished $config at $(date) ==="
done

echo "=== All 13B configs complete at $(date) ==="
