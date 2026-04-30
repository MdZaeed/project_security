#!/usr/bin/env bash
# Run 6 zkLLM proof queries across accuracy, sobol, shmembench
# and record detailed stats for each.
# Run from ~/opal/src/:   bash ../scripts/time_zkllm_queries.sh

set -euo pipefail

RESULTS_FILE="../zkllm_timing_results.csv"
ZKLLM_DIR="../zkllm"
PROOF_DIR="../zkllm_proofs"

# CSV header
echo "config,benchmark,info_source,\
prompt_tokens,response_tokens,total_tokens,\
inference_time_s,tokens_per_sec,\
proof_time_s,total_time_s,\
proof_success,layers_proved,\
response_words,has_code_block,code_block_lines,\
status" > "$RESULTS_FILE"

run_query() {
    local config="$1"
    local benchmark="$2"
    local info_source="$3"

    echo ""
    echo "======================================================"
    echo "Running: $config  [$benchmark / $info_source]"
    echo "======================================================"

    local wall_start=$SECONDS
    if python opal_cli_codex.py \
        --config "config/${config}" \
        --zkllm_dir "$ZKLLM_DIR" \
        --zkllm_proof_dir "$PROOF_DIR" \
        --no_build 2>&1 | tee /tmp/zkllm_run.log; then
        local status="success"
    else
        local status="failed"
    fi
    local wall_elapsed=$(( SECONDS - wall_start ))

    # Pull stats from the most recent proof JSON
    local latest_proof
    latest_proof=$(ls -t "$PROOF_DIR"/proof_*.json 2>/dev/null | head -1 || echo "")

    if [[ -n "$latest_proof" ]]; then
        prompt_tokens=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('prompt_tokens',0))")
        response_tokens=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('response_tokens',0))")
        total_tokens=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('total_tokens',0))")
        inference_time=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('inference_time_s',0))")
        tokens_per_sec=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('tokens_per_sec',0))")
        proof_time=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('proof_time_s',0))")
        total_time=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('total_time_s',0))")
        proof_success=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('proof_success',False))")
        layers_proved=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('layers_proved',0))")
        response_words=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('response_words',0))")
        has_code=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('has_code_block',False))")
        code_lines=$(python3 -c "import json; d=json.load(open('$latest_proof')); print(d.get('code_block_lines',0))")
    else
        prompt_tokens=0; response_tokens=0; total_tokens=0
        inference_time=0; tokens_per_sec=0; proof_time=0; total_time=$wall_elapsed
        proof_success=False; layers_proved=0; response_words=0; has_code=False; code_lines=0
    fi

    echo "${config},${benchmark},${info_source},\
${prompt_tokens},${response_tokens},${total_tokens},\
${inference_time},${tokens_per_sec},\
${proof_time},${total_time},\
${proof_success},${layers_proved},\
${response_words},${has_code},${code_lines},\
${status}" >> "$RESULTS_FILE"

    echo ">>> Done in ${wall_elapsed}s (wall) | inference=${inference_time}s proof=${proof_time}s"
}

run_query "accuracy_ia_zkllm.yaml"        "accuracy"    "ia"
run_query "accuracy_pc+rl_zkllm.yaml"     "accuracy"    "pc+rl"
run_query "sobol_ia_zkllm.yaml"           "sobol"       "ia"
run_query "sobol_pc+ia_zkllm.yaml"        "sobol"       "pc+ia"
run_query "shmembench_rl_zkllm.yaml"      "shmembench"  "rl"
run_query "shmembench_pc+rl+ia_zkllm.yaml" "shmembench" "pc+rl+ia"

echo ""
echo "======================================================"
echo "All queries complete. Results:"
echo "======================================================"
cat "$RESULTS_FILE"
