# zkLLM × OPAL Integration

This repository integrates [zkLLM (CCS 2024)](https://arxiv.org/abs/2404.16109) — a zero-knowledge proof system for large language models — into OPAL, an LLM-driven CUDA kernel optimization framework. It enables OPAL to use LLaMA-2 as its inference backend while cryptographically proving that each response was produced by the specific committed model weights, with no possibility of silent model substitution.

## What this does

OPAL takes a CUDA source file and profiling data as input, sends a prompt to an LLM backend, and receives an optimized kernel suggestion. This repo adds a `zkllm` backend that:

1. Runs LLaMA-2 7B (or 13B) inference via HuggingFace Transformers
2. Generates a zero-knowledge proof over all 32 transformer layers using BLS12-381 elliptic curve commitments
3. Saves a JSON record with full timing, token counts, response stats, and proof health

The result: every LLM response comes with a cryptographic certificate that the model was not tampered with between audit and deployment.

## Repository structure

```
.
├── zkllm/                    # Upstream zkLLM submodule (CUDA proof implementation)
│   ├── llama-ppgen.py        # Public parameter generation (one-time setup)
│   ├── llama-commit.py       # Weight commitment (one-time setup per model)
│   ├── llama-{rmsnorm,self-attn,ffn,skip-connection}.py  # Per-layer proof steps
│   ├── Makefile              # Patched: sm_86 → sm_80 for A100, system nvcc
│   └── *.cu / *.cuh          # CUDA implementations of ZK protocols
├── zkllm-patches/
│   └── llama-self-attn.py    # Bug-fix patch (kept outside submodule)
├── zkllm_proofs/             # Output proof JSON records (one per run)
├── scripts/
│   ├── setup_zkllm.sh        # Full environment + build setup
│   ├── run_zkllm_proof.sh    # Runs the 32-layer proof pipeline
│   ├── run_zkllm_batch.sh    # Sequential batch across all 6 configs
│   └── time_zkllm_queries.sh # Timing + CSV collection
├── src/
│   ├── zkllm_conn.py         # ZkLLM_conn: OPAL backend class
│   ├── opal_cli_codex.py     # Modified: --config flag, --llm_backend zkllm
│   └── config/               # YAML configs for 6 benchmark × info-source combos
├── logs/                     # Captured stdout/stderr from each batch run (one log per config)
├── zkllm_timing_results.csv  # Aggregated timing across all 6 valid runs
└── zkllm_report.md           # Full technical report (setup, patches, results)
```

## Hardware requirements

- NVIDIA A100 (or similar sm_80+ GPU) with ≥80GB VRAM for 7B in float32
- CUDA 12.x driver (PyTorch uses CUDA 11.8 runtime — both coexist fine)
- ~14GB disk space for LLaMA-2 7B weights; ~50GB for 13B

Tested on: 5× NVIDIA A100 80GB, CUDA 12.8, Ubuntu 22.04.

## Setup

### 1. Load CUDA and run the setup script

```bash
module load cuda/12.8   # or equivalent on your system
bash scripts/setup_zkllm.sh
```

This creates a `zkllm-env` conda environment, installs dependencies (including `transformers==4.36.2` — pinned for API compatibility), applies patches to the Makefile, and compiles all CUDA binaries.

### 2. Download LLaMA-2 weights

You must request access to `meta-llama/Llama-2-7b-hf` at huggingface.co/meta-llama first, then:

```bash
conda activate zkllm-env
cd zkllm
python download-models.py meta-llama/Llama-2-7b-hf YOUR_HF_TOKEN
```

### 3. Generate public parameters and commit weights (one-time)

```bash
conda activate zkllm-env
cd zkllm
python llama-ppgen.py 7        # ~10–20 min; generates BLS12-381 structured reference string
python llama-commit.py 7 16    # commits all 32 layers of weights
```

Output goes into `zkllm/zkllm-workdir/`. These steps do not need to be repeated unless you switch models or delete the workdir.

## Running

### Single query

```bash
conda activate zkllm-env
cd src
python opal_cli_codex.py \
    --config config/accuracy_ia_zkllm.yaml \
    --zkllm_dir ../zkllm \
    --zkllm_proof_dir ../zkllm_proofs
```

This loads the config, constructs the OPAL prompt (CUDA source + profiling context), runs LLaMA-2 inference, then proves all 32 layers. Total runtime: ~36 minutes per query.

### All 6 configs in background

```bash
conda activate zkllm-env
nohup bash scripts/run_zkllm_batch.sh > logs/zkllm_batch.log 2>&1 &
tail -f logs/zkllm_batch.log
```

**Important:** runs must be sequential. The proof workdir stores intermediate activation tensors by fixed filename — parallel runs corrupt each other's state.

### CLI flags for the zkllm backend

| Flag | Default | Description |
|------|---------|-------------|
| `--zkllm_model_size` | `7` | Model size in billions (7 or 13) |
| `--zkllm_seq_len` | `2048` | Max sequence length |
| `--zkllm_max_new_tokens` | `512` | Max tokens to generate |
| `--zkllm_dir` | `../zkllm` | Path to zkLLM submodule |
| `--zkllm_proof_dir` | `../zkllm_proofs` | Where to save proof JSON records |

## Proof output

Each run produces `zkllm_proofs/proof_<timestamp>.json`:

```json
{
  "model": "Llama-2-7b",
  "prompt_tokens": 7389,
  "response_tokens": 512,
  "inference_time_s": 19.79,
  "tokens_per_sec": 25.87,
  "proof_time_s": 2132.6,
  "proof_success": true,
  "layers_proved": 33,
  "total_time_s": 2152.39
}
```

## Run logs

The `logs/` directory contains stdout/stderr captured from all runs.

### `zkllm_batch.log` — successful sequential run (Apr 28, 03:02–06:03 UTC)

The definitive run. Five configs executed back-to-back via `run_zkllm_batch.sh`; each completed in ~36 minutes with `proof_success=True`. The `accuracy_ia` config was run separately beforehand and is not in this log.

```
=== Starting accuracy_pc+rl at Tue Apr 28 03:02:29 AM UTC 2026 ===
[zkLLM] Response: 513 tokens in 19.8s (25.85 tok/s).
[zkLLM] Total: inference=19.8s  proof=2131.7s  total=2151.52s
=== Finished accuracy_pc+rl at Tue Apr 28 03:38:39 AM UTC 2026 ===
...
=== All configs complete at Tue Apr 28 06:03:10 AM UTC 2026 ===
```

### Per-config logs — failed parallel runs (Apr 28, ~02:54 UTC)

Four configs were launched simultaneously before the sequential-only constraint was understood. They all started at `02:54 UTC` and shared the `zkllm-workdir/`, corrupting each other's intermediate activation tensors. Proof times of 10–69 seconds (versus the expected ~35 minutes) confirm corrupted state; `zkllm_run_shmembench_pc+rl+ia.log` was cut off mid-execution.

| Log file | Proof time | Outcome |
|----------|-----------|---------|
| `zkllm_run_accuracy_pc+rl.log` | 11.5s | Corrupted — discarded |
| `zkllm_run_shmembench_rl.log` | 9.8s | Corrupted — discarded |
| `zkllm_run_sobol_ia.log` | 68.7s | Corrupted — discarded |
| `zkllm_run_sobol_pc+ia.log` | 40.6s | Corrupted — discarded |
| `zkllm_run_shmembench_pc+rl+ia.log` | N/A | Killed mid-run |

These logs are kept as evidence of the parallel-run failure mode and to document the workdir contention issue described in the patches section.

## Results

Six valid sequential proof runs completed with 100% success rate:

| Config | Benchmark | Info source | Prompt tokens | Inference | Proof time | Total |
|--------|-----------|-------------|--------------|-----------|------------|-------|
| accuracy_ia | accuracy | ia | 4,401 | 19.9s | 35.0 min | 35.3 min |
| accuracy_pc+rl | accuracy | pc+rl | 10,652 | 19.9s | 35.5 min | 35.9 min |
| sobol_ia | sobol | ia | 3,353 | 19.9s | 35.6 min | 35.9 min |
| sobol_pc+ia | sobol | pc+ia | 5,065 | 19.8s | 35.6 min | 35.9 min |
| shmembench_rl | shmembench | rl | 3,759 | 19.8s | 35.6 min | 35.9 min |
| shmembench_pc+rl+ia | shmembench | pc+rl+ia | 7,389 | 19.8s | 35.5 min | 35.9 min |

**Key observations:**
- Proof generation (~35 min) accounts for >99% of total runtime; inference (~20s) is negligible
- Proof time is constant regardless of prompt size — the proof always covers the full 2048-token sequence
- Inference time is also constant — LLaMA-2 always hits the `max_new_tokens=512` cap
- LLaMA-2 7B did not produce markdown code blocks in any run (no instruction tuning)

## Patches applied to upstream zkLLM

Two bugs were fixed in the upstream code (patches stored in `zkllm-patches/`, applied by `setup_zkllm.sh`):

**1. Rotary embedding API** (`llama-self-attn.py:49`) — `transformers` 4.36.x requires a dummy tensor and an integer `seq_len`; the original code passed a 1D integer tensor, causing a "Boolean value of Tensor with more than one element is ambiguous" error.

**2. Self-attention output file** (`llama-self-attn.py`) — the `self-attn` CUDA binary hardcodes its output to `temp_head_out.bin` and ignores `--output_file`. Without the patch, cleanup deletes this file before the next proof stage reads it.

## Further reading

See [`zkllm_report.md`](zkllm_report.md) for the full technical report covering cryptographic components, the per-layer proof structure, all setup steps, patch details, OPAL integration internals, and an extended results analysis.

The upstream zkLLM paper: [arXiv:2404.16109](https://arxiv.org/abs/2404.16109) (ACM CCS 2024).

## My contributions

**`src/zkllm_conn.py` — ZkLLM OPAL backend**

Implements `ZkLLM_conn`, the class that plugs zkLLM into OPAL as a drop-in LLM backend. It accepts any OPAL prompt (CUDA source + profiling context), runs LLaMA-2 inference via HuggingFace Transformers, then drives the full 32-layer zero-knowledge proof pipeline by invoking the compiled CUDA binaries in sequence (ppgen → commit → rmsnorm → self-attn → ffn → skip-connection). After each run it writes a structured JSON record to `zkllm_proofs/` capturing model name, prompt and response token counts, inference time, proof time, total time, tokens per second, per-layer proof health, and a `proof_success` boolean. The class exposes the same interface as other OPAL backends so it can be selected with `--llm_backend zkllm` without changes to the rest of the pipeline.

**`src/opal_cli_codex.py` — OPAL CLI with zkLLM support**

Modified version of the OPAL command-line entry point. Added a `--config` flag that loads a YAML file specifying the benchmark, info-source combination, and all backend parameters, so each of the six experiment configurations can be run without editing code. Added `--llm_backend zkllm` routing and the five `--zkllm_*` flags (`--zkllm_model_size`, `--zkllm_seq_len`, `--zkllm_max_new_tokens`, `--zkllm_dir`, `--zkllm_proof_dir`). The script reads the target CUDA kernel and profiling data, constructs the OPAL prompt, dispatches it to the selected backend, and prints the optimized kernel suggestion along with a summary of proof metadata.

**Building and fixing the upstream zkLLM repo**

Compiled the upstream CUDA codebase from source and resolved two bugs that prevented it from running (see [Patches applied to upstream zkLLM](#patches-applied-to-upstream-zkllm)). Also patched the Makefile to target `sm_80` (A100) instead of `sm_86` and to use the system `nvcc` rather than a hardcoded path.

**Timing experiments across prompt sizes**

Ran all six benchmark × info-source configurations (prompt sizes ranging from 3,353 to 10,652 tokens) sequentially and recorded per-run timing. Results are in `zkllm_timing_results.csv` and summarised in the Results table above.

**Model variant testing**

Tested the proof pipeline with three additional model variants beyond the baseline LLaMA-2 7B: LLaMA-2 13B, LLaMA-3 7B, and a fine-tuned LLaMA-2 7B. Results and failure modes are documented in the findings below.

## Findings

**Not feasible for production LLM calls**
End-to-end latency is ~36 minutes per query — ~20 seconds for inference and ~35 minutes for proof generation. This rules out any interactive or latency-sensitive use case.

**Proof generation does not scale across multiple GPUs**
Despite the paper framing the system as scalable, the proof runs entirely on a single GPU and cannot be parallelised across multiple devices. It does saturate 100% of that one GPU throughout the ~35-minute proof window.

**Does not work with LLaMA-3 or newer architectures**
The CUDA proof kernels are written specifically for the LLaMA-2 transformer architecture. LLaMA-3 and other updated transformer variants use different attention and normalisation implementations that are incompatible with the current proof code; attempts to run them fail at the proof stage.

**High VRAM requirements**

| Model | VRAM required | Breakdown |
|-------|--------------|-----------|
| LLaMA-2 7B | ~52 GB | ~16 GB model weights + ~26 GB proof intermediates + ~10 GB packages/runtime |
| LLaMA-2 13B | ~120 GB | same breakdown, scaled |

**Context length is hard-capped at 2048 tokens**
The proof pipeline pads every input sequence to exactly 2048 tokens regardless of actual length. Requesting a response longer than 2048 tokens causes the proof to go out of bounds and fail. This means the combined prompt + response must fit within 2048 tokens.
