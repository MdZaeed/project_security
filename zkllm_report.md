# zkLLM Integration into OPAL: Complete Technical Report

## 1. What is zkLLM?

zkLLM (Zero-Knowledge Proofs for Large Language Models, CCS 2024) is a cryptographic system that
proves a specific LLM output was produced by a specific set of committed model weights — without
revealing the weights themselves. This is useful in any setting where a model provider wants to
guarantee that inference was performed honestly (e.g., the same model that was audited is the one
being used in production).

### Core cryptographic components

| Component | What it does |
|-----------|-------------|
| **Hyrax/Pedersen commitments** | Commit to model weight tensors using BLS12-381 elliptic curve. The commitment is a short digest that binds the prover to specific weights. |
| **tlookup** | A parallelized ZKP protocol for non-arithmetic tensor operations — primarily the non-linear activation functions (SiLU/SwiGLU in LLaMA-2 FFN). |
| **zkAttn** | A ZKP protocol tailored for the Softmax attention mechanism. Proves the attention computation including the Softmax normalization. |
| **ppgen** | Public parameter generation — the trusted setup phase that creates the structured reference string needed for the proofs. |
| **commit** | Commits the model weights to the public parameters. Must be done once per model before any proof can be generated. |

### Per-layer proof structure

For each transformer layer, zkLLM proves six operations in sequence:

```
layer_input.bin
    → [1] RMSNorm (input)       → attn_input.bin
    → [2] Self-Attention         → attn_output.bin
    → [3] Skip connection        → post_attn_norm_input.bin
    → [4] RMSNorm (post-attn)   → ffn_input.bin
    → [5] FFN (SwiGLU)          → ffn_output.bin
    → [6] Skip connection        → layer_output.bin (→ next layer's input)
```

LLaMA-2 7B has 32 such layers, giving 192 individual proof steps total.

---

## 2. Hardware and Software Environment

**Server:** palomino (accessed via zeus.cs.txstate.edu as login node)
- 5x NVIDIA A100 80GB GPUs
- CUDA 12.8 (system module)
- 7TB NVMe storage (shared filesystem)

**Software stack:**
- Conda environment: `zkllm-env` (Python 3.11)
- PyTorch: cu118 build (required — system CUDA 12.8 driver only supports up to CUDA 11.8 runtime)
- Transformers: 4.36.2 (pinned — newer versions broke the rotary embedding API)
- CUDA compute capability: sm_80 (A100; upstream code targets sm_86 for RTX A6000)

---

## 3. Repository Structure

```
opal/
├── zkllm/                          # Git submodule (upstream: github.com/jvhs0706/zkLLM-ccs2024)
│   ├── model-storage/              # Downloaded LLaMA-2 weights (gitignored)
│   ├── zkllm-workdir/              # Intermediate proof binaries (gitignored)
│   ├── layer_input.bin             # Input embeddings for proof pipeline (gitignored)
│   ├── llama-ppgen.py              # Public parameter generation
│   ├── llama-commit.py             # Weight commitment
│   ├── llama-rmsnorm.py            # Per-layer RMSNorm proof
│   ├── llama-self-attn.py          # Per-layer self-attention proof (PATCHED)
│   ├── llama-ffn.py                # Per-layer FFN proof
│   ├── llama-skip-connection.py    # Per-layer skip connection proof
│   ├── Makefile                    # Builds CUDA binaries (PATCHED for sm_80)
│   └── *.cu                        # CUDA proof implementations
├── zkllm-patches/
│   └── llama-self-attn.py          # Our patched version (kept outside submodule)
├── zkllm_proofs/                   # Output proof JSON records
│   └── proof_<timestamp>.json
├── scripts/
│   ├── setup_zkllm.sh              # Environment setup
│   ├── run_zkllm_proof.sh          # Full 32-layer proof pipeline
│   ├── run_zkllm_batch.sh          # Sequential batch for all 6 configs
│   └── time_zkllm_queries.sh       # Timing + CSV collection
└── src/
    ├── zkllm_conn.py               # NEW: OPAL LLM backend for zkLLM
    ├── opal_cli_codex.py           # MODIFIED: added --config, --llm_backend zkllm
    └── config/
        ├── accuracy_ia_zkllm.yaml
        ├── accuracy_pc+rl_zkllm.yaml
        ├── sobol_ia_zkllm.yaml
        ├── sobol_pc+ia_zkllm.yaml
        ├── shmembench_rl_zkllm.yaml
        └── shmembench_pc+rl+ia_zkllm.yaml
```

---

## 4. Setup: Step by Step

### Step 1 — Add the submodule

```bash
git submodule add git@github.com:jvhs0706/zkLLM-ccs2024.git zkllm
git submodule update --init
```

The submodule is read-only (no push access to upstream), so any patches are stored in
`zkllm-patches/` and copied in during setup.

### Step 2 — Run the setup script

```bash
bash scripts/setup_zkllm.sh
```

What this does internally:

```bash
# Creates conda environment with Python 3.11
conda create -n zkllm-env python=3.11 -y

# Installs PyTorch for CUDA 11.8 (works with CUDA 12.x driver)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Pins transformers to 4.36.2 (rotary embedding API compatibility)
pip install "transformers==4.36.2" datasets huggingface_hub

# Patches the Makefile: sm_86 → sm_80 (RTX A6000 → A100)
sed -i 's/ARCH := sm_86/ARCH := sm_80/' zkllm/Makefile

# Uses system nvcc instead of conda's
sed -i 's|NVCC := $(CONDA_PREFIX)/bin/nvcc|NVCC := nvcc|' zkllm/Makefile

# Copies patched self-attention script
cp zkllm-patches/llama-self-attn.py zkllm/llama-self-attn.py

# Builds all CUDA binaries
cd zkllm && make clean && make all -j$(nproc)
```

**Why pin transformers to 4.36.2?**
In transformers 4.37+, `rotary_emb.forward()` changed its signature — it now requires a tensor
input instead of an integer for `seq_len`. The zkLLM scripts call it with an integer. Pinning to
4.36.2 keeps the old API.

**Why sm_80 instead of sm_86?**
The upstream Makefile targets `sm_86` (NVIDIA RTX A6000). Our server has A100 GPUs which use
compute capability `sm_80`. Using the wrong target causes CUDA to fall back to slower PTX
compilation or fails entirely.

### Step 3 — Download the model

```bash
conda activate zkllm-env
python3 -c "
import os
os.environ['HF_HOME'] = '/path/to/opal/zkllm/model-storage'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='meta-llama/Llama-2-7b-hf', token='YOUR_HF_TOKEN')
"
```

**Flags explained:**
- `HF_HOME`: Sets the root directory for all HuggingFace downloads. The model lands at
  `$HF_HOME/hub/models--meta-llama--Llama-2-7b-hf/snapshots/<hash>/`
- `local_files_only=True`: Used by the zkLLM scripts after download — tells transformers not to
  contact HuggingFace and use only what is cached on disk.
- `token`: Required because LLaMA-2 is a gated model. You must request access at
  huggingface.co/meta-llama and be approved before the token grants access.

**Disk note:** LLaMA-2 7B weights are ~14GB (two safetensors shards). The server's 7TB filesystem
was nearly full (6.6TB used), requiring cleanup of stale model downloads before this could proceed.

### Step 4 — Generate public parameters and commit weights

```bash
cd ~/opal/zkllm
conda activate zkllm-env

python llama-ppgen.py 7
# Argument: model size in billions (7 or 13)
# Takes ~10-20 minutes. Generates the structured reference string for BLS12-381
# commitments. Output goes into zkllm-workdir/.

python llama-commit.py 7 16
# Arguments: model size, number of attention heads (LLaMA-2 7B has 32 heads,
# but 16 refers to a chunking parameter for the commitment computation).
# Commits all 32 layers of weights to the public parameters.
# Output: committed weight files in zkllm-workdir/.
```

These two steps are a **one-time setup per model**. They do not need to be repeated for each
inference — only if you switch models or delete the workdir.

---

## 5. Patches Applied to Upstream Code

Two bugs were found in the upstream zkLLM code that prevent it from running on our setup.

### Patch 1: rotary embedding API (`llama-self-attn.py`, line 49)

**Original (broken on transformers 4.36.x):**
```python
cos, sin = layer.self_attn.rotary_emb(
    torch.arange(args.seq_len, device=0).unsqueeze(0)
)
```

**Fixed:**
```python
cos, sin = layer.self_attn.rotary_emb(
    torch.randn(1, args.seq_len, embed_dim, device=0), args.seq_len
)
```

The transformers 4.36.x API for `rotary_emb.forward()` requires both a dummy tensor (to infer
dtype/device) and an integer `seq_len`. Passing a 1D integer tensor caused a "Boolean value of
Tensor with more than one element is ambiguous" error.

### Patch 2: self-attention output file (`llama-self-attn.py`)

**Added after the `./self-attn attn` call:**
```python
if os.path.isfile('temp_head_out.bin'):
    os.rename('temp_head_out.bin', args.output_file)
```

The CUDA binary `self-attn` (in `attn` mode) hardcodes its output to `temp_head_out.bin` and
ignores the `--output_file` argument. The cleanup step that follows then deleted this file before
the next stage could read it. The rename moves the output to the expected filename before cleanup.

---

## 6. The Full Proof Pipeline (`scripts/run_zkllm_proof.sh`)

```bash
bash scripts/run_zkllm_proof.sh [model_size] [num_layers] [seq_len]
# Defaults: 7, 32, 2048
```

This script runs all 32 transformer layers sequentially. For each layer `i` (0–31):

```bash
# RMSNorm on the layer input
python llama-rmsnorm.py 7 $i input 2048 \
    --input_file layer_input.bin \
    --output_file attn_input.bin

# Self-attention with rotary embeddings and zkAttn proof
python llama-self-attn.py 7 $i 2048 \
    --input_file attn_input.bin \
    --output_file attn_output.bin

# Residual (skip) connection: layer_input + attn_output
python llama-skip-connection.py \
    --block_input_file layer_input.bin \
    --block_output_file attn_output.bin \
    --output_file post_attn_norm_input.bin

# RMSNorm on post-attention result
python llama-rmsnorm.py 7 $i post_attention 2048 \
    --input_file post_attn_norm_input.bin \
    --output_file ffn_input.bin

# Feed-forward network (SwiGLU) with tlookup proof
python llama-ffn.py 7 $i 2048 \
    --input_file ffn_input.bin \
    --output_file ffn_output.bin

# Residual connection: post_attn + ffn_output
python llama-skip-connection.py \
    --block_input_file post_attn_norm_input.bin \
    --block_output_file ffn_output.bin \
    --output_file layer_output.bin

# Feed layer_output into next layer
cp layer_output.bin layer_input.bin
```

**Key flag meanings:**
- `model_size` (7 or 13): selects which weight shards to load from the workdir
- `layer` index: selects the specific transformer layer's committed weights
- `input_file` / `output_file`: intermediate activation tensors passed between proof stages as
  scaled int32 binary files (multiplied by 2^16 = 65536 for fixed-point representation)

When successful, the script prints:
```
=== All 32 layers proved successfully ===
```

---

## 7. OPAL Integration

### New file: `src/zkllm_conn.py`

This class extends OPAL's `Query_llm` abstract base class and implements the zkLLM backend.

```
ZkLLM_conn (extends Query_llm)
├── __init__(args)
│     Loads LLaMA-2 tokenizer and model from local cache.
│     Sets pad_token = eos_token (LLaMA-2 has no padding token by default).
│     Maps model_size → num_layers: {7: 32, 13: 40}
│
├── submit_query_to_llm(prompt) → str
│     1. Tokenize prompt, measure prompt_tokens
│     2. Generate response with greedy decoding (do_sample=False)
│     3. Measure inference_time_s, tokens_per_sec
│     4. Call _prove(prompt, response)
│     5. Save all stats to proof_<timestamp>.json
│     6. Return response text
│
├── _generate(prompt) → str
│     Runs model.generate() with max_new_tokens=512, greedy decoding.
│     Truncates input to seq_len=2048 tokens.
│
└── _prove(prompt, response) → dict
      1. Tokenizes (prompt + response) with padding to seq_len
      2. Extracts token embeddings via model.model.embed_tokens()
      3. Saves embeddings as scaled int32 binary: layer_input.bin
      4. Runs run_zkllm_proof.sh as subprocess
      5. Checks stdout for "All N layers proved successfully"
      6. Returns timing, success flag, layer count
```

**Why does zkLLM load its own model instead of reusing vLLM?**
zkLLM needs access to intermediate activations (embeddings, per-layer outputs) to generate the
proof. vLLM is a black-box inference server that only returns the final text output. zkLLM must
control the forward pass directly, so it loads the model via HuggingFace transformers and runs its
own generate loop.

### Modified file: `src/opal_cli_codex.py`

Three additions were made:

**1. YAML config loading (`--config` flag)**
```python
# Scans sys.argv for --config before argparse runs,
# then injects YAML values as CLI arguments so required
# args (like --app_name) are satisfied without typing them.
for _i, _a in enumerate(_argv):
    if _a == "--config" and _i + 1 < len(_argv):
        config_path = _argv[_i + 1]
        break
if config_path:
    _cfg = yaml.safe_load(open(config_path))
    for _k, _v in _cfg.items():
        if _k not in _existing_flags:
            sys.argv.extend([f"--{_k}", str(_v)])
```

**2. New backend registration**
```python
parser.add_argument(
    "--llm_backend",
    choices=["auto", "openai", "gemini", "ollama", "vllm", "zkllm"]
)
```

**3. Factory function**
```python
# In _select_llm_client():
if backend == "zkllm":
    return ZkLLM_conn(args)
```

**4. New CLI arguments for zkLLM**
| Flag | Default | Meaning |
|------|---------|---------|
| `--zkllm_model_size` | 7 | Model size in billions (7 or 13) |
| `--zkllm_seq_len` | 2048 | Maximum sequence length |
| `--zkllm_max_new_tokens` | 512 | Maximum tokens to generate |
| `--zkllm_dir` | `../zkllm` | Path to the zkLLM submodule directory |
| `--zkllm_proof_dir` | `../zkllm_proofs` | Where to save proof JSON records |

### Config files (`src/config/*_zkllm.yaml`)

Six YAML configs were created, one per experiment combination:

| Config | Benchmark | Info Source |
|--------|-----------|-------------|
| `accuracy_ia_zkllm.yaml` | accuracy | instruction-aware (ia) |
| `accuracy_pc+rl_zkllm.yaml` | accuracy | PC sampling + roofline |
| `sobol_ia_zkllm.yaml` | sobol | instruction-aware |
| `sobol_pc+ia_zkllm.yaml` | sobol | PC sampling + instruction-aware |
| `shmembench_rl_zkllm.yaml` | shmembench | roofline |
| `shmembench_pc+rl+ia_zkllm.yaml` | shmembench | PC + roofline + instruction-aware |

Each config sets `llm_backend: "zkllm"` and reuses the existing GPT-5 profiling summaries
(`*_gpt5_summaries.json`) as the profiling context fed into the prompt.

---

## 8. Running a Single Query

```bash
conda activate zkllm-env
cd ~/opal/src

python opal_cli_codex.py \
    --config config/accuracy_ia_zkllm.yaml \
    --zkllm_dir ../zkllm \
    --zkllm_proof_dir ../zkllm_proofs
```

**What happens:**
1. YAML is loaded; `app_name`, `source_file`, `llm_backend`, and zkllm settings are injected
2. The CUDA source file (`input/accuracy-new/main.cu`) is read
3. The profiling summaries (instruction analysis) are loaded
4. A prompt is constructed combining the source code and profiling context
5. `ZkLLM_conn` loads LLaMA-2 7B into GPU memory (~28GB in float32)
6. LLaMA-2 generates an optimized CUDA kernel (greedy, up to 512 new tokens)
7. The proof pipeline runs for all 32 layers (~35 minutes)
8. Stats are saved to `../zkllm_proofs/proof_<timestamp>.json`

**Running all 6 configs in background:**
```bash
nohup bash ../scripts/run_zkllm_batch.sh > ../zkllm_batch.log 2>&1 &
echo $!        # prints the background process ID
tail -f ../zkllm_batch.log   # monitor progress
```

---

## 9. Output: Proof JSON Records

Each run produces a JSON file in `zkllm_proofs/` with the following fields:

```json
{
  "model": "Llama-2-7b",
  "seq_len": 2048,
  "timestamp": 1777347518,

  "prompt_tokens": 7389,
  "response_tokens": 512,
  "total_tokens": 7901,
  "prompt_chars": 19234,

  "inference_time_s": 19.85,
  "tokens_per_sec": 25.78,

  "response_chars": 956,
  "response_words": 77,
  "has_code_block": false,
  "code_block_lines": 0,
  "code_block_chars": 0,

  "proof_time_s": 2131.67,
  "proof_success": true,
  "layers_proved": 33,
  "proof_file_kb": 0,
  "total_time_s": 2151.52,

  "stdout_tail": "...[31] Done.\n=== All 32 layers proved successfully ===\n",
  "stderr_tail": "..."
}
```

**Field explanations:**
| Field | Meaning |
|-------|---------|
| `prompt_tokens` | Tokens in the OPAL-constructed prompt (code + profiling context) |
| `response_tokens` | Tokens LLaMA-2 generated (capped at `max_new_tokens=512`) |
| `tokens_per_sec` | LLaMA-2 inference throughput |
| `inference_time_s` | Time for model.generate() only |
| `proof_time_s` | Time for all 32 layers of ZK proof generation |
| `total_time_s` | inference + proof |
| `proof_success` | True if all layers proved and subprocess returned 0 |
| `layers_proved` | Count of "Done." lines in stdout (note: 33 = 32 layers + 1 summary line) |
| `has_code_block` | Whether LLaMA-2 wrapped its output in a markdown code fence |

---

## 10. Results

Six valid proof runs were completed (sequential, on GPU 0). Four additional runs from a failed
parallel experiment were discarded (proof_success=False, proof_time ~10-70s indicating corrupted
shared state).

### Timing summary (valid runs)

| Config | Total Tokens | Inference | Proof Time | Total | Success |
|--------|-------------|-----------|------------|-------|---------|
| accuracy / ia | 4,914 | 19.9s | 2,100s | 35.3 min | True |
| accuracy / pc+rl | 11,165 | 19.85s | 2,131s | 35.8 min | True |
| sobol / ia | 3,865 | 19.86s | 2,133s | 35.9 min | True |
| sobol / pc+ia | 4,273 | 19.83s | 2,133s | 35.9 min | True |
| shmembench / rl | 5,578 | 19.78s | 2,136s | 35.9 min | True |
| shmembench / pc+rl+ia | 7,901 | 19.79s | 2,132s | 35.9 min | True |

### Key observations

**1. Proof time dominates (~99% of total time)**
Inference takes ~20 seconds regardless of prompt size. The ZK proof takes ~35 minutes for 32
layers. The cryptographic proof generation is the overwhelming bottleneck.

**2. Inference time is constant across prompt sizes**
Prompts range from 3,865 to 11,165 tokens yet inference time stays at ~19.8s. This is because
LLaMA-2 always generates exactly 512 new tokens (the `max_new_tokens` limit is hit), and the
generation time is dominated by the decoding loop length, not the prefill length.

**3. Proof time is constant across prompt and response sizes**
The proof covers the full 2048-token sequence (padded to `seq_len`), not just the generated
tokens. So proof time is always the cost of proving 32 layers × 2048 tokens, regardless of
actual prompt or response length.

**4. LLaMA-2 7B did not generate code blocks**
In all 6 runs, `has_code_block=False`. LLaMA-2 7B responded in plain text rather than wrapping
code in markdown fences. This is consistent with its limited instruction-following capability
compared to larger or instruction-tuned models.

**5. 100% proof success rate**
All 6 sequential runs completed with `proof_success=True`. The zkLLM proof system is deterministic
and reliable when given exclusive access to the GPU and the shared workdir.

---

## 11. Lessons Learned

**Parallel runs share the workdir and corrupt each other.**
The `zkllm-workdir/` stores intermediate binary tensors (layer activations) that are passed
between proof stages by filename. Running two configs simultaneously on different GPUs overwrites
these files, producing corrupted proofs that fail instantly. All runs must be sequential.

**The TRANSFORMERS_CACHE env variable is deprecated.**
Older HuggingFace code sets `os.environ['TRANSFORMERS_CACHE']`. Newer `huggingface_hub` versions
use `HF_HOME` instead. The original `download-models.py` script silently failed because it set
the old variable and downloads went to the default `~/.cache/huggingface/` which was also
unreachable on this server.

**PyTorch CUDA version must match the driver, not the CUDA toolkit.**
The server had CUDA 12.8 installed but the GPU driver only exposes CUDA runtime 11.x to
applications. Installing PyTorch with `--index-url .../cu118` resolved runtime mismatch errors.

**Submodule patches must stay in the parent repo.**
Since the zkLLM upstream is read-only (no push access), any changes to the submodule create local
commits that cannot be pushed, causing `git push` to fail with "not our ref". Patches are stored
in `zkllm-patches/` and applied by the setup script via `cp` and `sed`.
