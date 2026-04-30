import os
import re
import json
import time
import subprocess
import numpy as np
import torch
from pathlib import Path
from query_llm import Query_llm
from transformers import AutoTokenizer, AutoModelForCausalLM


SCALING_FACTOR = 1 << 16


def _save_int_bin(tensor: torch.Tensor, path: str):
    t = torch.round(tensor.float() * SCALING_FACTOR).to(torch.int32)
    t.cpu().detach().numpy().astype(np.int32).tofile(path)


def _count_layers_proved(stdout: str) -> int:
    return len(re.findall(r"layers proved successfully|Done\.", stdout))


def _extract_code_block(text: str) -> str:
    m = re.search(r"```(?:[^\n`]*)\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


class ZkLLM_conn(Query_llm):
    """
    LLM connection that generates a response with LLaMA-2 and produces
    a zkLLM zero-knowledge proof for the inference.

    Captures detailed stats (token counts, timing breakdown, response
    quality, proof health) and saves them as a JSON proof record.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_size = int(getattr(args, "zkllm_model_size", 7))
        self.seq_len = int(getattr(args, "zkllm_seq_len", 2048))
        self.max_new_tokens = int(getattr(args, "zkllm_max_new_tokens", 512))
        self.zkllm_dir = Path(getattr(args, "zkllm_dir", "zkllm")).resolve()
        self.proof_out_dir = Path(getattr(args, "zkllm_proof_dir", "zkllm_proofs")).resolve()
        self.proof_out_dir.mkdir(parents=True, exist_ok=True)

        model_card = f"meta-llama/Llama-2-{self.model_size}b-hf"
        cache_dir = str(self.zkllm_dir / "model-storage")

        print(f"[zkLLM] Loading {model_card} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_card, local_files_only=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_card, local_files_only=True, cache_dir=cache_dir,
            torch_dtype=torch.float32
        ).to("cuda")
        self.model.eval()
        self.num_layers = {7: 32, 13: 40}.get(self.model_size, 32)
        self.last_stats: dict = {}
        print("[zkLLM] Model loaded.")

    def submit_query_to_llm(self, prompt: str) -> str:
        prompt_tokens = len(self.tokenizer.encode(prompt))

        print("[zkLLM] Generating response ...")
        t0 = time.time()
        response = self._generate(prompt)
        inference_time = time.time() - t0

        response_tokens = len(self.tokenizer.encode(response))
        tokens_per_sec = round(response_tokens / inference_time, 2) if inference_time > 0 else 0
        code_block = _extract_code_block(response)

        print(f"[zkLLM] Response: {response_tokens} tokens in {inference_time:.1f}s "
              f"({tokens_per_sec} tok/s).")

        print("[zkLLM] Starting ZK proof pipeline ...")
        proof_meta = self._prove(prompt, response)

        stats = {
            # context
            "model": f"Llama-2-{self.model_size}b",
            "seq_len": self.seq_len,
            "timestamp": int(time.time()),
            # token counts
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "prompt_chars": len(prompt),
            # inference
            "inference_time_s": round(inference_time, 2),
            "tokens_per_sec": tokens_per_sec,
            # response quality
            "response_chars": len(response),
            "response_words": len(response.split()),
            "has_code_block": bool(code_block),
            "code_block_lines": code_block.count("\n") + 1 if code_block else 0,
            "code_block_chars": len(code_block),
            # proof
            "proof_time_s": proof_meta["proof_time_s"],
            "proof_success": proof_meta["success"],
            "layers_proved": proof_meta["layers_proved"],
            "proof_file_kb": proof_meta["proof_file_kb"],
            "total_time_s": round(inference_time + proof_meta["proof_time_s"], 2),
            # tails for debugging
            "stdout_tail": proof_meta["stdout_tail"],
            "stderr_tail": proof_meta["stderr_tail"],
        }

        proof_path = self.proof_out_dir / f"proof_{stats['timestamp']}.json"
        with open(proof_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[zkLLM] Stats saved to {proof_path}")
        print(f"[zkLLM] Total: inference={inference_time:.1f}s  "
              f"proof={proof_meta['proof_time_s']:.1f}s  "
              f"total={stats['total_time_s']}s")

        self.last_stats = stats
        return response

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.seq_len
        ).to("cuda")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _prove(self, prompt: str, response: str) -> dict:
        full_text = prompt + " " + response
        tokens = self.tokenizer(
            full_text, return_tensors="pt", truncation=True,
            max_length=self.seq_len, padding="max_length"
        ).to("cuda")

        with torch.no_grad():
            embeds = self.model.model.embed_tokens(tokens["input_ids"])

        _save_int_bin(embeds[0], str(self.zkllm_dir / "layer_input.bin"))

        script = Path(__file__).parent.parent / "scripts" / "run_zkllm_proof.sh"
        t0 = time.time()
        result = subprocess.run(
            ["bash", str(script), str(self.model_size), str(self.num_layers), str(self.seq_len)],
            cwd=str(self.zkllm_dir),
            capture_output=True, text=True
        )
        proof_time = time.time() - t0

        success = result.returncode == 0 and f"All {self.num_layers} layers proved successfully" in result.stdout
        layers_proved = _count_layers_proved(result.stdout)

        # Proof JSON size as a proxy for proof size
        proof_files = sorted(self.proof_out_dir.glob("proof_*.json"))
        proof_file_kb = round(
            proof_files[-1].stat().st_size / 1024, 1
        ) if proof_files else 0

        return {
            "success": success,
            "proof_time_s": round(proof_time, 2),
            "layers_proved": layers_proved,
            "proof_file_kb": proof_file_kb,
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-200:] if result.stderr else "",
        }
