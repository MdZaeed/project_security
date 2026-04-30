# import streamlit as st
import argparse
import pandas as pd
from pathlib import Path
import re
import json
import Levenshtein
from collections import defaultdict
import os
import re
import ast
import subprocess
import csv
import unicodedata
import time
from ollama import chat
from ollama import Client
from ollam_conn import Ollama_conn
from query_llm import Query_llm
from chatGPT_conn import ChatGPT_conn
from Gemini_conn import Gemini_conn
from GeminiVertex_conn import GeminiVertex_conn
from vllm_connection import VLLM_conn
from zkllm_conn import ZkLLM_conn
from DashingSummarization import DashingSummarization
from PCsummerization import PCsummerization
from RooflineSummarization import RooflineSummarization
from summarization_runtime import run_configured_summaries
# from transformers import AutoTokenizer

from util import *
from extract_data import *
#from interpret import *
#from viz import *
from code_annotation import * #highlighter functions
#from llm import * #reads the api key
from token_optimizer import *

ANSI_ESCAPE = re.compile(r'\x1B\[[0-9;]*[mK]')


def _gemini_api_key_from_args_or_env(args) -> str:
    return (
        getattr(args, "gemini_api_key", None)
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_KEY")
        or ""
    ).strip()


def _prompt_style(args) -> str:
    return (getattr(args, "prompt_style", "codex-default") or "codex-default").strip().lower()


def _is_finetune_compact_prompt_style(args) -> bool:
    return _prompt_style(args) == "finetune-compact"

def _read_file_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return ANSI_ESCAPE.sub('', f.read())
    except FileNotFoundError:
        return f"[error log not found at: {path}]"


def _strip_line_number_prefixes(source_text: str) -> str:
    lines = []
    for line in (source_text or "").splitlines():
        lines.append(re.sub(r"^\s*\d+\s*:\s?", "", line))
    return "\n".join(lines)


def _resolve_build_dir_and_script(build_dir, build_script):
    script_value = str(build_script or "run.sh").strip()
    if not script_value:
        script_value = "run.sh"

    build_dir_value = str(build_dir or "").strip()
    script_path = Path(script_value).expanduser()

    if not build_dir_value:
        script_parent = script_path.parent
        if str(script_parent) not in ("", "."):
            build_dir_value = str(script_parent)
            script_value = script_path.name

    return build_dir_value, script_value


def _resolve_build_target_path(build_dir: Path, source_file: str, build_target_source: str | None) -> Path:
    target_value = str(build_target_source or "").strip()
    if target_value:
        target_path = Path(target_value).expanduser()
        if not target_path.is_absolute():
            target_path = build_dir / target_path
        return target_path
    return build_dir / Path(source_file).name


def _find_matching_brace(text: str, open_brace_idx: int) -> int:
    depth = 0
    i = open_brace_idx
    n = len(text)
    in_line_comment = False
    in_block_comment = False
    in_string = False
    string_delim = ""

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == string_delim:
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch in ("'", '"'):
            in_string = True
            string_delim = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    raise ValueError("Unbalanced braces while locating CUDA function body.")


def _iter_global_function_ranges(code: str):
    search_pos = 0
    while True:
        start = (code or "").find("__global__", search_pos)
        if start == -1:
            return

        brace_idx = (code or "").find("{", start)
        if brace_idx == -1:
            return

        header = code[start:brace_idx]
        name_match = re.search(r"\bvoid\s+([A-Za-z_]\w*)\s*\(", header)
        if not name_match:
            search_pos = start + len("__global__")
            continue

        name = name_match.group(1)
        close_brace_idx = _find_matching_brace(code, brace_idx)
        yield name, start, brace_idx, close_brace_idx
        search_pos = close_brace_idx + 1


def _extract_function_blocks(code: str) -> dict[str, str]:
    blocks = {}
    for name, start, _brace_idx, close_brace_idx in _iter_global_function_ranges(code or ""):
        blocks[name] = code[start:close_brace_idx + 1]
    return blocks


def _replace_function_block(target_text: str, function_name: str, replacement_block: str) -> tuple[str, bool]:
    for name, start, _brace_idx, close_brace_idx in _iter_global_function_ranges(target_text or ""):
        if name != function_name:
            continue
        updated = target_text[:start] + replacement_block.rstrip() + target_text[close_brace_idx + 1:]
        return updated, True
    return target_text, False


def _merge_updated_code_into_build_target(updated_code: str, build_target_text: str) -> tuple[str, list[str]]:
    cleaned_updated_code = _strip_line_number_prefixes(updated_code).strip()
    function_blocks = _extract_function_blocks(cleaned_updated_code)
    if not function_blocks:
        raise ValueError("No __global__ kernel definitions found in optimized code to merge.")

    merged_text = build_target_text
    replaced = []
    missing = []
    for function_name, function_block in function_blocks.items():
        merged_text, did_replace = _replace_function_block(merged_text, function_name, function_block)
        if did_replace:
            replaced.append(function_name)
        else:
            missing.append(function_name)

    if missing:
        raise ValueError(
            "Could not find matching kernel definition(s) in build target: " + ", ".join(sorted(missing))
        )

    return merged_text, replaced
    
def run_script_in_build_dir_separate_logs(build_dir, script_name="run.sh",
                                          out_file="build.out", err_file="build.err"):
    """
    Run a bash script inside `build_dir` and capture stdout/stderr in separate files.

    Returns:
        int: process return code
    """
    bdir = Path(build_dir)
    bdir.mkdir(parents=True, exist_ok=True)

    script_path = bdir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    out_path = bdir / out_file
    err_path = bdir / err_file

    with open(out_path, "w", encoding="utf-8") as out, \
         open(err_path, "w", encoding="utf-8") as err:
        proc = subprocess.run(
            ["bash", str(script_path)],  # or [str(script_path)] if it's +x with a shebang
            cwd=str(bdir),
            stdout=out,
            stderr=err,
            text=True,
            check=False,
        )

    print(f"[build] script: {script_path}")
    print(f"[build] exit code: {proc.returncode}")
    print(f"[build] stdout -> {out_path}")
    print(f"[build] stderr -> {err_path}")
    return proc.returncode


def _select_llm_client(args):
    backend = (getattr(args, "llm_backend", "auto") or "auto").lower()
    gemini_api_key = _gemini_api_key_from_args_or_env(args)
    if gemini_api_key:
        args.gemini_api_key = gemini_api_key
    if backend == "vllm":
        return VLLM_conn(args)
    if backend == "zkllm":
        return ZkLLM_conn(args)
    if backend == "openai":
        return ChatGPT_conn(args)
    if backend == "gemini":
        if gemini_api_key:
            return Gemini_conn(args)
        return GeminiVertex_conn(args)
    if backend == "ollama":
        return Ollama_conn(args.llm)
    if args.llm in ("gpt-4o", "gpt-5"):
        return ChatGPT_conn(args)
    if args.llm in ("gemini-2.5-pro"):
        if gemini_api_key:
            return Gemini_conn(args)
        return GeminiVertex_conn(args)
    return Ollama_conn(args.llm)


def _extract_candidate_code(response_text: str) -> str:
    text = response_text or ""

    # Prefer the first fenced code block only. Gemini sometimes emits a second empty
    # fence before the optimization lists, which previously let list text leak into
    # the extracted .cu file.
    blocks = []
    for pattern in (
        re.compile(r"```(?:[^\n`]*)\n(.*?)```", re.DOTALL),
        re.compile(r"'''(?:[^\n']*)\n(.*?)'''", re.DOTALL),
    ):
        blocks = [blk.strip() for blk in pattern.findall(text) if blk and blk.strip()]
        if blocks:
            break

    candidate = blocks[0] if blocks else _extract_code_from_response(text).strip()

    # Hard-stop before structured lists in case the model appended them outside the
    # fenced region or the fallback path captured too much text.
    for marker in ("\noptimizations =", "\nsuggested_but_not_applied ="):
        idx = candidate.find(marker)
        if idx != -1:
            candidate = candidate[:idx]

    return candidate.strip()


def _extract_candidate_code_for_style(response_text: str, prompt_style: str = "codex-default") -> str:
    candidate = _extract_candidate_code(response_text)
    if (prompt_style or "").strip().lower() != "finetune-compact":
        return candidate

    text = _strip_line_number_prefixes(candidate).strip()

    if "```" in (response_text or "") or "'''" in (response_text or ""):
        return text

    lines = text.splitlines()

    metadata_prefixes = (
        "// Kernel:",
        "// Target:",
        "// Variant:",
        "// Source:",
        "// Response:",
        "// Optimized Source:",
    )

    cleaned_lines = []
    skipping_commented_block = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in metadata_prefixes):
            skipping_commented_block = True
            continue
        if skipping_commented_block:
            if stripped.startswith("//") or not stripped:
                continue
            skipping_commented_block = False
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()

    if "__global__" not in text and "int main(" not in text and "void main(" not in text:
        return text.strip()

    candidate_regions = []
    for pattern in (r"(?m)^#include\s+", r"(?m)^__global__\s*$", r"(?m)^int\s+main\s*\("):
        for match in re.finditer(pattern, text):
            candidate_regions.append(match.start())

    if candidate_regions:
        candidate_regions = sorted(set(candidate_regions))
        for start in candidate_regions:
            region = text[start:].strip()
            if "int main(" in region or "void main(" in region:
                text = region
                break

    depth = 0
    last_balanced_idx = -1
    in_line_comment = False
    in_block_comment = False
    in_string = False
    string_delim = ""
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == string_delim:
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch in ("'", '"'):
            in_string = True
            string_delim = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    last_balanced_idx = i
        i += 1

    if last_balanced_idx != -1:
        return text[: last_balanced_idx + 1].strip()

    return text.strip()


def _strip_comments_for_noop_check(code: str) -> str:
    text = code or ""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def _normalize_code_for_noop_check(code: str) -> str:
    text = _strip_comments_for_noop_check(code)
    return re.sub(r"\s+", "", text)


def _has_substantive_code_change(previous_code: str, candidate_code: str) -> bool:
    if not candidate_code.strip():
        return False
    return _normalize_code_for_noop_check(previous_code) != _normalize_code_for_noop_check(candidate_code)


def _build_noop_retry_prompt(query: str, prompt_style: str = "codex-default") -> str:
    if (prompt_style or "").strip().lower() == "finetune-compact":
        compact_notice = (
            " Your previous answer was rejected because it did not make a substantive code change. "
            "Formatting-only, whitespace-only, comment-only, or reorder-only rewrites are invalid. "
            "Return optimized code only, and make at least one concrete performance-motivated code change "
            "when a safe opportunity exists."
        )
        instruction_marker = "### Instruction:\n"
        input_marker = "\n\n### Input:\n"
        if instruction_marker in query and input_marker in query:
            start = query.index(instruction_marker) + len(instruction_marker)
            end = query.index(input_marker, start)
            return query[:start] + query[start:end].rstrip() + compact_notice + query[end:]
    return (
        query
        + "\n\n# No-op rejection notice\n"
        + "Your previous answer was rejected because it did not make a substantive code change.\n"
        + "Formatting-only, whitespace-only, comment-only, or reorder-only rewrites are invalid.\n"
        + "You must now either:\n"
        + "1. Return code with at least one concrete performance-motivated change implemented in the code itself, or\n"
        + "2. Return the original code and explicitly state in the optimizations list that no safe substantive optimization could be justified.\n"
        + "Do not claim an optimization unless the code actually changed to implement it.\n"
        + "# End no-op rejection notice\n"
    )


def _write_noop_outcome(
    output_path: str,
    run_name: str,
    iteration: int,
    reason: str,
    response_text: str,
) -> str:
    outcome_path = os.path.join(output_path, f"{run_name}_noop_outcome_iter_{iteration}.txt")
    with open(outcome_path, "w", encoding="utf-8") as file:
        file.write(reason.rstrip() + "\n\n")
        file.write("# Begin Rejected Response\n")
        file.write(response_text or "")
        file.write("\n# End Rejected Response\n")
    return outcome_path


def _infer_benchmark_name(args, content: str = "") -> str:
    candidates = [
        str(getattr(args, "app_name", "") or "").strip(),
        str(getattr(args, "name", "") or "").strip(),
        str(getattr(args, "source_file", "") or "").strip(),
        content,
    ]
    lowered = "\n".join(candidates).lower()
    for benchmark in ("accuracy", "sobol", "shmembench", "sw4lite"):
        if benchmark in lowered:
            return benchmark
    source_file = str(getattr(args, "source_file", "") or "").strip()
    if source_file:
        return Path(source_file).stem
    return "unknown"


def _build_finetune_compact_prompt(
    content,
    pc_data=None,
    important_counters=None,
    filtered_roofline_data=None,
    args=None,
    user_guidance=None,
    extra_summary_sections=None,
):
    def _compact_benchmark_instructions(benchmark_name: str, args_obj) -> str:
        benchmark_name = (benchmark_name or "").strip().lower()
        build_write_mode = str(getattr(args_obj, "build_write_mode", "") or "").strip().lower()
        build_target_source = str(getattr(args_obj, "build_target_source", "") or "").strip()
        if benchmark_name == "sw4lite":
            if build_write_mode == "merge-kernels":
                return (
                    "Return only the optimized __global__ CUDA kernel definition(s) that should replace the existing kernel(s) in the build target.\n"
                    "Do not return a full standalone source file. Do not return includes, macros, helper mains, prose, markdown fences, or explanations.\n"
                    "Preserve the original __global__ kernel name(s), parameter list(s), and externally visible behavior exactly so the kernel(s) can be merged into "
                    f"{build_target_source or 'the existing build target'}.\n"
                    "Performance Guidelines:\n"
                    "1. Reuse values already brought into shared memory or registers before touching global memory again; cache central stencil values and frequently reused neighbors in registers.\n"
                    "2. Hoist invariant index arithmetic, loader flags, coefficient gathers, and multiplicative scales out of the hot loop when they do not change per use.\n"
                    "3. Use the read-only path for truly read-only coefficient/material arrays (for example lambda/mu/strx/stry/strz/rho) while preserving the existing shared-memory tile and stencil structure.\n"
                    "Prefer one concrete local optimization inside the existing kernel structure: hoist invariant index arithmetic, precompute flags, reduce redundant "
                    "global loads, improve shared-memory loading structure, or simplify synchronization.\n"
                    "Do not delete large regions of logic, do not introduce placeholder code, and do not emit helper wrappers unless they are strictly required inside "
                    "the returned kernel definition(s).\n"
                    "Preserve stencil semantics, boundary handling, and float_sw4-based computations exactly unless a local optimization requires a safe equivalent rewrite."
                )
            return (
                "Return one complete compilable CUDA/C++ source file only. Do not return an empty response, prose, or markdown fences.\n"
                "Preserve all existing function names, signatures, macros, typedefs, and externally visible behavior.\n"
                "For this SW4Lite kernel, prefer one concrete local optimization over a broad rewrite: hoist invariant index arithmetic, "
                "precompute boolean flags, reduce redundant global loads, improve shared-memory loading structure, or simplify synchronization.\n"
                "Do not delete large parts of the kernel, do not introduce placeholder code, and do not replace the kernel with a stub.\n"
                "Preserve stencil semantics, boundary handling, and float_sw4-based computations exactly unless a local optimization requires a safe equivalent rewrite.\n"
                "If uncertain, keep most of the source unchanged and apply exactly one clear optimization inside the existing kernel body."
            )
        return (
            "Return one complete compilable CUDA/C++ source file only. Do not return prose, markdown fences, or an empty response.\n"
            "Preserve correctness and external behavior. If uncertain, keep most of the source unchanged and apply one clear local optimization."
        )

    benchmark = _infer_benchmark_name(args, content)
    build_write_mode = str(getattr(args, "build_write_mode", "") or "").strip().lower()
    sw4lite_merge_mode = benchmark == "sw4lite" and build_write_mode == "merge-kernels"

    sections = []

    if pc_data is not None and not sw4lite_merge_mode:
        stall_lines = []
        for i, stall in enumerate(pc_data, 1):
            stall_reasons = ", ".join(stall["Stall Reasons"])
            stall_lines.append(
                f"{i}. Line {stall.get('Source', stall.get('Line No', '?'))} has high stalls due to: {stall_reasons}."
            )
        if stall_lines:
            sections.append("### Stall Analysis\n" + "\n".join(stall_lines))

    if important_counters is not None and len(important_counters) > 0 and not sw4lite_merge_mode:
        counter_lines = []
        for counter in important_counters:
            counter_lines.append(
                f"Kernel: {counter['Kernel']}, Group: {counter['Group Name']}, "
                f"Counter: {counter['Counter Name']}, Impact: {counter['Value']}, "
                f"Description: {counter['Description']}."
            )
        if counter_lines:
            sections.append("### Important Hardware Counters\n" + "\n".join(counter_lines))

    if filtered_roofline_data is not None and len(filtered_roofline_data) > 0:
        roofline_lines = []
        idx = 1
        for kernel_name, comments in filtered_roofline_data.items():
            for comment in comments:
                roofline_lines.append(f"{idx}. Kernel '{kernel_name}': {comment}")
                idx += 1
        if roofline_lines:
            sections.append("### Roofline Analysis\n" + "\n".join(roofline_lines))

    if extra_summary_sections and not sw4lite_merge_mode:
        for section in extra_summary_sections:
            if str(section).strip():
                sections.append(str(section).strip())

    if user_guidance:
        if isinstance(user_guidance, str):
            guidance_text = user_guidance.strip()
        elif isinstance(user_guidance, (list, tuple)):
            guidance_text = "\n".join(str(item).strip() for item in user_guidance if str(item).strip())
        else:
            guidance_text = str(user_guidance).strip()
        if guidance_text:
            sections.append("### User Guidance\n" + guidance_text)

    performance_guidance = "\n\n".join(section for section in sections if section.strip()) or "None"
    variant = str(getattr(args, "name", "") or getattr(args, "app_name", "") or "default").strip() or "default"
    benchmark_specific_instruction = _compact_benchmark_instructions(benchmark, args)

    return (
        "### Instruction:\n"
        "Optimize the following GPU kernel using the provided performance guidance. Preserve correctness and external behavior.\n"
        f"{benchmark_specific_instruction}\n\n"
        "### Input:\n"
        f"### Benchmark\n{benchmark}\n\n"
        f"### Variant\n{variant}\n\n"
        "### Target\nNVIDIA H100 / CUDA\n\n"
        f"### Source Code\n{content.strip()}\n\n"
        f"### Performance Guidance\n{performance_guidance.strip()}\n\n"
        "### Response:\n"
    )


def _is_gemini_model(args) -> bool:
    llm_value = str(getattr(args, "llm", "") or "").strip().lower()
    backend = str(getattr(args, "llm_backend", "") or "").strip().lower()
    return llm_value.startswith("gemini-") or backend == "gemini"


def _gemini_negative_guidance() -> str:
    return (
        "Gemini-specific compatibility guardrails:\n"
        "1. Do NOT use cuda::pipeline, cuda::barrier, cuda::std::pipeline, cuda::std::barrier, "
        "cuda::memcpy_async, cuda::std::memcpy_async, cp.async, cooperative pipeline APIs, or "
        "new CUDA C++ synchronization primitives.\n"
        "2. Do NOT add headers such as <cuda/pipeline> or <cuda/barrier>.\n"
        "3. Restrict changes to conventional CUDA/CUB constructs that compile in a standard NVCC build.\n"
        "4. Prefer optimizations like load hoisting, pointer reuse, loop restructuring, shared-memory caching "
        "with standard __shared__ arrays, reducing synchronization, and reducing atomic pressure.\n"
        "5. Return exactly one fenced code block containing only compilable source code, then the two structured lists.\n"
    )


def _entrypoint_guardrails(source_code: str) -> str:
    main_count = len(re.findall(r"\b(?:int|auto)\s+main\s*\(", source_code or ""))
    if main_count > 0:
        return (
            "Entrypoint guardrails:\n"
            f"1. The source already contains {main_count} main() definition(s). Preserve exactly that many main() definitions.\n"
            "2. Do NOT add a new main(), duplicate main(), rename main(), or move logic into a second entry point.\n"
            "3. Do NOT add wrapper executables, test harnesses, helper mains, or alternate standalone programs.\n"
            "4. Keep the existing top-level program structure intact; only optimize within the existing file structure.\n"
            "5. Do NOT change externally referenced function names or break linkage between existing declarations and definitions.\n"
        )

    return (
        "Entrypoint guardrails:\n"
        "1. The source does not contain a main() definition. Do NOT introduce one.\n"
        "2. Do NOT add wrapper executables, test harnesses, helper mains, or alternate standalone programs.\n"
        "3. Keep the existing top-level program structure intact; only optimize within the existing file structure.\n"
        "4. Do NOT change externally referenced function names or break linkage between existing declarations and definitions.\n"
    )


def _has_build_errors(return_code: int, stderr_text: str) -> bool:
    if return_code != 0:
        return True

    text = ANSI_ESCAPE.sub("", stderr_text or "")
    if not text.strip():
        return False

    ignored_warning_prefixes = (
        "warning:",
        "cmake warning",
        "cmake deprecation warning",
        "nvcc warning",
        "remark:",
    )
    ignored_warning_substrings = (
        "there was an error initializing an openfabrics device",
        "this warning is for project developers",
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(ignored_warning_prefixes):
            continue
        if any(msg in lowered for msg in ignored_warning_substrings):
            continue
        if re.search(r"\b(error|fatal|undefined reference)\b", line, re.IGNORECASE):
            return True

    return False


def summarize_build_errors(build_errors: str, max_lines: int = 80, max_chars: int = 8000) -> str:
    text = ANSI_ESCAPE.sub("", build_errors or "")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "No stderr output captured."

    error_like = []
    err_regex = re.compile(r"\b(error|fatal|undefined reference)\b", re.IGNORECASE)
    for ln in lines:
        if err_regex.search(ln):
            error_like.append(ln)

    chosen = error_like if error_like else lines[-max_lines:]
    summary = "\n".join(chosen[:max_lines])
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[truncated]..."
    return summary


def _update_average_runtime_csv(output_csv_path: str, app_name: str, build_dir_name: str, avg_value: float) -> None:
    output_file = Path(output_csv_path)

    if output_file.exists():
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        if reader:
            header = reader[0]
            rows = {row[0]: row[1:] for row in reader[1:] if row}
        else:
            header = ["App"]
            rows = {}

        if build_dir_name not in header:
            header.append(build_dir_name)
            for row in rows.values():
                row.append("")

        if app_name not in rows:
            rows[app_name] = [""] * (len(header) - 1)

        col_idx = header.index(build_dir_name) - 1
        rows[app_name][col_idx] = str(avg_value)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for app, row in rows.items():
                writer.writerow([app] + row)
    else:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["App", build_dir_name])
            writer.writerow([app_name, avg_value])


def _extract_runtime_us_from_stdout(stdout_text: str) -> float | None:
    text = ANSI_ESCAPE.sub("", stdout_text or "")
    if not text.strip():
        return None

    if re.search(r"(^|[^\w])FAIL([^\w]|$)", text, re.IGNORECASE):
        print("[runtime] build stdout reports FAIL; refusing to record runtime.")
        return None

    patterns = (
        re.compile(
            r"Average\s+kernel\s+execution\s+time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\((us|ms|ns|s)\)",
            re.IGNORECASE,
        ),
        re.compile(
            r"Average\s+execution\s+time(?:\s+of\s+(?:the|accuracy)\s+kernel\d*)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\((us|ms|ns|s)\)",
            re.IGNORECASE,
        ),
        re.compile(
            r"elapsed\s+time(?:\s+for\s+each\s+run)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|ns|s)\b",
            re.IGNORECASE,
        ),
    )
    unit_scale = {
        "us": 1.0,
        "ms": 1000.0,
        "ns": 0.001,
        "s": 1000000.0,
    }

    matches: list[tuple[float, str]] = []
    for pattern in patterns:
        for value, unit in pattern.findall(text):
            matches.append((float(value), unit.lower()))
        if matches:
            break

    if not matches:
        return None

    value, unit = matches[-1]
    return value * unit_scale[unit]


def _record_runtime_from_build_stdout(
    stdout_text: str,
    build_dir: Path,
    output_path: str,
    args,
    iteration: int,
) -> bool:
    runtime_us = _extract_runtime_us_from_stdout(stdout_text)
    if runtime_us is None:
        return False

    runtime_out_path = os.path.join(output_path, f"{args.name}_runtime_iter_{iteration}.txt")
    with open(runtime_out_path, "w", encoding="utf-8") as file:
        file.write(f"runtime_us={runtime_us:.12f}\n")

    print(f"[runtime] parsed runtime from build stdout: {runtime_us} us")

    if getattr(args, "output_file", None):
        try:
            _update_average_runtime_csv(
                output_csv_path=args.output_file,
                app_name=args.app_name,
                build_dir_name=build_dir.name,
                avg_value=runtime_us,
            )
            print(f"[runtime] updated CSV from build stdout: {args.output_file}")
        except Exception as exc:
            print(f"[runtime] failed to update CSV {args.output_file}: {exc}")

    return True


def _run_average_runtime(build_dir: Path, output_path: str, args, iteration: int) -> bool:
    avg_script = build_dir / "average_runtime.sh"
    if not avg_script.exists():
        print(f"[runtime] average runtime script not found: {avg_script}")
        return False

    print(f"[runtime] iteration {iteration}: running {avg_script}")
    try:
        avg_output = subprocess.check_output(
            ["bash", str(avg_script)],
            cwd=str(build_dir),
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        print(f"[runtime] average_runtime.sh failed with exit code {exc.returncode}")
        if exc.output:
            print(exc.output)
        return False

    avg_output_path = os.path.join(output_path, f"{args.name}_average_runtime_iter_{iteration}.txt")
    with open(avg_output_path, "w", encoding="utf-8") as file:
        file.write(avg_output + "\n")

    try:
        avg_value = float(avg_output)
    except ValueError:
        print(f"[runtime] Unexpected output from average_runtime.sh: {avg_output}")
        return False

    print(f"[runtime] iteration {iteration}: average_runtime.sh reported {avg_value} us")

    if getattr(args, "output_file", None):
        try:
            _update_average_runtime_csv(
                output_csv_path=args.output_file,
                app_name=args.app_name,
                build_dir_name=build_dir.name,
                avg_value=avg_value,
            )
            print(f"[runtime] updated CSV: {args.output_file}")
        except Exception as exc:
            print(f"[runtime] failed to update CSV {args.output_file}: {exc}")

    return True


def _write_build_logs_to_output(
    output_path: str,
    run_name: str,
    iteration: int,
    stdout_text: str,
    stderr_text: str,
) -> tuple[str, str]:
    """Persist raw build stdout/stderr into output_dir for each fix iteration."""
    out_copy_path = os.path.join(output_path, f"{run_name}_build_out_iter_{iteration}.txt")
    err_copy_path = os.path.join(output_path, f"{run_name}_build_err_iter_{iteration}.txt")

    with open(out_copy_path, "w", encoding="utf-8") as out_file:
        out_file.write(stdout_text or "")

    with open(err_copy_path, "w", encoding="utf-8") as err_file:
        err_file.write(stderr_text or "")

    return out_copy_path, err_copy_path


def _write_build_logs_to_dir(
    target_dir: str,
    run_name: str,
    iteration: int,
    stdout_text: str,
    stderr_text: str,
) -> tuple[str, str]:
    """Persist raw build stdout/stderr into a target directory for each iteration."""
    out_copy_path = os.path.join(target_dir, f"{run_name}_build_out_iter_{iteration}.txt")
    err_copy_path = os.path.join(target_dir, f"{run_name}_build_err_iter_{iteration}.txt")

    with open(out_copy_path, "w", encoding="utf-8") as out_file:
        out_file.write(stdout_text or "")

    with open(err_copy_path, "w", encoding="utf-8") as err_file:
        err_file.write(stderr_text or "")

    return out_copy_path, err_copy_path


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "runtime")).strip("._") or "runtime"


def _shared_runtime_extractor_dir() -> str:
    shared_dir = os.path.join(Path(__file__).resolve().parent, "runtime_extractors")
    os.makedirs(shared_dir, exist_ok=True)
    return shared_dir


def _runtime_extractor_script_path(app_name: str) -> str:
    safe_app_name = _safe_filename(app_name)
    return os.path.join(_shared_runtime_extractor_dir(), f"{safe_app_name}_runtime_extractor.sh")


def _prepare_runtime_extractor_prompt(build_stdout_text: str) -> str:
    return (
        "You are generating a bash script that extracts runtime information from a program output log.\n"
        "Return ONLY a bash script inside one fenced code block. Do not add explanation.\n"
        "Requirements:\n"
        "1. The script must accept exactly one argument: the path to the text log file.\n"
        "2. The script must parse the runtime from that file.\n"
        "3. Print the result in exactly one line using this format: runtime_ms=<number>\n"
        "4. If the log contains multiple runtime-like values, prefer the final end-to-end runtime or the most explicit average runtime.\n"
        "5. If no runtime can be found, print a clear error to stderr and exit with a non-zero status.\n"
        "6. Use portable bash plus standard tools like grep, sed, awk, or perl.\n"
        "7. Do not hardcode file paths.\n\n"
        "Here is the sample log:\n"
        "# Begin Log\n"
        f"{build_stdout_text}\n"
        "# End Log\n"
    )


def _ensure_runtime_extractor_script(output_path: str, app_name: str, llm, build_stdout_text: str) -> tuple[str, bool]:
    script_path = _runtime_extractor_script_path(app_name)
    if os.path.exists(script_path):
        return script_path, False

    prompt = _prepare_runtime_extractor_prompt(build_stdout_text)
    safe_app_name = _safe_filename(app_name)
    prompt_path = os.path.join(output_path, f"{safe_app_name}_runtime_extractor_prompt.txt")
    response_path = os.path.join(output_path, f"{safe_app_name}_runtime_extractor_response.txt")

    with open(prompt_path, "w", encoding="utf-8") as file:
        file.write(prompt)

    response = llm.submit_query_to_llm(prompt)
    with open(response_path, "w", encoding="utf-8") as file:
        file.write(response)

    script_text = _extract_candidate_code(response)
    if not script_text:
        raise ValueError("Could not extract runtime-extractor script from LLM response.")

    if not script_text.startswith("#!"):
        script_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + script_text.lstrip()

    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_text.rstrip() + "\n")
    os.chmod(script_path, 0o755)

    return script_path, True


def _run_runtime_extractor_script(
    script_path: str,
    build_stdout_path: str,
    output_path: str,
    run_name: str,
    iteration: int,
) -> tuple[bool, str, str]:
    proc = subprocess.run(
        ["bash", script_path, build_stdout_path],
        capture_output=True,
        text=True,
        check=False,
    )

    runtime_out_path = os.path.join(output_path, f"{run_name}_runtime_iter_{iteration}.txt")
    runtime_err_path = os.path.join(output_path, f"{run_name}_runtime_extract_err_iter_{iteration}.txt")

    with open(runtime_out_path, "w", encoding="utf-8") as out_file:
        out_file.write(proc.stdout or "")

    with open(runtime_err_path, "w", encoding="utf-8") as err_file:
        err_file.write(proc.stderr or "")

    return proc.returncode == 0, runtime_out_path, runtime_err_path


RE_TRIPLE_BACKTICK = re.compile(
    r"```(?:[^\n`]*\n)?(.*?)```",
    re.DOTALL
)

# 2) '''lang\n ... \n'''
RE_TRIPLE_SINGLE = re.compile(
    r"'''(?:[^\n']*\n)?(.*?)'''",
    re.DOTALL
)

def extract_code_blocks(text: str, join_with: str = "\n\n"):
    """
    Extracts all code blocks delimited by triple fences and concatenates them.

    Supports:
      - ```code``` or ```lang\ncode```
      - '''code''' or '''lang\ncode'''
    Also captures a trailing, incomplete fenced block at EOF.

    Returns:
        (concatenated, blocks)
    """
    blocks = []

    i = 0
    n = len(text)
    open_fence = None          # "```" or "'''"
    content_start = None       # index where code content starts (after optional lang line)

    while i <= n:
        if open_fence is None:
            # look for next opening fence
            idx_bt = text.find("```", i)
            idx_sq = text.find("'''", i)

            if idx_bt == -1 and idx_sq == -1:
                break

            # pick earliest
            if idx_bt != -1 and (idx_sq == -1 or idx_bt < idx_sq):
                fence = "```"
                start = idx_bt
            else:
                fence = "'''"
                start = idx_sq

            # compute content start (skip optional lang line)
            j = start + len(fence)
            if j < n and text[j] == "\n":
                content_start = j + 1
            else:
                nl = text.find("\n", j)
                content_start = (nl + 1) if nl != -1 else j

            open_fence = fence
            i = content_start
        else:
            # look for the matching closing fence
            end = text.find(open_fence, i)
            if end == -1:
                # Incomplete final block: take till EOF
                block = text[content_start:n].strip("\n")
                if block:
                    blocks.append(block)
                break
            else:
                block = text[content_start:end].strip("\n")
                if block:
                    blocks.append(block)
                # move past the closing fence and continue scanning
                i = end + len(open_fence)
                open_fence = None
                content_start = None

    concatenated = join_with.join(blocks)
    return concatenated, blocks


def _extract_code_from_response(text: str) -> str:
    """If the response has fenced code blocks, concatenate those; otherwise return full text."""
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+\-]*)\n(.*?)```", text, flags=re.DOTALL)
    return "\n\n".join(blocks) if blocks else text

def _normalize_type(param: str) -> str:
    # Drop parameter names and array extents; keep a rough 'type signature'
    p = param.strip()
    # remove default values
    p = re.sub(r"=\s*[^,]+", "", p)
    # remove array suffixes like [][N]
    p = re.sub(r"\s*\[[^\]]*\]", "", p)
    # remove parameter name (last identifier if there's a space)
    # but keep pointers/references next to the type
    # e.g., 'double * __restrict__ a' -> 'double * __restrict__'
    tokens = p.split()
    if not tokens:
        return ""
    # heuristic: remove the last token if it's a valid identifier and previous token is not '*' or '&'
    if re.match(r"[A-Za-z_]\w*$", tokens[-1]):
        # but if preceding char in original string right before name is '*' or '&', keep it with the type
        before = p[: p.rfind(tokens[-1])]
        p = before.strip()
    # collapse whitespace
    p = re.sub(r"\s+", " ", p).strip()
    return p

def _parse_kernel_defs(code: str) -> dict:
    """Return mapping: kernel_name -> signature string (normalized)."""
    # Allow optional __launch_bounds__ attributes between qualifiers and void
    pattern = re.compile(
        r"""__global__\s*
            (?:\(\s*__launch_bounds__\s*\([^)]*\)\s*\))?
            (?:__launch_bounds__\s*\([^)]*\)\s*)*
            (?:\s*__forceinline__\s*)*
            void\s+([A-Za-z_]\w*)\s*\((.*?)\)""",
        re.DOTALL | re.VERBOSE,
    )
    kernels = {}
    for m in pattern.finditer(code):
        name = m.group(1)
        params = m.group(2)
        # split top-level commas (not inside parens) — simple split works for CUDA C params
        parts = [p for p in (s.strip() for s in params.split(",")) if p]
        norm_parts = [_normalize_type(p) for p in parts]
        signature = f"void {name}(" + ", ".join(norm_parts) + ")"
        kernels[name] = signature
    return kernels

def _parse_kernel_launches(code: str) -> list:
    """Return list of kernel names launched via <<< >>>."""
    # Capture qualified names like ns::foo<<<...>>>
    launches = re.findall(r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*<<<", code)
    # De-qualify (strip namespaces)
    clean = [n.split("::")[-1] for n in launches]
    return clean

def analyze_cuda_response(source_code: str, response_text: str) -> dict:

    response_code = _extract_code_from_response(response_text)

    src_kernels = _parse_kernel_defs(source_code)
    rsp_kernels = _parse_kernel_defs(response_code)

    src_launches = _parse_kernel_launches(source_code)
    rsp_launches = _parse_kernel_launches(response_code)

    src_kernel_set = set(src_kernels)
    rsp_kernel_set = set(rsp_kernels)

    added_defs = sorted(rsp_kernel_set - src_kernel_set)
    removed_defs = sorted(src_kernel_set - rsp_kernel_set)

    rsp_launch_set = set(rsp_launches)
    src_launch_set = set(src_launches)

    rsp_launch_unknown_in_response = sorted([k for k in rsp_launch_set if k not in rsp_kernel_set])
    rsp_launch_unknown_in_source = sorted([k for k in rsp_launch_set if k not in src_kernel_set])

    removed_launches = sorted(list(src_launch_set - rsp_launch_set))
    added_launches = sorted(list(rsp_launch_set - src_launch_set))

    mismatches = {}
    for name in sorted(src_kernel_set & rsp_kernel_set):
        if src_kernels[name] != rsp_kernels[name]:
            mismatches[name] = {"source": src_kernels[name], "response": rsp_kernels[name]}

    report = {
        "kernels_in_source": sorted(src_kernel_set),
        "kernels_in_response": sorted(rsp_kernel_set),
        "kernels_added_in_response": added_defs,
        "kernels_removed_in_response": removed_defs,
        "kernel_signature_mismatches": mismatches,
        "kernel_launches_in_source": sorted(src_launch_set),
        "kernel_launches_in_response": sorted(rsp_launch_set),
        "launches_added_in_response": added_launches,
        "launches_removed_in_response": removed_launches,
        "response_launches_unknown_in_response_code": rsp_launch_unknown_in_response,
        "response_launches_unknown_in_source_code": rsp_launch_unknown_in_source,
    }
    return report

def get_output_folder():
    """
    Adds an output folder selection widget in Streamlit's sidebar.
    Returns the selected folder path.
    """
    st.sidebar.title("Output Folder Selection")
    output_folder = st.sidebar.text_input(
        "Specify Output Folder Path",
        value="./results",
        help="Enter the path where result files will be stored."
    )

    # Ensure directory exists or create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        st.sidebar.success(f"Created directory: {output_folder}")

    return output_folder

def parse_input_folder(full_path):
    """Parses and returns the input folder name from the full file path."""
    return os.path.basename(os.path.dirname(full_path)), os.path.basename(full_path)


def construct_src(pc_selected, ia_selected, roofline_selected):
    """Constructs the Src string based on selected sources."""
    sources = []
    if pc_selected:
        sources.append("pc")
    if ia_selected:
        sources.append("ia")
    if roofline_selected:
        sources.append("roofline")
    return "+".join(sources)


def process_and_save_results(output_path, pc_selected, ia_selected, roofline_selected, optimizations):
    """Processes input information, constructs dictionary, and writes results to CSV."""
    folder_name, app_name = parse_input_folder(output_path)
    src = construct_src(pc_selected, ia_selected, roofline_selected)

    write_results_to_csv(output_path, app_name, src, optimizations)


def sanitize_text(text):
    if isinstance(text, str):
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text

# ------------- Utility Functions -------------


def extract_code_block(response: str) -> str:
    """
    Extracts the first code block (optionally labeled cpp) from the LLM response.
    Supports cases like ```cpp ... ```, ``` ... ```, or ``` cpp ... ```
    """
    # Match anything between ``` (with optional language identifier) and ```
    code_pattern = r"```(?:cpp)?\s*([\s\S]*?)```"
    match = re.search(code_pattern, response)
    
    if match:
        code = match.group(1).strip()
        return code
    else:
        raise ValueError("Code block not found in the response.")

def extract_optimization_list(response_text):
    """
    Extract the `optimizations = [...]` list from the model's response.
    Tries multiple fallback strategies to parse it robustly.
    """
    try:
        # Clean up any code block markers
        cleaned = response_text.replace("```cpp", "").replace("```", "").strip()

        # Try to find the optimizations = [...] block anywhere
        pattern = r"optimizations\s*=\s*(\[\s*{[\s\S]*?}\s*\])"
        match = re.search(pattern, cleaned)
        if not match:
            raise ValueError("Optimizations list not found in the response.")

        optimizations_block = match.group(1)
        print(match.group)
        # Convert the text list to actual Python list
        optimizations = ast.literal_eval(optimizations_block)

        # Optionally: Validate format
        for item in optimizations:
            if not isinstance(item, dict) or 'lines' not in item or 'reason' not in item:
                raise ValueError("Malformed optimization item: missing keys.")

        return optimizations

    except Exception as e:
        print("Failed to extract optimization list.")
        print("Full Response Tail:\n", response_text[-500:])
        raise ValueError("Optimizations list not found or malformed in the response.") from e



############
def format_optimizations_markdown(parsed_result):
    optimizations, suggestions = parsed_result

    markdown = "### Optimizations Applied\n"
    if optimizations:
        for opt in optimizations:
            if isinstance(opt, dict) and 'lines' in opt and 'reason' in opt:
                lines = ', '.join(map(str, opt['lines']))
                reason = opt['reason']
                markdown += f"- **Lines [{lines}]**: {reason}\n"
            else:
                markdown += f"- Malformed optimization entry: {opt}\n"
    else:
        markdown += "- None\n"

    lines = ''
    reason = ''
    markdown += "\n### Suggestions Not Applied\n"
    if suggestions:
        for sugg in suggestions:
            if isinstance(sugg, dict) and 'lines' in sugg and 'reason' in sugg:
                lines = ', '.join(map(str, sugg['lines']))
                reason = sugg['reason']
                markdown += f"- **Lines [{lines}]**: {reason}\n"
            else:
                markdown += f"- Malformed suggestion entry: {opt}\n"
    else:
        markdown += "- None\n"

    return markdown
def format_optimizations_markdown2(result_tuple):
    optimizations, suggestions = result_tuple
    markdown = ""

    # Section for applied optimizations
    if isinstance(optimizations, list) and optimizations:
        markdown += "###Optimizations Performed\n"
        for i, opt in enumerate(optimizations, 1):
            if isinstance(opt, dict):
                lines = ', '.join(map(str, opt.get('lines', [])))
                reason = opt.get('reason', 'No reason provided.')
                markdown += f"**{i}.** Lines `{lines}` — {reason}\n\n"
            else:
                markdown += f"**{i}.** Malformed optimization entry: `{opt}`\n\n"

    # Section for suggestions not applied
    if isinstance(suggestions, list) and suggestions:
        markdown += "---\n### Suggestions (Not Applied)\n"
        for i, sug in enumerate(suggestions, 1):
            if isinstance(sug, dict):
                reason = sug.get('reason#', 'No reason provided.')
                markdown += f"**{i}.** {reason}\n\n"
            else:
                markdown += f"**{i}.** `{sug}`\n\n"

    return markdown

def format_optimizations_markdown3(optimizations):
    """
    Convert a list of optimization dictionaries to clean markdown.
    Each dict should have 'lines' and 'reason' keys.
    """
    if not optimizations:
        return "_No optimizations found._"

    markdown = "###Optimization Explanations\n\n"
    for idx, opt in enumerate(optimizations, start=1):
        lines = ', '.join(map(str, opt['lines']))
        reason = opt['reason']
        markdown += (
            f"**Optimization {idx}**  \n"
            f"**Code Lines:** {lines}  \n"
            f"**Reason:** {reason}  \n\n"
        )
    return markdown






# ----------------------------------------------

def extract_cuda_kernels(content):
    pattern = r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)"
    return [f"{name}({params})" for name, params in re.findall(pattern, content)]

def remove_similar_entries(descriptions, threshold=0.85):
    unique = []
    for desc in descriptions:
        if not any(Levenshtein.ratio(desc, existing) > threshold for existing in unique):
            unique.append(desc)
    return unique

def parse_chatgpt_response(response):
    code_block_pattern = r"```[a-z]*\n(.*?)```"
    code_match = re.search(code_block_pattern, response, re.DOTALL)

    if code_match:
        optimized_code = code_match.group(1).strip()
        explanation = response[code_match.end():].strip()
    else:
        optimized_code = "Optimized code not found in response."
        explanation = response.strip()

    return optimized_code, explanation

def parse_response(response):
    parts = response.split("Explanation:", 1)
    code = parts[0].replace("Optimized Code:", "").strip()
    explanation = parts[1].strip() if len(parts) > 1 else "No detailed explanation provided."
    return code, explanation

def prepare_sources_to_submit(content, pc_data=None, important_counters=None, filtered_roofline_data=None, filename=None, args=None):
    if _is_finetune_compact_prompt_style(args):
        combined_query = _build_finetune_compact_prompt(
            content=content,
            pc_data=pc_data,
            important_counters=important_counters,
            filtered_roofline_data=filtered_roofline_data,
            args=args,
        )
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(combined_query)
        return combined_query

    formal_reasoning_mode = getattr(args, "formal_reasoning_mode", "enabled")
    formal_reasoning_enabled = formal_reasoning_mode != "disabled"

    combined_query = f"***Code:***\n\n{content}\n\n"
    sections = []

    # Check explicitly for DataFrame validity
    if pc_data is not None:
        stall_info = "### STALL ANALYSIS:\n"
        # large_stalls = find_large_stalls(pc_data, threshold=0.1)
        for i, stall in enumerate(pc_data, 1):
            stall_reasons = ', '.join(stall['Stall Reasons'])
            stall_info += f"{i}. Line {stall['Line No']} has high stalls due to: {stall_reasons}.\n"
        sections.append(stall_info)

    if important_counters is not None and len(important_counters) > 0:
        counter_info = "### IMPORTANT HARDWARE COUNTERS:\n"
        for counter in important_counters:
            counter_info += (f"Kernel: {counter['Kernel']}, "
                             f"Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, "
                             f"Impact: {counter['Value']}, Description: {counter['Description']}.\n")
        sections.append(counter_info)

    if filtered_roofline_data is not None and len(filtered_roofline_data) > 0:
        roofline_info = "### ROOFLINE ANALYSIS:\n"
        idx = 1
        for kernel_name, comments in filtered_roofline_data.items():
            for comment in comments:
                roofline_info += f"{idx}. Kernel '{kernel_name}': {comment}\n"
                idx += 1
        sections.append(roofline_info)

    if sections:
        combined_query += "***Performance Analysis Data:***\n\n"
        combined_query += "\n".join(sections)
        combined_query += "\n\n"

    # Legacy prompt block retained for comparability with previously collected data.
    # combined_query += (
    # "***Instructions:***\n\n"
    # "You are an HPC performance optimization expert. Optimize the provided CUDA code specifically for an NVIDIA H100 GPU.\n\n"
    # "Optimize the following HPC code for execution time. "
    # "When presenting your optimized code:\n"
    # "1. Clearly reference the provided performance analysis data by number or explicitly quoted text in each inline comment to justify why each specific optimization was applied.\n"
    # "2. Keep all changes inside the existing kernel/function bodies whenever possible.\n"
    # "3. Do NOT introduce any new helper functions, utility functions, macros, templates, classes, structs, global variables, or other new top-level symbols outside the existing kernels/functions.\n"
    # "4. Do NOT add wrapper load helpers such as ro(), custom ldg wrappers, or any new reusable abstraction outside the existing kernel/function body.\n"
    # "5. Do NOT increase the total shared-memory footprint of any kernel beyond the original source. Do not enlarge existing __shared__ arrays, do not add new shared-memory tiles, and prefer register/cache/read-only load optimizations over larger shared-memory staging.\n"
    # "**Required Output Format:**\n"
    # "1. Provide the complete optimized CUDA kernel wrapped within ```cpp``` code blocks.\n"
    # "6. Immediately after the code, explicitly list each optimization THAT WERE PERFORMED as a structured list named `optimizations`, using the following exact format:\n\n"
    # "optimizations = [\n"
    # "    {'lines': [line_numbers], 'reason': 'brief reason'},\n"
    # "    {'lines': [line_numbers], 'reason': 'another reason'},\n"
    # "    # ... more reasons\n"
    # "]\n"
    # "7. Then give a list named `suggested_but_not_applied`. This list contains ideas that were considered based on the performance data but not implemented due to uncertainty or risk. These are actionable suggestions for a human expert to evaluate further. Follow this exact format:\n\n"
    # "suggested_but_not_applied = [\n"
    # "    {'lines': [line_numbers], 'reason': 'brief reason'},\n"
    # "    {'lines': [line_numbers], 'reason': 'another reason'},\n"
    # "    # ... more reasons\n"
    # "]\n"
    # "**Important:** Ensure no additional explanation or text is added beyond the code block, the `optimizations = [...]` and the `suggested_but_not_applied = [...]` blocks."
    # )
    combined_query += (
    "***Instructions:***\n\n"
    "You are an HPC performance optimization expert. Optimize the provided CUDA code specifically for an NVIDIA H100 GPU.\n\n"
    "Optimize the following HPC code for execution time. "
    "When presenting your optimized code:\n"
    "1. Clearly reference the provided performance analysis data by number or explicitly quoted text in each inline comment to justify why each specific optimization was applied.\n"
    "2. Keep all changes inside the existing kernel/function bodies whenever possible.\n"
    "3. Do NOT introduce any new helper functions, utility functions, macros, templates, classes, structs, global variables, or other new top-level symbols outside the existing kernels/functions.\n"
    "4. Do NOT add wrapper load helpers such as ro(), custom ldg wrappers, or any new reusable abstraction outside the existing kernel/function body.\n"
    "5. Do NOT increase the total shared-memory footprint of any kernel beyond the original source. Do not enlarge existing __shared__ arrays, do not add new shared-memory tiles, and prefer register/cache/read-only load optimizations over larger shared-memory staging.\n"
    "6. The first ```cpp``` block must contain the full compilable source code that will replace the original file.\n"
    "7. Whitespace-only, formatting-only, comment-only, or reorder-only rewrites are invalid and must not be returned as optimized code.\n"
    "8. When the performance data indicates safe opportunities, apply at least one substantive performance-motivated code change in the code itself rather than only describing ideas in prose.\n"
    "9. If no safe substantive optimization can be justified from the provided performance data, keep the code semantically unchanged and state that explicitly in the `optimizations` list instead of inventing a fake optimization.\n"
    "**Required Output Format:**\n"
    "1. Provide the complete optimized CUDA kernel wrapped within ```cpp``` code blocks.\n"
    "2. Immediately after the code, explicitly list each optimization THAT WERE PERFORMED as a structured list named `optimizations`, using the following exact format:\n\n"
    "optimizations = [\n"
    "    {'lines': [line_numbers], 'reason': 'brief reason'},\n"
    "    {'lines': [line_numbers], 'reason': 'another reason'},\n"
    "    # ... more reasons\n"
    "]\n"
    "3. Then give a list named `suggested_but_not_applied`. This list contains ideas that were considered based on the performance data but not implemented due to uncertainty or risk. These are actionable suggestions for a human expert to evaluate further. Follow this exact format:\n\n"
    "suggested_but_not_applied = [\n"
    "    {'lines': [line_numbers], 'reason': 'brief reason'},\n"
    "    {'lines': [line_numbers], 'reason': 'another reason'},\n"
    "    # ... more reasons\n"
    "]\n"
    "**Important:** Ensure no additional explanation or text is added beyond the code block, the `optimizations = [...]` and the `suggested_but_not_applied = [...]` blocks."
    )

    if formal_reasoning_enabled:
        combined_query = combined_query.replace(
            "Optimize the following HPC code for execution time. ",
            "Optimize the following HPC code for execution time. You must define mathematical preconditions and postconditions, provide explicit loop invariants to guarantee semantic equivalence, and provide a logical proof showing why the invariants remain unbroken after your structural optimization.\n\n",
            1,
        )

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(combined_query)

    return combined_query


def prepare_sources_multi_gpu_to_submit(
    content,
    pc_data=None,
    important_counters=None,
    filtered_roofline_data=None,
    filename=None,
    args=None,
    user_guidance=None,
    extra_summary_sections=None,
):
    if _is_finetune_compact_prompt_style(args):
        combined_query = _build_finetune_compact_prompt(
            content=content,
            pc_data=pc_data,
            important_counters=important_counters,
            filtered_roofline_data=filtered_roofline_data,
            args=args,
            user_guidance=user_guidance,
            extra_summary_sections=extra_summary_sections,
        )
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(combined_query)
        return combined_query

    formal_reasoning_mode = getattr(args, "formal_reasoning_mode", "enabled")
    formal_reasoning_enabled = formal_reasoning_mode != "disabled"

    combined_query = f"# Begin Source Code\n```cpp```\n{content}\n```cpp```\n# End Source Code\n"
    sections = []

    # Check explicitly for DataFrame validity
    if pc_data is not None:
        stall_info = "### STALL ANALYSIS:\n"
        # large_stalls = find_large_stalls(pc_data, threshold=0.1)
        for i, stall in enumerate(pc_data, 1):
            stall_reasons = ', '.join(stall['Stall Reasons'])
            stall_info += f"{i}. Line {stall['Source']} has high stalls due to: {stall_reasons}.\n"
        sections.append(stall_info)


    if important_counters is not None and len(important_counters) > 0:
        counter_info = "### IMPORTANT HARDWARE COUNTERS:\n"
        for counter in important_counters:
            counter_info += (f"Kernel: {counter['Kernel']}, "
                             f"Group: {counter['Group Name']}, Counter: {counter['Counter Name']}, "
                             f"Impact: {counter['Value']}, Description: {counter['Description']}.\n")
        sections.append(counter_info)

    if filtered_roofline_data is not None and len(filtered_roofline_data) > 0:
        roofline_info = "### ROOFLINE ANALYSIS:\n"
        idx = 1
        for kernel_name, comments in filtered_roofline_data.items():
            for comment in comments:
                roofline_info += f"{idx}. Kernel '{kernel_name}': {comment}\n"
                idx += 1
        sections.append(roofline_info)

    if extra_summary_sections:
        sections.extend(extra_summary_sections)

    if user_guidance:
        if isinstance(user_guidance, str):
            guidance_lines = [ln.strip() for ln in user_guidance.splitlines() if ln.strip()]
        elif isinstance(user_guidance, (list, tuple)):
            guidance_lines = [str(ln).strip() for ln in user_guidance if str(ln).strip()]
        else:
            guidance_lines = [str(user_guidance).strip()]
        combined_query += "# User guidance- (Follow this with the highest priority)"
        combined_query += "\n" + "\n".join(guidance_lines)
        combined_query += "\n# End User guidance\n"

    if args is not None and _is_gemini_model(args):
        combined_query += "# Gemini compatibility guardrails- (Follow this with the highest priority)\n"
        combined_query += _gemini_negative_guidance()
        combined_query += "# End Gemini compatibility guardrails\n"

    if sections:
        combined_query += "# Begin Performance Analysis\n\n"
        combined_query += "\n".join(sections)
        combined_query += "\n# End Performance Analysis\n"

    combined_query += "\n\n# Begin H100 Optimization Guidance\n"
    combined_query += (
        "Optimize specifically for NVIDIA H100 (Hopper) by targeting L2/cache reuse, HBM3 bandwidth, and Tensor Core throughput. "
        "Prioritize tiling and data locality: make global accesses coalesced/contiguous, reuse data from registers/shared memory/L2, "
        "and reduce redundant global traffic. Use Hopper async pipelines to hide memory latency: stage global→shared with cp.async / "
        "cuda::memcpy_async using double/triple buffering so loads overlap compute; keep the pipeline full instead of blocking on loads. "
        "If compute is matrix/attention-like, map work to Tensor Cores (WMMA/CUTLASS/cuBLASLt) and use mixed precision (FP16/BF16/TF32, "
        "FP8 when numerically safe) to increase throughput. Tune launch configuration to avoid occupancy cliffs: pick block sizes that sustain "
        "high occupancy without excessive register pressure or shared-memory usage; avoid large per-thread local arrays that cause spills. "
        "Reduce warp inefficiency: minimize divergence, expensive atomics, and unnecessary __syncthreads(); prefer warp-level primitives "
        "and cooperative groups where applicable. If memory-bound, reduce passes and consider fusion/batching to increase L2 hit rate; "
        "if compute-bound, increase ILP, unroll judiciously, and favor FMA-friendly instruction patterns."
    )
    combined_query += "\n# End H100 Optimization Guidance\n"
    combined_query += "\n# Begin Entrypoint Guardrails\n"
    combined_query += _entrypoint_guardrails(content)
    combined_query += "# End Entrypoint Guardrails\n"
    
    # Legacy prompt block retained for comparability with previously collected data.
    # combined_query += (
    # "# Instructions\n"
    # "You must optimize the source code provided above using only the performance data provided above. \n\n"
    # "Optimize the following HPC code for execution time.\n"
    # "Can you give me the fixed optimized code that will replace the full source code including headers and main function body?\n"
    # "2. Do not add line numbers to the generated code. \n"
    # "3. Do not rename or duplicate any kernel or function.\n"
    # "4. Do not change any function signatures.\n"
    # "5. Preserve the existing entry-point structure exactly. If the file already has main(), keep exactly one equivalent main(); if it does not, do not add one.\n"
    # "6. Do not add duplicate definitions for main or for any externally linked function.\n"
    # "7. Do not turn declarations into disconnected definitions or break declaration/definition linkage.\n"
    # "8. Keep all changes inside the existing kernel/function bodies whenever possible.\n"
    # "9. Do NOT introduce any new helper functions, utility functions, macros, templates, classes, structs, global variables, or other new top-level symbols outside the existing kernels/functions.\n"
    # "10. Do NOT add wrapper load helpers such as ro(), custom ldg wrappers, or any new reusable abstraction outside the existing kernel/function body.\n"
    # "11. Do NOT increase the total shared-memory footprint of any kernel beyond the original source. Do not enlarge existing __shared__ arrays, do not add new shared-memory tiles, and prefer register/cache/read-only load optimizations over larger shared-memory staging.\n"
    # "12. Use the performance analysis reference number(s) to guide your changes.\n"
    # "13. Return:\n"
    # "   (a) the full source code including the optimized kernels inside ``` block (starts with ``` and ends with ``` )\n"
    # "   (b) optimizations list = [{'lines': [...], 'reason': '...'}]\n"
    # "   (c) suggested_but_not_applied list = [{'lines': [...], 'reason': '...'}]\n"
    # "14. Do not generate more than one copy of the same kernel.\n"
    # "15. Do not invent any new kernels or placeholder code.\n\n"
    # "ONLY respond with the updated code and two structured lists. No explanation, no notes, no markdown titles. Follow this format exactly."
    # )
    combined_query += (
    "# Instructions\n"
    "You must optimize the source code provided above using only the performance data provided above. \n\n"
    "Optimize the following HPC code for execution time.\n"
    "Can you give me the fixed optimized code that will replace the full source code including headers and main function body?\n"
    "2. Do not add line numbers to the generated code. \n"
    "3. Do not rename or duplicate any kernel or function.\n"
    "4. Do not change any function signatures.\n"
    "5. Preserve the existing entry-point structure exactly. If the file already has main(), keep exactly one equivalent main(); if it does not, do not add one.\n"
    "6. Do not add duplicate definitions for main or for any externally linked function.\n"
    "7. Do not turn declarations into disconnected definitions or break declaration/definition linkage.\n"
    "8. Keep all changes inside the existing kernel/function bodies whenever possible.\n"
    "9. Do NOT introduce any new helper functions, utility functions, macros, templates, classes, structs, global variables, or other new top-level symbols outside the existing kernels/functions.\n"
    "10. Do NOT add wrapper load helpers such as ro(), custom ldg wrappers, or any new reusable abstraction outside the existing kernel/function body.\n"
    "11. Do NOT increase the total shared-memory footprint of any kernel beyond the original source. Do not enlarge existing __shared__ arrays, do not add new shared-memory tiles, and prefer register/cache/read-only load optimizations over larger shared-memory staging.\n"
    "12. Use the performance analysis reference number(s) to guide your changes.\n"
    "13. The first fenced code block must be the full compilable replacement source file.\n"
    "14. Whitespace-only, formatting-only, comment-only, or reorder-only rewrites are invalid and must not be returned as optimized code.\n"
    "15. When the performance data supports it, apply at least one substantive performance-motivated code change in the code itself.\n"
    "16. If no safe substantive optimization can be justified from the provided performance data, keep the code semantically unchanged and say so explicitly in the `optimizations` list instead of claiming changes that were not applied.\n"
    "17. Return:\n"
    "   (a) the full source code including the optimized kernels inside ``` block (starts with ``` and ends with ``` )\n"
    "   (b) optimizations list = [{'lines': [...], 'reason': '...'}]\n"
    "   (c) suggested_but_not_applied list = [{'lines': [...], 'reason': '...'}]\n"
    "18. Do not generate more than one copy of the same kernel.\n"
    "19. Do not invent any new kernels or placeholder code.\n\n"
    "ONLY respond with the updated code and two structured lists. No explanation, no notes, no markdown titles. Follow this format exactly."
    )

    if formal_reasoning_enabled:
        combined_query = combined_query.replace(
            "Optimize the following HPC code for execution time.\n",
            "Optimize the following HPC code for execution time. You must define mathematical preconditions and postconditions, provide explicit loop invariants to guarantee semantic equivalence, and provide a logical proof showing why the invariants remain unbroken after your structural optimization.\n",
            1,
        )

    approx_tokens = len(combined_query.split()) * 1.3
    print(f"Estimated tokens: {int(approx_tokens)}")

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(combined_query)

    return combined_query

def prepare_second_prompt(original_source, optimized_code, build_errors, filename="second_prompt.txt"):

    combined_query = f"After requesting for optimized code from you, I have gotten the following code:\n"
    combined_query += f"# Begin optimized Code\n```cpp```\n{optimized_code}\n```cpp```\n# End optimized Code\n"

    combined_query += f"While building we get the following build errors: {build_errors}\n"

    combined_query += "\nBelow is added the original source code.\n"
    combined_query += f"# Begin Source Code\n```cpp```\n{original_source}\n```cpp```\n# End Source Code\n"

    combined_query += (
    "# Instructions\n"
    "Can you give me the fixed optimized code that will replace the full source code including headers and main function body?\n"
    "Return exactly one fenced code block containing only compilable source code, then the two structured lists.\n"
    "Preserve the original entry-point structure exactly: do not add, duplicate, rename, or remove main().\n"
    "Do not add helper executables, wrapper mains, or alternate standalone programs.\n"
    "Do not change externally referenced function names or break linkage between declarations and definitions.\n"
    "Keep all changes inside the existing kernel/function bodies whenever possible.\n"
    "Do NOT introduce any new helper functions, utility functions, macros, templates, classes, structs, global variables, or other new top-level symbols outside the existing kernels/functions.\n"
    "Do NOT add wrapper load helpers such as ro(), custom ldg wrappers, or any new reusable abstraction outside the existing kernel/function body.\n"
    "Do NOT increase the total shared-memory footprint of any kernel beyond the original source. Do not enlarge existing __shared__ arrays, do not add new shared-memory tiles, and prefer register/cache/read-only load optimizations over larger shared-memory staging.\n"
    "Do not use cuda::pipeline, cuda::barrier, cuda::std::pipeline, cuda::std::barrier, "
    "cuda::memcpy_async, cuda::std::memcpy_async, cp.async, or add <cuda/pipeline> / <cuda/barrier> headers."
    )

    approx_tokens = len(combined_query.split()) * 1.3
    print(f"Estimated tokens: {int(approx_tokens)}")

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(combined_query)

    return combined_query


def prepare_second_prompt_for_style(
    original_source,
    optimized_code,
    build_errors,
    filename="second_prompt.txt",
    args=None,
):
    if not _is_finetune_compact_prompt_style(args):
        return prepare_second_prompt(
            original_source=original_source,
            optimized_code=optimized_code,
            build_errors=build_errors,
            filename=filename,
        )

    benchmark = _infer_benchmark_name(args, original_source)
    variant = str(getattr(args, "name", "") or getattr(args, "app_name", "") or "default").strip() or "default"
    build_write_mode = str(getattr(args, "build_write_mode", "") or "").strip().lower()
    build_target_source = str(getattr(args, "build_target_source", "") or "").strip()
    if benchmark == "sw4lite" and build_write_mode == "merge-kernels":
        instruction = (
            "Fix the optimized CUDA kernel definition(s) so they compile successfully with the provided build errors. "
            "Return only corrected __global__ kernel definition(s) that can be merged into "
            f"{build_target_source or 'the existing build target'}. "
            "Do not return a full standalone source file, includes, markdown fences, or prose. "
            "Preserve the original kernel name(s), parameter list(s), and external behavior. "
            "Reuse values already brought into shared memory or registers before touching global memory again, hoist invariant index arithmetic/flags/coefficient gathers "
            "out of the hot loop where safe, and use the read-only path only for truly read-only coefficient/material arrays while preserving the existing stencil/shared-memory structure."
        )
    else:
        instruction = (
            "Fix the optimized GPU kernel so it compiles successfully with the provided build errors. "
            "Preserve correctness and external behavior. Return corrected code only."
        )
    combined_query = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"### Benchmark\n{benchmark}\n\n"
        f"### Variant\n{variant}\n\n"
        "### Target\nNVIDIA H100 / CUDA\n\n"
        f"### Original Source Code\n{original_source.strip()}\n\n"
        f"### Current Optimized Code\n{optimized_code.strip()}\n\n"
        f"### Build Errors\n{build_errors.strip()}\n\n"
        "### Response:\n"
    )

    with open(filename, "w", encoding="utf-8") as file:
        file.write(combined_query)

    return combined_query


def balance_brackets(code: str) -> str:
    """Simple fixer for missing '}' or ']' in a list of dicts."""
    open_braces = code.count('{')
    close_braces = code.count('}')
    if open_braces > close_braces:
        code += '}' * (open_braces - close_braces)

    open_brackets = code.count('[')
    close_brackets = code.count(']')
    if open_brackets > close_brackets:
        code += ']' * (open_brackets - close_brackets)

    return code

def safe_parse_list_from_string(string):
    """
    Attempts to safely parse a Python list from a string, even if it ends with a comma before closing bracket.
    """
    try:
        # Remove any trailing comma before the closing bracket
        string = re.sub(r',\s*\]', ']', string)
        return ast.literal_eval(string)
    except Exception as e:
        st.error(f"Failed to parse list from string: {e}")
        return []

def safe_extract_block2(name, source_text):
    try:
        match = re.search(fr"{name}\s*=\s*(\[[\s\S]*?\])", source_text)
        if match:
            block_text = match.group(1)
            block_text = balance_brackets(block_text)
            return ast.literal_eval(block_text)
        else:
            print(f"Warning: Block '{name}' not found.")
    except Exception as e:
        print(f"Failed to parse '{name}':", e)
    return []  # fallback to empty list
    print("\nParsed Suggestions (not applied):")
    for sug in suggestions:
        print(sug)

    return optimizations, suggestions

def parse_llm_response_safe(response_text):
    # Remove any code blocks
    newtext = re.sub(r"```.*?```", "", response_text, flags=re.DOTALL).strip()
    response_text = response_text.strip()

    optimizations_text = ""
    suggestions_text = ""

    if "suggested_but_not_applied" in response_text:
        parts = response_text.split("suggested_but_not_applied =")
        optimizations_text = parts[0].split("optimizations =")[-1].strip()
        suggestions_text = parts[1].strip()

    optimizations = safe_parse_list_from_string(optimizations_text)
    suggested = safe_parse_list_from_string(suggestions_text)

    return optimizations, suggested

def optimization_list(text):
    
    newtext=re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    newtext = newtext.lstrip()
    newtext= newtext.rstrip()
    # Lambda to strip the prefix and convert to list of dicts
    extract_optimizations = lambda s: ast.literal_eval(s.split('=', 1)[1].strip())

    # Use the lambda
    optimizations = extract_optimizations(newtext)

    for opt in optimizations:
        print(opt)
    return optimizations #gonig to be a list

def call_for_optimization(
    updated_code,
    output_path,
    args,
    llm,
    iteration_count,
    pc_data = None,
    important_counters = None,
    kernel_dict = None,
    user_guidance = None,
    extra_summary_sections = None,
):
    query = prepare_sources_multi_gpu_to_submit(
        content=updated_code,
        pc_data=pc_data,
        important_counters=important_counters,
        filtered_roofline_data=kernel_dict,
        filename=os.path.join(output_path, args.name + "_prompt_" + str(iteration_count) + ".txt"),
        args=args,
        user_guidance=user_guidance,
        extra_summary_sections=extra_summary_sections,
    )

    response = llm.submit_query_to_llm(query)

    response_filename = os.path.join(
        output_path,
        args.name + "_response_" + str(iteration_count) + ".txt"
    )
    retry_prompt_filename = os.path.join(
        output_path,
        args.name + f"_prompt_{iteration_count}_noop_retry.txt"
    )
    rejected_response_filename = os.path.join(
        output_path,
        args.name + f"_response_{iteration_count}_rejected_noop.txt"
    )

    candidate = _extract_candidate_code_for_style(response, _prompt_style(args))
    if candidate and not _has_substantive_code_change(updated_code, candidate):
        retry_query = _build_noop_retry_prompt(query, _prompt_style(args))
        with open(retry_prompt_filename, "w", encoding="utf-8") as file:
            file.write(retry_query)
        with open(rejected_response_filename, "w", encoding="utf-8") as file:
            file.write(response)
        response = llm.submit_query_to_llm(retry_query)
        retry_candidate = _extract_candidate_code_for_style(response, _prompt_style(args))
        if retry_candidate and not _has_substantive_code_change(updated_code, retry_candidate):
            final_rejected_response_filename = os.path.join(
                output_path,
                args.name + f"_response_{iteration_count}_rejected_noop_retry.txt"
            )
            with open(final_rejected_response_filename, "w", encoding="utf-8") as file:
                file.write(response)
            _write_noop_outcome(
                output_path=output_path,
                run_name=args.name,
                iteration=iteration_count,
                reason=(
                    "No substantive optimization was applied after the retry prompt. "
                    "The run was rejected before build validation."
                ),
                response_text=response,
            )
            return query, response, "", []

    with open(response_filename, "w", encoding="utf-8") as file:
        file.write(response)

    extracted_code, explanation_blocks = extract_code_blocks(response)
    if not extracted_code.strip():
        extracted_code = _extract_candidate_code_for_style(response, _prompt_style(args))
        explanation_blocks = []

    return query, response, extracted_code, explanation_blocks


def save_call_for_optimization_state(
    save_dir,
    updated_code,
    args,
    pc_data=None,
    important_counters=None,
    kernel_dict=None,
    extra_summary_sections=None,
):
    """
    Save everything needed to call call_for_optimization() later (for re-optimization).
    Writes a single binary pickle file. Does not save the llm instance; re-create it
    from saved args when loading (e.g. if args.llm in ("gpt-4o", "gpt-5"): llm = ChatGPT_conn(args)).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    state = {
        "updated_code": updated_code,
        "args": args,
        "pc_data": pc_data,
        "important_counters": important_counters,
        "kernel_dict": kernel_dict,
        "extra_summary_sections": extra_summary_sections,
    }
    with open(save_path / "reopt_state.json", "wb") as f:
        json.dump(state, f)

    return str(save_path)


# ------------- Streamlit UI -------------
def streamlit_ui(args):
    pc_data = None
    important_counters = None
    kernel_dict = None
    user_guidance = None
    extra_summary_sections = []

    # output_path = get_output_folder()
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    optimization_stats_file = "optimization_results.csv"

    # Load source file
    with open(args.source_file, "r", encoding="utf-8") as code_file:
        source_code = code_file.read()

    numbered_lines = [
    f"{i+1:4}: {line}" for i, line in enumerate(source_code.splitlines())
    ]
    source_code_with_line_numbers = "\n".join(numbered_lines)
    prompt_source = source_code if _is_finetune_compact_prompt_style(args) else source_code_with_line_numbers
    # print(original_code)
    original_code = preprocess_file(prompt_source, output_file=None, extract_functions=False, keywords=None)
    # print(original_code)
    # return


    # Load PC sampling data
    if args.pc_file is not None:
        pcSummary = PCsummerization(args.pc_file,0.1)
        pcSummary.preprocess()
        pc_data = pcSummary.generate_summary()
        # pc_data = pd.read_csv(args.pc_file)
        # if "# Samples" in pc_data.columns:
        #     pc_data = pc_data[pc_data["# Samples"] > 0]
        # else:
        #     st.warning("Column '# Samples' not found in PC data. Skipping zero-sample filtering.")

    # Load Importance JSON
    important_counters = None
    if args.importance_file:
        # with open(args.importance_file, "r") as f:
        #     json_data = json.load(f)
        # # Replace backend below if needed
        # important_counters = extract_important_counters(json_data, backend="CUDA", threshold=0.1)
        dashingSummary = DashingSummarization(args.importance_file, backend = "CUDA", threshold = 0.1)
        # dashingSummary.preprocess()
        important_counters = dashingSummary.generate_summary()

    # Load and process Roofline CSV
    if args.roofline_file is not None:
        rooflineSummary = RooflineSummarization(
            args.roofline_file,
            similarity_threshold=0.85,
            min_speedup=0.0,
            top_k_per_kernel=None,
        )
        kernel_dict = rooflineSummary.generate_summary()

    if args.summaries_config:
        extra_summary_sections = run_configured_summaries(args.summaries_config)
    # print(extra_summary_sections)

    # return
    # with open("test", 'w', encoding='utf-8') as file:
    #     file.write(original_code)

    user_guidance = args.user_guidance

    query = prepare_sources_multi_gpu_to_submit(
        content=original_code,
        pc_data=pc_data,
        important_counters=important_counters,
        filtered_roofline_data=kernel_dict,
        filename=os.path.join(output_path, args.name + "_prompt_1.txt"),
        args=args,
        user_guidance=user_guidance,
        extra_summary_sections=extra_summary_sections,
    )

    print(f"[llm] prompt saved to {os.path.join(output_path, args.name + '_prompt_1.txt')}")
    print(f"[llm] submitting initial optimization request for {args.name}")
    llm = _select_llm_client(args)
    response = llm.submit_query_to_llm(query)
    response_filename = os.path.join(output_path, args.name + "_response_1.txt")
    retry_prompt_path = os.path.join(output_path, args.name + "_prompt_1_noop_retry.txt")
    rejected_response_path = os.path.join(output_path, args.name + "_response_1_rejected_noop.txt")
    initial_candidate = _extract_candidate_code_for_style(response, _prompt_style(args))
    if initial_candidate and not _has_substantive_code_change(source_code, initial_candidate):
        print("[llm] initial response rejected as no-op; submitting one retry.")
        retry_query = _build_noop_retry_prompt(query, _prompt_style(args))
        with open(retry_prompt_path, "w", encoding="utf-8") as file:
            file.write(retry_query)
        with open(rejected_response_path, "w", encoding="utf-8") as file:
            file.write(response)
        response = llm.submit_query_to_llm(retry_query)
        retry_candidate = _extract_candidate_code_for_style(response, _prompt_style(args))
        if retry_candidate and not _has_substantive_code_change(source_code, retry_candidate):
            final_rejected_response_path = os.path.join(
                output_path, args.name + "_response_1_rejected_noop_retry.txt"
            )
            with open(final_rejected_response_path, "w", encoding="utf-8") as file:
                file.write(response)
            outcome_path = _write_noop_outcome(
                output_path=output_path,
                run_name=args.name,
                iteration=1,
                reason=(
                    "No substantive optimization was applied after the retry prompt. "
                    "The run was rejected before build validation."
                ),
                response_text=response,
            )
            print(f"[llm] retry response also rejected as no-op. Outcome saved to {outcome_path}")
            return

    with open(response_filename, "w", encoding="utf-8") as file:
        file.write(response)

    print(f"[llm] response saved to {response_filename}")

    updated_code = _extract_candidate_code_for_style(response, _prompt_style(args))
    if not updated_code:
        print("[error] Could not extract code from model response.")
        return

    optimized_code_path = os.path.join(output_path, f"opt_iter_1_{Path(args.source_file).name}")
    with open(optimized_code_path, "w", encoding="utf-8") as file:
        file.write(updated_code)

    build_dir_value, build_script = _resolve_build_dir_and_script(
        getattr(args, "build_dir", None),
        getattr(args, "build_script", None),
    )
    if not build_dir_value:
        print("[info] Build-validation loop skipped (set --build_dir or provide a path in --build_script to enable).")
        return

    build_dir = Path(build_dir_value)
    build_dir.mkdir(parents=True, exist_ok=True)

    build_write_mode = str(getattr(args, "build_write_mode", "overwrite") or "overwrite").strip().lower()
    build_source_path = _resolve_build_target_path(
        build_dir=build_dir,
        source_file=args.source_file,
        build_target_source=getattr(args, "build_target_source", None),
    )
    build_source_path.parent.mkdir(parents=True, exist_ok=True)

    build_target_template = None
    if build_write_mode == "merge-kernels":
        if not build_source_path.exists():
            print(f"[build] merge-kernels target does not exist: {build_source_path}")
            return
        build_target_template = _read_file_text(build_source_path)
        if build_target_template.startswith("[error log not found at:"):
            print(f"[build] failed to read merge-kernels target: {build_source_path}")
            return
        print(f"[build] merge-kernels target: {build_source_path}")
    else:
        print(f"[build] overwrite target: {build_source_path}")

    max_iters = max(1, int(args.build_fix_iterations))
    for iteration in range(1, max_iters + 1):
        print(f"[build] starting iteration {iteration}/{max_iters}")
        if build_write_mode == "merge-kernels":
            try:
                build_file_text, replaced_functions = _merge_updated_code_into_build_target(
                    updated_code=updated_code,
                    build_target_text=build_target_template,
                )
                print(
                    "[build] iteration "
                    f"{iteration}: merged optimized kernels into target: {', '.join(replaced_functions)}"
                )
            except Exception as exc:
                print(f"[build] iteration {iteration}: failed to merge optimized kernels: {exc}")
                break
        else:
            build_file_text = updated_code

        with open(build_source_path, "w", encoding="utf-8") as file:
            file.write(build_file_text)

        merged_code_path = os.path.join(output_path, f"build_iter_{iteration}_{build_source_path.name}")
        with open(merged_code_path, "w", encoding="utf-8") as file:
            file.write(build_file_text)

        print(f"[build] iteration {iteration}: wrote build source to {build_source_path}")
        print(f"[build] iteration {iteration}: saved merged snapshot to {merged_code_path}")

        rc = run_script_in_build_dir_separate_logs(
            build_dir=str(build_dir),
            script_name=build_script,
            out_file=args.build_out_file,
            err_file=args.build_err_file,
        )

        out_path = build_dir / args.build_out_file
        err_path = build_dir / args.build_err_file
        stdout_text = _read_file_text(out_path)
        stderr_text = _read_file_text(err_path)
        build_out_copy_path, build_err_copy_path = _write_build_logs_to_dir(
            target_dir=str(build_dir),
            run_name=args.name,
            iteration=iteration,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        out_copy_path, err_copy_path = _write_build_logs_to_output(
            output_path=output_path,
            run_name=args.name,
            iteration=iteration,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        print(f"[build] iteration {iteration}: archived stdout to {build_out_copy_path}")
        print(f"[build] iteration {iteration}: archived stderr to {build_err_copy_path}")
        print(f"[build] iteration {iteration}: copied stdout to {out_copy_path}")
        print(f"[build] iteration {iteration}: copied stderr to {err_copy_path}")
        has_errors = _has_build_errors(rc, stderr_text)

        if not has_errors:
            print(f"[build] iteration {iteration}: success.")
            avg_runtime_ok = _run_average_runtime(
                build_dir=build_dir,
                output_path=output_path,
                args=args,
                iteration=iteration,
            )
            if avg_runtime_ok:
                print("[runtime] average_runtime.sh succeeded; skipping LLM runtime extractor.")
            else:
                parsed_runtime_ok = _record_runtime_from_build_stdout(
                    stdout_text=stdout_text,
                    build_dir=build_dir,
                    output_path=output_path,
                    args=args,
                    iteration=iteration,
                )
                if parsed_runtime_ok:
                    print("[runtime] parsed runtime directly from build stdout; skipping LLM runtime extractor.")
                    break
                try:
                    runtime_script_path, was_generated = _ensure_runtime_extractor_script(
                        output_path=output_path,
                        app_name=args.app_name,
                        llm=llm,
                        build_stdout_text=stdout_text,
                    )
                    if was_generated:
                        print(f"[runtime] generated extractor script: {runtime_script_path}")
                    else:
                        print(f"[runtime] reusing extractor script: {runtime_script_path}")

                    runtime_ok, runtime_out_path, runtime_err_path = _run_runtime_extractor_script(
                        script_path=runtime_script_path,
                        build_stdout_path=str(build_out_copy_path),
                        output_path=output_path,
                        run_name=args.name,
                        iteration=iteration,
                    )
                    if runtime_ok:
                        print(f"[runtime] extracted runtime written to {runtime_out_path}")
                    else:
                        print(f"[runtime] extractor failed. See {runtime_err_path}")
                except Exception as exc:
                    print(f"[runtime] failed to generate or run runtime extractor: {exc}")
                print("[runtime] average_runtime.sh did not produce a CSV-updatable runtime value.")
            break

        error_summary = summarize_build_errors(
            stderr_text,
            max_lines=args.build_error_summary_lines,
            max_chars=args.build_error_summary_chars,
        )
        summary_path = os.path.join(output_path, f"{args.name}_build_errors_iter_{iteration}.txt")
        with open(summary_path, "w", encoding="utf-8") as ef:
            ef.write(error_summary)

        print(f"[build] iteration {iteration}: errors found. Summary saved to {summary_path}")

        if iteration >= max_iters:
            print(f"[build] reached max fix iterations ({max_iters}).")
            break

        repair_query = prepare_second_prompt_for_style(
            original_source=original_code,
            optimized_code=updated_code,
            build_errors=error_summary,
            filename=os.path.join(output_path, args.name + f"_repair_prompt_{iteration+1}.txt"),
            args=args,
        )
        print(f"[llm] submitting repair request for iteration {iteration + 1}")
        repair_response = llm.submit_query_to_llm(repair_query)
        repair_response_filename = os.path.join(output_path, args.name + f"_response_{iteration+1}.txt")
        repair_retry_prompt_filename = os.path.join(
            output_path, args.name + f"_repair_prompt_{iteration+1}_noop_retry.txt"
        )
        repair_rejected_response_filename = os.path.join(
            output_path, args.name + f"_response_{iteration+1}_rejected_noop.txt"
        )
        repair_candidate = _extract_candidate_code_for_style(repair_response, _prompt_style(args))
        if repair_candidate and not _has_substantive_code_change(updated_code, repair_candidate):
            print(f"[llm] repair response for iteration {iteration + 1} rejected as no-op; submitting one retry.")
            repair_retry_query = _build_noop_retry_prompt(repair_query, _prompt_style(args))
            with open(repair_retry_prompt_filename, "w", encoding="utf-8") as file:
                file.write(repair_retry_query)
            with open(repair_rejected_response_filename, "w", encoding="utf-8") as file:
                file.write(repair_response)
            repair_response = llm.submit_query_to_llm(repair_retry_query)
            repair_retry_candidate = _extract_candidate_code_for_style(repair_response, _prompt_style(args))
            if repair_retry_candidate and not _has_substantive_code_change(updated_code, repair_retry_candidate):
                final_rejected_response_filename = os.path.join(
                    output_path, args.name + f"_response_{iteration+1}_rejected_noop_retry.txt"
                )
                with open(final_rejected_response_filename, "w", encoding="utf-8") as file:
                    file.write(repair_response)
                outcome_path = _write_noop_outcome(
                    output_path=output_path,
                    run_name=args.name,
                    iteration=iteration + 1,
                    reason=(
                        "No substantive repair optimization was applied after the retry prompt. "
                        "The build-fix loop stopped without accepting a formatting-only rewrite."
                    ),
                    response_text=repair_response,
                )
                print(f"[llm] repair retry response also rejected as no-op. Outcome saved to {outcome_path}")
                break
        with open(repair_response_filename, "w", encoding="utf-8") as file:
            file.write(repair_response)
        print(f"[llm] repair response saved to {repair_response_filename}")

        candidate = _extract_candidate_code_for_style(repair_response, _prompt_style(args))
        if not candidate:
            print(f"[build] iteration {iteration}: failed to extract repaired code. Stopping.")
            break

        updated_code = candidate
        optimized_code_path = os.path.join(
            output_path, f"opt_iter_{iteration+1}_{Path(args.source_file).name}"
        )
        with open(optimized_code_path, "w", encoding="utf-8") as file:
            file.write(updated_code)

    return

    if st.sidebar.button("Run Optimization"):

        if not code_file:
            st.error("Please upload at least the code file.")
            return

        # Load and decode CUDA code content
        source_code = code_file.getvalue().decode('utf-8')
        original_code = preprocess_file(source_code, output_file = None, extract_functions=False, keywords=None)


        # Initialize variables explicitly
        pc_data = None
        important_counters = None
        kernel_dict = None

        # Load and process PC sampling data exactly as original logic
        if pc_file:
            pc_data = pd.read_csv(pc_file)
            if "# Samples" in pc_data.columns:
                pc_data = pc_data[pc_data["# Samples"] > 0]
            else:
                st.warning("Column '# Samples' not found in PC data. Skipping zero-sample filtering.")

            large_stalls = find_large_stalls(pc_data, threshold=0.1)

        # Load and process Importance JSON exactly as original logic
        if json_file:
            json_data = json.load(json_file)
            important_counters = extract_important_counters(json_data, backend, threshold=0.1)

        # Load and process Roofline CSV exactly as original logic
        if roofline_file:
            roofline_data = pd.read_csv(roofline_file)
            nadropped_roofline_data = roofline_data.dropna(subset=["Rule Name", "Rule Description", "Kernel Name"])
            nadropped_roofline_data["Estimated Speedup"] = pd.to_numeric(
                nadropped_roofline_data["Estimated Speedup"], errors='coerce').fillna(0)

            kernel_dict = {}
            for kernel, group in nadropped_roofline_data.groupby("Kernel Name"):
                sorted_rules = sorted(
                    set(group.sort_values("Estimated Speedup", ascending=False)["Rule Description"].tolist()),
                    key=lambda x: group.loc[group["Rule Description"] == x, "Estimated Speedup"].max(),
                    reverse=True
                )
                # if kernel.__contains__('addsgd4_SM'):
                kernel_dict[kernel] = remove_similar_entries(sorted_rules)

        # Construct the prompt exactly as your original logic intended
        query = prepare_sources_to_submit(
        content=original_code,
        pc_data=pc_data,
        important_counters=important_counters,
        filtered_roofline_data=kernel_dict,
        filename=os.path.join(output_path,"prompt.txt"),
        args=args  # Pass args explicitly!
        )

        # Display widgets clearly separated
        st.subheader("Generated Prompt")
        st.text_area("Prompt", query, height=250)

        return

        # Submit query and obtain response from ChatGPT
        with st.spinner("Querying ChatGPT for optimizations..."):
            args.logprobs_file = os.path.join(output_path, 'logprobes.txt')
            response = submit_query(query, args)  # Pass args explicitly here as well
            with open(os.path.join(output_path,'response.txt'), 'w', encoding='utf-8') as file:
                file.write(response)

        # DEBUG AID: Print the tail of the response to inspect structure
        print("==== RAW RESPONSE TAIL ====")
        print(response[-500:])  # Adjust length as needed

        # Parse ChatGPT response explicitly into optimized code and explanation
        #optimized_code, explanation = parse_chatgpt_response(response)
        optimized_code = extract_code_block(response)
        code_output_file_name = os.path.join(output_path,f"opt_{code_file.name}")
        #print(optimized_code)

        try:
            explanation = parse_llm_response_safe(response)
            # Simply unpack like this:
            optimizations, suggestions = explanation

        except ValueError as e:
            st.error("Could not parse the optimizations list. Showing raw model response for debugging.")
            st.code(response)
            explanation = []

        ############## Code display-related block
        sanitized_code = sanitize_text(optimized_code).replace('\r\n', '\n')
        # Write sanitized_code to the specified output file -- for provenance
        with open(code_output_file_name, "w", encoding="utf-8") as file:
            file.write(sanitized_code)

        ######## The line numbers between opt and unopt may not align. Need some pre-processing to handle that.
        # Detect starting line number automatically
        starting_line_number = detect_starting_line_number_diff(original_code, optimized_code)

        if starting_line_number != -1:
            st.success(f"Optimized snippet aligned at original line: {starting_line_number}")

            # Extract lines to highlight from optimizations
            highlight_lines = extract_highlight_lines_from_optimizations(optimizations)

            # FIX: Now render optimized code with correct highlighting. Line numbers are not aligning with the original and optimized. Need to debug more.
#            highlighted_code = render_code_with_highlights(
#                optimized_code, optimizations, 1#starting_line_number
#            )
            
            # The following only renders line numbers, not highlight though.
            highlighted_code = render_code_with_line_numbers(optimized_code)
            # Show code in a fancy way
            #highlighted_code = render_code_with_highlights(sanitized_code, optimizations)
            ## Show code with highlights
            st.subheader("Optimized Code")
            #st.code(sanitized_code, language='cuda')
            st.markdown(highlighted_code, unsafe_allow_html=True)

        else:
            st.error("Failed to align optimized snippet with original source code.")
       
        
        st.subheader("Optimization Explanation")
        st.markdown(format_optimizations_markdown(explanation))

        # Use the visualize function previously provided
        #visualize_code_with_optimizations(optimized_code, explanation)
        # Usage in Streamlit
        # Slider to set top N tokens for logparse interpretation

        st.sidebar.title("LogProbs Analysis")



        # This value `top_tokens` can be used in your logprobs rendering widget logic
        st.sidebar.markdown(f"Top-{top_n} influential tokens will be highlighted in interpretation views.")


        # Visualizing Logprobe Tokens
        st.subheader("Token Interpretation from Logprobes")
        belief_dict, word_freq_dict, narrative = process_logprobes(os.path.join(output_path,"logprobes.txt"), os.path.join(output_path,"prompt.txt"))

        #show_belief_bar_chart(belief_dict, top_n=10)
        #show_belief_histogram(belief_dict, bins=20)
        #show_frequency_bar_chart(word_freq_dict, top_n=20)
        #show_belief_graph_interactive(belief_dict)

        col1, col2, col3 = st.columns(3)
        with col1:
            plot_belief_flame(belief_dict, os.path.join(output_path,"belief_flame.pdf"), top_n)
        with col2:
            show_belief_histogram(belief_dict, top_n, os.path.join(output_path,"belief_hist.pdf"))
        with col3:
            show_frequency_bar_chart(word_freq_dict, top_n, os.path.join(output_path,"word_freq.pdf"))


#        col1, col2 = st.columns(2)
#        with col1:
##            show_belief_bar_chart(belief_dict, top_n, os.path.join(output_path,"belief_bar.pdf"))
#            plot_belief_flame(belief_dict, os.path.join(output_path,"belief_flame.pdf"), top_n)
#
#        with col2:
#            show_belief_histogram(belief_dict, top_n, os.path.join(output_path,"belief_hist.pdf"))
#
#        col3, col4 = st.columns(2)
#        with col3:
#            show_frequency_bar_chart(word_freq_dict, top_n, os.path.join(output_path,"word_freq.pdf"))
#        with col4:

#        col5, col6 = st.columns(2)
#        with col5:
#            show_narrative(narrative)
#        with col6:
#            show_belief_graph_interactive(belief_dict)
#
#        col7, col8, col9 = st.columns(3)
#        with col7:
#            spiral_reasoning_plot(belief_dict, os.path.join(output_path,"belief_spiral.pdf"), top_n)
#        with col8:
#            word_cloud_with_gravity(belief_dict, os.path.join(output_path,"belief_gravity.pdf"), top_n)
#        with col9:
#            reasoning_lens_plot(belief_dict, os.path.join(output_path,"belief_lens.pdf"), top_n)
            
        ###### Write a stats file
        process_and_save_results(output_path, pc_file is not None, json_file is not None, roofline_file is not None, optimizations)



# ------------- Main Function -------------
def main():
    parser = argparse.ArgumentParser(description="OPAL")

    parser.add_argument(
        "--source_file", type=str,
        default="../input/accuracy/main.cu",
        # default=None,
        # required=True,
        help="Path to source code file (.cu, .cpp, .py)"
    )
    parser.add_argument(
        "--pc_file", type=str,
        # default="../input/accuracy/default-nrows_8192-ndims_5000-topk_10-rep_100/accuracy_pcsamp.csv",
        default=None,
        help="Path to PC sampling CSV file"
    )
    parser.add_argument(
        "--importance_file", type=str,
        # default="../input/accuracy/default-nrows_8192-ndims_5000-topk_10-rep_100/errors_accuracy.json",
        default=None,
        help="Path to importance JSON file"
    )
    parser.add_argument(
        "--roofline_file", type=str,
        # default="../input/accuracy/default-nrows_8192-ndims_5000-topk_10-rep_100/accuracy_roofline.csv",
        default=None,
        help="Path to Roofline CSV file"
    )
    parser.add_argument(
        "--logprobs", action="store_true",
        help="Enable logprobs"
    )
    parser.add_argument(
        "--logprobs_file", type=str,
        default="logprobs.txt",
        help="File for logprobs output"
    )
    parser.add_argument(
        "--name", type=str,
        default="result",
        help="Name of the output files"
    )
    parser.add_argument(
        "--llm", type=str,
        default="gpt-4o",
        help="Name of the LLM"
    )
    parser.add_argument(
        "--llm_backend", type=str,
        default="auto",
        choices=["auto", "openai", "gemini", "ollama", "vllm", "zkllm"],
        help="Backend/provider selection. Use 'zkllm' for LLaMA-2 with ZK proof generation."
    )
    parser.add_argument(
        "--vllm_model", type=str,
        default=None,
        help="vLLM model name served by the local endpoint (falls back to --llm if unset)."
    )
    parser.add_argument(
        "--vllm_host", type=str,
        default="localhost",
        help="vLLM host."
    )
    parser.add_argument(
        "--vllm_port", type=int,
        default=8000,
        help="vLLM port."
    )
    parser.add_argument(
        "--vllm_api_key", type=str,
        default="",
        help="Optional vLLM API key for OpenAI-compatible auth."
    )
    parser.add_argument(
        "--prompt_style", type=str,
        default="codex-default",
        choices=["codex-default", "finetune-compact"],
        help="Prompt/output contract to use. 'finetune-compact' matches the compact SFT validation format."
    )
    parser.add_argument(
        "--vllm_timeout_sec", type=float,
        default=180.0,
        help="HTTP timeout in seconds for vLLM requests."
    )
    parser.add_argument(
        "--vllm_max_tokens", type=int,
        default=None,
        help="Maximum output tokens requested from vLLM. Defaults to 2048 for finetune-compact and 16384 otherwise."
    )
    parser.add_argument(
        "--vllm_temperature", type=float,
        default=None,
        help="Sampling temperature for vLLM. Defaults to 0.0 for finetune-compact and 0.15 otherwise."
    )
    parser.add_argument(
        "--zkllm_model_size", type=int,
        default=7, choices=[7, 13],
        help="LLaMA-2 model size for zkLLM backend (7 or 13 billion parameters)."
    )
    parser.add_argument(
        "--zkllm_seq_len", type=int,
        default=2048,
        help="Sequence length for zkLLM proof (must match commit-param setup)."
    )
    parser.add_argument(
        "--zkllm_max_new_tokens", type=int,
        default=512,
        help="Maximum new tokens to generate with zkLLM backend."
    )
    parser.add_argument(
        "--zkllm_dir", type=str,
        default="zkllm",
        help="Path to the zkllm directory (containing model-storage and zkllm-workdir)."
    )
    parser.add_argument(
        "--zkllm_proof_dir", type=str,
        default="zkllm_proofs",
        help="Directory where zkLLM proof JSON files are saved."
    )
    parser.add_argument(
        "--build_dir", "-B", dest="build_dir", type=str, default=None, metavar="DIR",
        help="Optional directory to build the code in. If set, it will be created if missing."
    )
    parser.add_argument(
        "--build_script", type=str, default=None,
        help="Build script filename inside build_dir (e.g., run.sh). Enables auto build-fix loop when set with build_dir."
    )
    parser.add_argument(
        "--build_write_mode", type=str, default="overwrite",
        choices=["overwrite", "merge-kernels"],
        help="How optimized code is written into the build tree. Use 'merge-kernels' to replace only matching __global__ kernel definitions inside --build_target_source."
    )
    parser.add_argument(
        "--build_target_source", type=str, default=None,
        help="Target source file to update inside build_dir (or absolute path). With --build_write_mode merge-kernels, matching optimized kernels are merged into this file instead of overwriting the optimized source file."
    )
    parser.add_argument(
        "--build_fix_iterations", type=int, default=3,
        help="Maximum number of build-fix iterations after the first optimization response."
    )
    parser.add_argument(
        "--build_out_file", type=str, default="build.out",
        help="Filename to store build stdout inside build_dir."
    )
    parser.add_argument(
        "--build_err_file", type=str, default="build.err",
        help="Filename to store build stderr inside build_dir."
    )
    parser.add_argument(
        "--build_error_summary_lines", type=int, default=80,
        help="Maximum number of lines from stderr to include in repair prompt."
    )
    parser.add_argument(
        "--build_error_summary_chars", type=int, default=8000,
        help="Maximum number of characters from summarized stderr to include in repair prompt."
    )
    parser.add_argument(
        "--output_dir", "-o", dest="output_dir", type=str, default=None, metavar="DIR",
        help="Optional directory to build the code in. If set, it will be created if missing."
    )
    parser.add_argument(
        "--output_file", type=str,
        default="results.csv",
        help="Path to the CSV file where average runtimes will be stored"
    )
    parser.add_argument(
        "--app_name", type=str,
        required=True,
        help="Application name to be used as the row label in the CSV"
    )
    parser.add_argument(
        "--gemini_api_key", type=str,
        default="",
        help="gemini api key"
    )
    parser.add_argument(
        "--gcp_project", type=str,
        default="moonlit-album-474714-n2",
        help="gemini api key"
    )
    parser.add_argument(
        "--gcp_location", type=str,
        default="us-central1",
        help="gemini api key"
    )
    parser.add_argument(
        "--user_guidance", type=str,
        default=None,
        help="Custom guidance from user."
    )
    parser.add_argument(
        "--summaries_config", type=str,
        default=None,
        help="Path to JSON config describing extra summary classes and input files."
    )
    parser.add_argument(
        "--formal_reasoning_mode", type=str,
        default="enabled",
        choices=["enabled", "disabled"],
        help="Controls whether the optimization prompt requires mathematical preconditions, postconditions, loop invariants, and proof obligations."
    )

    # Load YAML config and inject missing values into sys.argv so required args are satisfied
    import sys as _sys
    _argv = _sys.argv[1:]
    config_path = None
    for _i, _a in enumerate(_argv):
        if _a == "--config" and _i + 1 < len(_argv):
            config_path = _argv[_i + 1]
            break
    if config_path:
        import yaml as _yaml
        with open(config_path) as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Collect flags already on the command line
        _existing_flags = {a.lstrip("-").replace("-", "_") for a in _argv if a.startswith("--")}
        for _k, _v in _cfg.items():
            if _k not in _existing_flags and _v is not None:
                _sys.argv.extend([f"--{_k}", str(_v)])
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args = parser.parse_args()
    streamlit_ui(args)


# ------------- Entrypoint -------------
if __name__ == "__main__":
    main()
