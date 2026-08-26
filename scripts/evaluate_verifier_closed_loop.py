from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import pandas as pd
import torch

from evaluate_teacher_forced import (
    SparseTeacherController,
    build_policy,
    fit_tokens,
    load_prompts,
    patch_model,
)
from temporal_dsa.v2_collect import MODEL_REVISION, fixed_device_map


@torch.no_grad()
def greedy_tokens(
    model: torch.nn.Module,
    controller: SparseTeacherController,
    mode: str,
    policies: dict[int, Any],
    token_ids: list[int],
    steps: int,
    verifier_spec: dict[str, Any] | None = None,
) -> tuple[list[int], float]:
    controller.reset(mode, policies, verifier_spec)
    device = model.model.embed_tokens.weight.device
    prefill = torch.tensor(token_ids, device=device, dtype=torch.long).view(1, -1)
    prefill_output = model.model(input_ids=prefill, use_cache=True, return_dict=True)
    past = prefill_output.past_key_values
    logits = model.lm_head(prefill_output.last_hidden_state[:, -1:])[:, -1]
    generated: list[int] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(steps):
        token = logits.argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        if step + 1 == steps:
            break
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        logits = output.logits[:, -1]
    torch.cuda.synchronize()
    return generated, time.perf_counter() - started


def niah_tokens(tokenizer: Any, context_length: int) -> list[int]:
    intro = tokenizer(
        "A fact is hidden in the following notes. Remember it exactly.\n",
        add_special_tokens=True,
    )["input_ids"]
    needle = tokenizer(
        "\nIMPORTANT FACT: The pass key is KOREA1582.\n",
        add_special_tokens=False,
    )["input_ids"]
    question = tokenizer(
        "\nQuestion: What is the pass key? Answer with only the pass key.\nAnswer:",
        add_special_tokens=False,
    )["input_ids"]
    filler = tokenizer(
        " The archive records ordinary weather, books, roads, and garden notes.",
        add_special_tokens=False,
    )["input_ids"]
    filler_needed = context_length - len(intro) - len(needle) - len(question)
    if filler_needed <= 0:
        raise ValueError("context too short for NIAH prompt")
    repeated = (filler * ((filler_needed + len(filler) - 1) // len(filler)))[:filler_needed]
    midpoint = len(repeated) // 2
    tokens = intro + repeated[:midpoint] + needle + repeated[midpoint:] + question
    if len(tokens) != context_length:
        raise RuntimeError("NIAH construction did not match requested context")
    return tokens


def first_divergence(baseline: list[int], approximate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(baseline, approximate), start=1):
        if left != right:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop greedy verifier validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--verifier-configs", type=Path, required=True)
    parser.add_argument("--verifier-name", required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--code-prompt-id", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--role", default="Aggressive")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache

    if not hasattr(Cache, "get_usable_length"):
        def get_usable_length(self: Any, new_seq_length: int, layer_idx: int = 0) -> int:
            del new_seq_length
            return int(self.get_seq_length(layer_idx))
        Cache.get_usable_length = get_usable_length  # type: ignore[attr-defined]
    if torch.cuda.device_count() != 2:
        raise RuntimeError("expected CUDA_VISIBLE_DEVICES=0,1")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in name for name in gpu_names):
        raise RuntimeError(f"L40 guard rejected {gpu_names}")

    selection = json.loads(args.selection.read_text())
    policy_spec = next(
        row for row in selection["policies"] if row["selection_role"] == args.role
    )
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    policies = {layer: build_policy(policy_spec, layer, scales) for layer in args.layers}
    verifier_payload = json.loads(args.verifier_configs.read_text())
    verifier_spec = next(
        row for row in verifier_payload["configs"] if row["name"] == args.verifier_name
    )

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=fixed_device_map(config.num_hidden_layers),
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    first_attention = model.model.layers[0].self_attn
    apply_rotary = first_attention.__class__.forward.__globals__["apply_rotary_pos_emb"]
    controller = SparseTeacherController(
        model,
        apply_rotary,
        args.layers,
        args.checkpoints,
        int(selection["k"]),
        int(selection["block_size"]),
    )
    if patch_model(model, controller) != config.num_hidden_layers:
        raise RuntimeError("failed to patch all attention modules")

    prompts = load_prompts(args.prompts)
    code_prompt = prompts[args.code_prompt_id]
    code_raw = tokenizer(code_prompt["text"], add_special_tokens=True)["input_ids"]
    rows: list[dict[str, Any]] = []
    for context_length in args.contexts:
        samples = [
            ("ruler_niah_small", "niah", niah_tokens(tokenizer, context_length)),
            (
                "long_code_completion",
                args.code_prompt_id,
                fit_tokens(code_raw, context_length),
            ),
        ]
        for benchmark, prompt_id, tokens in samples:
            baseline, baseline_seconds = greedy_tokens(
                model, controller, "baseline", {}, tokens, args.steps
            )
            approximate, approximate_seconds = greedy_tokens(
                model,
                controller,
                "verifier",
                policies,
                tokens,
                args.steps,
                verifier_spec,
            )
            agreement = sum(a == b for a, b in zip(baseline, approximate)) / args.steps
            baseline_text = tokenizer.decode(baseline, skip_special_tokens=True)
            approximate_text = tokenizer.decode(approximate, skip_special_tokens=True)
            rows.append(
                {
                    "benchmark": benchmark,
                    "prompt_id": prompt_id,
                    "context_length": context_length,
                    "steps": args.steps,
                    "policy": policy_spec["config_id"],
                    "selection_role": args.role,
                    "verifier": verifier_spec["name"],
                    "first_divergence_step": first_divergence(baseline, approximate),
                    "generated_token_agreement": agreement,
                    "baseline_task_success": (
                        "KOREA1582" in baseline_text if benchmark == "ruler_niah_small" else None
                    ),
                    "approx_task_success": (
                        "KOREA1582" in approximate_text if benchmark == "ruler_niah_small" else None
                    ),
                    "baseline_decode_seconds": baseline_seconds,
                    "approx_decode_seconds": approximate_seconds,
                    "baseline_text": baseline_text,
                    "approx_text": approximate_text,
                }
            )
            print(f"completed {benchmark} context={context_length}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "closed_loop_quality.csv", index=False)
    summary = {
        "model_revision": args.revision,
        "gpus": gpu_names,
        "python": platform.python_version(),
        "policy": policy_spec,
        "verifier": verifier_spec,
        "rows": len(frame),
        "generated_token_agreement_mean": float(frame.generated_token_agreement.mean()),
        "first_divergence_min": (
            None
            if frame.first_divergence_step.dropna().empty
            else int(frame.first_divergence_step.dropna().min())
        ),
        "niah_baseline_success_rate": float(
            frame.loc[frame.benchmark == "ruler_niah_small", "baseline_task_success"].mean()
        ),
        "niah_approx_success_rate": float(
            frame.loc[frame.benchmark == "ruler_niah_small", "approx_task_success"].mean()
        ),
    }
    (args.output / "closed_loop_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
