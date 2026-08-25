from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import platform
import time
import types
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from temporal_dsa.approx import ApproxPolicy, ApproxState, initialize_state, replay_step
from temporal_dsa.metrics import stable_topk
from temporal_dsa.sidecar import LightningIndexerSidecar
from temporal_dsa.v2_collect import MODEL_REVISION, fixed_device_map


def family_policy(family: str, *, name: str, gamma: float) -> ApproxPolicy:
    common: dict[str, Any] = {"name": name, "gamma": gamma}
    if family == "static":
        return ApproxPolicy(**common)
    if family == "dynamic_address":
        return ApproxPolicy(**common, dynamic_threshold=True)
    if family == "dynamic_previous_max":
        return ApproxPolicy(**common, dynamic_threshold=True, order="previous_max")
    if family in {"dynamic_bucket8", "dynamic_bucket16"}:
        return ApproxPolicy(
            **common, dynamic_threshold=True, order=family.removeprefix("dynamic_")
        )
    if family in {"streak2_bucket8", "streak4_bucket8"}:
        streak = int(family.removeprefix("streak").split("_", 1)[0])
        return ApproxPolicy(
            **common, dynamic_threshold=True, order="bucket8", cold_streak=streak
        )
    if family == "ema_bucket8_age8":
        return ApproxPolicy(
            **common,
            dynamic_threshold=True,
            order="bucket8",
            risk_model="ema",
            ema_alpha=0.9,
            volatility_lambda=2.0,
            age_cap=8,
        )
    raise ValueError(f"unknown family: {family}")


def build_policy(spec: dict[str, Any], layer: int, scales: dict[int, float]) -> ApproxPolicy:
    policy = family_policy(
        str(spec["family"]),
        name=str(spec["config_id"]),
        gamma=float(spec["multiplier"]) * scales[layer],
    )
    return replace(policy, **spec.get("overrides", {}))


@dataclass
class RunOutput:
    logits: torch.Tensor
    ground_truth: torch.Tensor
    final_hidden: torch.Tensor
    selected_hidden: torch.Tensor
    elapsed_seconds: float


class SparseTeacherController:
    """Reference decode-only sparse MLA with a frozen research sidecar."""

    def __init__(
        self,
        model: torch.nn.Module,
        apply_rotary: Any,
        selected_layers: list[int],
        checkpoint_root: Path,
        k: int,
        block_size: int,
    ) -> None:
        from safetensors.torch import load_file

        self.apply_rotary = apply_rotary
        self.selected_layers = set(selected_layers)
        self.k = k
        self.block_size = block_size
        self.indexers: dict[int, LightningIndexerSidecar] = {}
        for layer in selected_layers:
            device = model.model.layers[layer].self_attn.o_proj.weight.device
            indexer = LightningIndexerSidecar().to(device=device, dtype=torch.float32)
            checkpoint = checkpoint_root / f"layer_{layer:02d}.safetensors"
            indexer.load_state_dict(load_file(checkpoint, device=str(device)))
            self.indexers[layer] = indexer.eval()
        self.mode = "dense"
        self.policy_by_layer: dict[int, ApproxPolicy] = {}
        self.reset("dense", {})

    def reset(self, mode: str, policy_by_layer: dict[int, ApproxPolicy]) -> None:
        if mode not in {"dense", "baseline", "approx"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.policy_by_layer = policy_by_layer
        self.encoded_keys: dict[int, torch.Tensor] = {}
        self.policy_state: dict[int, ApproxState] = {}
        self.policy_step: dict[int, int] = {layer: 0 for layer in self.selected_layers}

    @torch.no_grad()
    def _selection(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> np.ndarray:
        indexer = self.indexers[layer]
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            current_keys = indexer.encode_keys(hidden_states, position_ids)
            if layer not in self.encoded_keys:
                self.encoded_keys[layer] = current_keys
            else:
                self.encoded_keys[layer] = torch.cat(
                    (self.encoded_keys[layer], current_keys), dim=1
                )
            queries, weights = indexer.encode_queries(hidden_states[:, -1:], position_ids[:, -1:])
            scores = indexer.score_encoded(queries, weights, self.encoded_keys[layer])
        score_vector = scores[0, 0].float().cpu().numpy()
        if score_vector.size < self.k:
            return np.arange(score_vector.size, dtype=np.int64)
        if self.mode == "baseline":
            return stable_topk(score_vector, self.k)
        if layer not in self.policy_state:
            state = initialize_state(score_vector, k=self.k, block_size=self.block_size)
            self.policy_state[layer] = state
            return state.topk
        step = self.policy_step[layer] + 1
        state, _, detail = replay_step(
            self.policy_state[layer],
            score_vector,
            policy=self.policy_by_layer[layer],
            k=self.k,
            block_size=self.block_size,
            step=step,
            history_mode="own",
            previous_length=score_vector.size - 1,
        )
        self.policy_state[layer] = state
        self.policy_step[layer] = step
        return detail["approximate"]

    @torch.no_grad()
    def _encode_prefill(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        indexer = self.indexers[layer]
        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.bfloat16):
            self.encoded_keys[layer] = indexer.encode_keys(hidden_states, position_ids)

    def forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        del use_cache, kwargs
        bsz, q_len, _ = hidden_states.shape
        if bsz != 1 or position_ids is None:
            raise ValueError("reference evaluator requires batch=1 and position_ids")
        layer = int(module.layer_idx)
        selected_layer = layer in self.selected_layers
        if selected_layer and self.mode != "dense" and q_len > 1:
            self._encode_prefill(layer, hidden_states, position_ids)
            selected_ids = None
        elif selected_layer and self.mode != "dense":
            selected_ids = self._selection(layer, hidden_states, position_ids)
        else:
            selected_ids = None

        if module.q_lora_rank is None:
            q = module.q_proj(hidden_states)
        else:
            q = module.q_b_proj(module.q_a_layernorm(module.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, module.num_heads, module.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1
        )
        compressed_kv = module.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, module.qk_rope_head_dim).transpose(1, 2)
        kv = (
            module.kv_b_proj(module.kv_a_layernorm(compressed_kv))
            .view(
                bsz,
                q_len,
                module.num_heads,
                module.qk_nope_head_dim + module.v_head_dim,
            )
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            kv, [module.qk_nope_head_dim, module.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, module.layer_idx)
        cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = self.apply_rotary(q_pe, k_pe, cos, sin, position_ids)
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat(
            (k_nope, k_pe.expand(-1, module.num_heads, -1, -1)), dim=-1
        )
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                module.layer_idx,
                {"sin": sin, "cos": cos},
            )

        if q_len > 1:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,
                scale=float(module.softmax_scale),
            )
            attention_weights = None
        else:
            if selected_ids is not None:
                ids = torch.as_tensor(selected_ids, device=key_states.device, dtype=torch.long)
                selected_keys = key_states[:, :, ids]
                selected_values = value_states[:, :, ids]
                logits = torch.matmul(query_states, selected_keys.transpose(2, 3))
                logits = logits * module.softmax_scale
                if attention_mask is not None:
                    logits = logits + attention_mask[..., ids]
            else:
                selected_values = value_states
                logits = torch.matmul(query_states, key_states.transpose(2, 3))
                logits = logits * module.softmax_scale
                if attention_mask is not None:
                    logits = logits + attention_mask
            attention_weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_output = torch.matmul(attention_weights, selected_values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, module.num_heads * module.v_head_dim)
        attn_output = module.o_proj(attn_output)
        if not output_attentions:
            attention_weights = None
        return attn_output, attention_weights, past_key_value


def patch_model(model: torch.nn.Module, controller: SparseTeacherController) -> int:
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV2Attention":
            continue

        def wrapped(this: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
            return controller.forward(this, *args, **kwargs)

        module.forward = types.MethodType(wrapped, module)
        count += 1
    return count


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[str(row["id"])] = row
    return result


def fit_tokens(tokens: list[int], needed: int) -> list[int]:
    if len(tokens) >= needed:
        return tokens[:needed]
    repeats = (needed + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:needed]


@torch.no_grad()
def run_sequence(
    model: torch.nn.Module,
    controller: SparseTeacherController,
    mode: str,
    policies: dict[int, ApproxPolicy],
    token_ids: list[int],
    context_length: int,
    steps: int,
    selected_layers: list[int],
) -> RunOutput:
    controller.reset(mode, policies)
    input_device = model.model.embed_tokens.weight.device
    prefill = torch.tensor(
        token_ids[:context_length], device=input_device, dtype=torch.long
    ).view(1, -1)
    # Bypass the CausalLM head during prefill.  The pinned remote model always
    # projects every prefill position to the 102k-token vocabulary, which is a
    # needless ~12.5 GiB allocation at 32K; only the KV cache is required here.
    prefill_output = model.model(input_ids=prefill, use_cache=True, return_dict=True)
    past = prefill_output.past_key_values
    logits = []
    ground_truth = []
    finals = []
    selected = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(steps):
        token = torch.tensor(
            [[token_ids[context_length + step]]], device=input_device, dtype=torch.long
        )
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        past = output.past_key_values
        logits.append(output.logits[0, -1].float().cpu())
        ground_truth.append(token_ids[context_length + step + 1])
        finals.append(output.hidden_states[-1][0, -1].float().cpu())
        selected.append(
            torch.stack(
                [output.hidden_states[layer + 1][0, -1].float().cpu() for layer in selected_layers]
            )
        )
    torch.cuda.synchronize()
    return RunOutput(
        logits=torch.stack(logits),
        ground_truth=torch.tensor(ground_truth, dtype=torch.long),
        final_hidden=torch.stack(finals),
        selected_hidden=torch.stack(selected),
        elapsed_seconds=time.perf_counter() - started,
    )


def hidden_metrics(baseline: torch.Tensor, approximate: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    cosine = F.cosine_similarity(baseline.float(), approximate.float(), dim=-1).numpy()
    relative_l2 = (
        torch.linalg.vector_norm(approximate.float() - baseline.float(), dim=-1)
        / torch.linalg.vector_norm(baseline.float(), dim=-1).clamp_min(1e-12)
    ).numpy()
    return cosine, relative_l2


def comparison_rows(
    baseline: RunOutput,
    approximate: RunOutput,
    *,
    policy: str,
    role: str,
    prompt_id: str,
    workload: str,
    context_length: int,
    selected_layers: list[int],
    comparison: str,
) -> list[dict[str, Any]]:
    base_logp = F.log_softmax(baseline.logits.float(), dim=-1)
    approx_logp = F.log_softmax(approximate.logits.float(), dim=-1)
    base_p = base_logp.exp()
    kl = (base_p * (base_logp - approx_logp)).sum(dim=-1).numpy()
    base_top1 = baseline.logits.argmax(dim=-1)
    approx_top1 = approximate.logits.argmax(dim=-1)
    base_top5 = baseline.logits.topk(5, dim=-1).indices
    approx_top5 = approximate.logits.topk(5, dim=-1).indices
    target = baseline.ground_truth
    base_nll = -base_logp[torch.arange(target.size(0)), target].numpy()
    approx_nll = -approx_logp[torch.arange(target.size(0)), target].numpy()
    final_cos, final_l2 = hidden_metrics(baseline.final_hidden, approximate.final_hidden)
    selected_cos, selected_l2 = hidden_metrics(
        baseline.selected_hidden, approximate.selected_hidden
    )
    rows = []
    for step in range(target.size(0)):
        overlap = len(set(base_top5[step].tolist()) & set(approx_top5[step].tolist())) / 5
        rows.append(
            {
                "comparison": comparison,
                "policy": policy,
                "selection_role": role,
                "prompt_id": prompt_id,
                "workload": workload,
                "context_length": context_length,
                "step": step,
                "logit_kl": float(kl[step]),
                "top1_agreement": bool(base_top1[step] == approx_top1[step]),
                "top5_overlap": overlap,
                "baseline_nll": float(base_nll[step]),
                "approx_nll": float(approx_nll[step]),
                "final_hidden_cosine": float(final_cos[step]),
                "final_hidden_relative_l2": float(final_l2[step]),
                "selected_hidden_cosine_mean": float(selected_cos[step].mean()),
                "selected_hidden_cosine_worst": float(selected_cos[step].min()),
                "selected_hidden_relative_l2_mean": float(selected_l2[step].mean()),
                "selected_hidden_relative_l2_worst": float(selected_l2[step].max()),
                "baseline_decode_seconds": baseline.elapsed_seconds,
                "approx_decode_seconds": approximate.elapsed_seconds,
            }
        )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for keys, group in rows.groupby(["comparison", "policy", "selection_role"], sort=False):
        comparison, policy, role = keys
        base_ppl = math.exp(float(group.baseline_nll.mean()))
        approx_ppl = math.exp(float(group.approx_nll.mean()))
        output.append(
            {
                "comparison": comparison,
                "policy": policy,
                "selection_role": role,
                "tokens": len(group),
                "logit_kl_mean": group.logit_kl.mean(),
                "logit_kl_median": group.logit_kl.median(),
                "logit_kl_p95": group.logit_kl.quantile(0.95),
                "logit_kl_p99": group.logit_kl.quantile(0.99),
                "top1_agreement": group.top1_agreement.mean(),
                "top5_overlap": group.top5_overlap.mean(),
                "baseline_ppl": base_ppl,
                "approx_ppl": approx_ppl,
                "ppl_delta": (approx_ppl - base_ppl) / base_ppl,
                "final_hidden_cosine_median": group.final_hidden_cosine.median(),
                "final_hidden_cosine_p5": group.final_hidden_cosine.quantile(0.05),
                "final_hidden_relative_l2_p95": group.final_hidden_relative_l2.quantile(0.95),
                "selected_hidden_cosine_p5": group.selected_hidden_cosine_worst.quantile(0.05),
                "selected_hidden_relative_l2_p95": group.selected_hidden_relative_l2_worst.quantile(0.95),
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced sparse-model validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-ids", nargs="+", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--roles", nargs="+", default=["Safe"])
    parser.add_argument("--include-dense", action="store_true")
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
    specs = [spec for spec in selection["policies"] if spec["selection_role"] in args.roles]
    if not specs:
        raise ValueError(f"no selected policies for roles {args.roles}")
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=fixed_device_map(config.num_hidden_layers), attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    first_attention = model.model.layers[0].self_attn
    apply_rotary = first_attention.__class__.forward.__globals__["apply_rotary_pos_emb"]
    controller = SparseTeacherController(
        model, apply_rotary, args.layers, args.checkpoints,
        int(selection["k"]), int(selection["block_size"]),
    )
    patched = patch_model(model, controller)
    if patched != config.num_hidden_layers:
        raise RuntimeError(f"patched {patched} attention modules")
    prompts = load_prompts(args.prompts)
    rows = []
    run_audit = []
    for prompt_id in args.prompt_ids:
        prompt = prompts[prompt_id]
        raw_tokens = tokenizer(prompt["text"], add_special_tokens=True)["input_ids"]
        for context_length in args.contexts:
            token_ids = fit_tokens(raw_tokens, context_length + args.steps + 1)
            dense = None
            if args.include_dense:
                dense = run_sequence(
                    model, controller, "dense", {}, token_ids, context_length,
                    args.steps, args.layers,
                )
            baseline = run_sequence(
                model, controller, "baseline", {}, token_ids, context_length,
                args.steps, args.layers,
            )
            if dense is not None:
                rows.extend(
                    comparison_rows(
                        dense, baseline, policy="full-indexer-sparse", role="Baseline",
                        prompt_id=prompt_id, workload=prompt["workload"],
                        context_length=context_length, selected_layers=args.layers,
                        comparison="dense_vs_full_indexer",
                    )
                )
            for spec in specs:
                policies = {
                    layer: build_policy(spec, layer, scales) for layer in args.layers
                }
                approximate = run_sequence(
                    model, controller, "approx", policies, token_ids, context_length,
                    args.steps, args.layers,
                )
                rows.extend(
                    comparison_rows(
                        baseline, approximate, policy=spec["config_id"],
                        role=spec["selection_role"], prompt_id=prompt_id,
                        workload=prompt["workload"], context_length=context_length,
                        selected_layers=args.layers,
                        comparison="full_indexer_vs_approx",
                    )
                )
            run_audit.append(
                {"prompt_id": prompt_id, "context_length": context_length, "steps": args.steps}
            )
            print(f"completed {prompt_id} context={context_length}", flush=True)
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "teacher_forced_quality.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output / "teacher_forced_summary.csv", index=False)
    audit = {
        "model_revision": args.revision,
        "gpus": gpu_names,
        "visible_gpu_ids": [0, 1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "layers": args.layers,
        "roles": args.roles,
        "include_dense": args.include_dense,
        "runs": run_audit,
        "reference_implementation": True,
    }
    (args.output / "teacher_forced_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
