from __future__ import annotations

import argparse
import json
import platform
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"


def fixed_device_map(num_layers: int) -> dict[str, int]:
    split = (num_layers + 1) // 2
    mapping: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.norm": 1,
        "lm_head": 1,
    }
    for layer in range(num_layers):
        mapping[f"model.layers.{layer}"] = 0 if layer < split else 1
    return mapping


@dataclass
class LayerCapture:
    hidden_chunks: list[torch.Tensor] = field(default_factory=list)
    teacher_rows: list[torch.Tensor] = field(default_factory=list)
    context_length: int = 0

    def add_hidden(self, hidden: torch.Tensor, *, prefill: bool) -> None:
        value = hidden[0].detach().to(device="cpu", dtype=torch.float16)
        self.hidden_chunks.append(value)
        if prefill:
            self.context_length += int(value.shape[0])

    def add_teacher(self, attention_logits: torch.Tensor) -> None:
        probabilities = torch.softmax(attention_logits[0, :, 0].float(), dim=-1)
        target = probabilities.sum(dim=0)
        target = target / target.sum().clamp_min(1e-12)
        self.teacher_rows.append(target.detach().to(device="cpu", dtype=torch.float16))

    def payload(self) -> dict[str, Any]:
        hidden = torch.cat(self.hidden_chunks, dim=0)
        lengths = torch.tensor([row.numel() for row in self.teacher_rows], dtype=torch.int32)
        width = int(lengths.max()) if lengths.numel() else 0
        teacher = torch.zeros((len(self.teacher_rows), width), dtype=torch.float16)
        for row_id, row in enumerate(self.teacher_rows):
            teacher[row_id, : row.numel()] = row
        return {
            "hidden": hidden,
            "teacher_probabilities": teacher,
            "lengths": lengths,
            "context_length": self.context_length,
        }


class DenseTeacherController:
    """V2-Lite MLA forward with selected-layer decode capture.

    Algebra follows DeepSeek-V2-Lite's trusted remote attention implementation.
    Prefill uses dense SDPA without materializing attention weights. Decode
    materializes the one-query logits needed for the official Indexer target.
    """

    def __init__(self, apply_rotary: Any, selected_layers: set[int]) -> None:
        self.apply_rotary = apply_rotary
        self.selected_layers = selected_layers
        self.captures = {layer: LayerCapture() for layer in selected_layers}

    def reset(self) -> None:
        self.captures = {layer: LayerCapture() for layer in self.selected_layers}

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
        if bsz != 1:
            raise NotImplementedError("the pilot collector requires batch size 1")
        selected = int(module.layer_idx) in self.selected_layers
        if selected:
            self.captures[int(module.layer_idx)].add_hidden(hidden_states, prefill=q_len > 1)

        if module.q_lora_rank is None:
            q = module.q_proj(hidden_states)
        else:
            q = module.q_b_proj(module.q_a_layernorm(module.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, module.num_heads, module.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)

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
        k_nope, value_states = torch.split(kv, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, module.layer_idx)
        cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
        if position_ids is None:
            raise ValueError("position_ids is required")
        q_pe, k_pe = self.apply_rotary(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, module.num_heads, q_len, module.q_head_dim)
        query_states[..., : module.qk_nope_head_dim] = q_nope
        query_states[..., module.qk_nope_head_dim :] = q_pe
        key_states = k_pe.new_empty(bsz, module.num_heads, q_len, module.q_head_dim)
        key_states[..., : module.qk_nope_head_dim] = k_nope
        key_states[..., module.qk_nope_head_dim :] = k_pe
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
            logits = torch.matmul(query_states, key_states.transpose(2, 3)) * module.softmax_scale
            if attention_mask is not None:
                logits = logits + attention_mask
            if selected:
                self.captures[int(module.layer_idx)].add_teacher(logits)
            attention_weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attention_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, module.num_heads * module.v_head_dim)
        attn_output = module.o_proj(attn_output)
        if not output_attentions:
            attention_weights = None
        return attn_output, attention_weights, past_key_value


def patch_attention(model: torch.nn.Module, controller: DenseTeacherController) -> int:
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV2Attention":
            continue

        def wrapped(this: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
            return controller.forward(this, *args, **kwargs)

        module.forward = types.MethodType(wrapped, module)
        count += 1
    return count


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"id", "workload", "text"}.issubset(row):
                raise ValueError(f"{path}:{line_number}: expected id, workload, text")
            prompts.append(row)
    return prompts


def _fit_context(token_ids: list[int], context_length: int, allow_repeat: bool) -> list[int]:
    if len(token_ids) >= context_length:
        return token_ids[:context_length]
    if not allow_repeat:
        raise ValueError(f"prompt has {len(token_ids)} tokens, needs {context_length}")
    repeats = (context_length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:context_length]


@torch.no_grad()
def collect_prompt(
    model: torch.nn.Module,
    controller: DenseTeacherController,
    token_ids: list[int],
    decode_steps: int,
    input_device: torch.device,
) -> tuple[list[int], float, float]:
    controller.reset()
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=input_device).view(1, -1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    past = output.past_key_values
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True).to(input_device)
    generated = []
    started = time.perf_counter()
    for _ in range(decode_steps):
        generated.append(int(token.item()))
        output = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True).to(input_device)
    torch.cuda.synchronize()
    return generated, prefill_seconds, time.perf_counter() - started


def run(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache

    if not hasattr(Cache, "get_usable_length"):
        def get_usable_length(self: Any, new_seq_length: int, layer_idx: int = 0) -> int:
            del new_seq_length
            return int(self.get_seq_length(layer_idx))
        Cache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly two visible GPUs, found {torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not args.allow_non_l40 and not all("L40" in name for name in gpu_names):
        raise RuntimeError(f"L40 guard rejected GPUs: {gpu_names}; use --allow-non-l40 only for smoke tests")

    args.output.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=True
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
        args.model, revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=True
    )
    input_device = model.model.embed_tokens.weight.device
    first_attention = model.model.layers[0].self_attn
    apply_rotary = first_attention.__class__.forward.__globals__["apply_rotary_pos_emb"]
    layers = set(args.layers)
    controller = DenseTeacherController(apply_rotary, layers)
    patched = patch_attention(model, controller)
    if patched != config.num_hidden_layers:
        raise RuntimeError(f"patched {patched} layers, expected {config.num_hidden_layers}")

    audit = {
        "model": args.model,
        "revision": args.revision,
        "selected_layers": sorted(layers),
        "context_length": args.context_length,
        "decode_steps": args.decode_steps,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "gpus": gpu_names,
        "device_map": getattr(model, "hf_device_map", None),
    }
    (args.output / "runtime_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summaries = []
    for prompt in _load_prompts(args.prompts):
        token_ids = tokenizer(prompt["text"], add_special_tokens=True)["input_ids"]
        token_ids = _fit_context(token_ids, args.context_length, args.allow_repeat)
        generated, prefill_seconds, decode_seconds = collect_prompt(
            model, controller, token_ids, args.decode_steps, input_device
        )
        prompt_root = args.output / prompt["id"]
        prompt_root.mkdir(parents=True, exist_ok=True)
        for layer, capture in controller.captures.items():
            payload = capture.payload()
            payload["metadata"] = {
                "prompt_id": prompt["id"],
                "workload": prompt["workload"],
                "layer": layer,
                "model_revision": args.revision,
                "source": prompt.get("source"),
                "split": prompt.get("split"),
            }
            torch.save(payload, prompt_root / f"layer_{layer:02d}.pt")
        summaries.append(
            {
                "prompt_id": prompt["id"],
                "workload": prompt["workload"],
                "source": prompt.get("source"),
                "split": prompt.get("split"),
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "generated_token_ids": generated,
            }
        )
    (args.output / "collection_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect V2-Lite dense MLA teacher samples")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11, 14, 17, 21, 25])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--allow-non-l40", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
