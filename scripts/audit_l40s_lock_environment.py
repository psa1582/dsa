from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess

import torch
import triton


def command(*args: str) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def relative_paths(repo: Path, pattern: str) -> list[str]:
    return sorted(str(path.relative_to(repo)) for path in repo.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the L40S DSA lock environment")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-base-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    smi_rows = command(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ).splitlines()
    gpus = []
    for row in smi_rows:
        index, name, uuid, memory, driver, capability = [value.strip() for value in row.split(",")]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "memory_total_mib": int(memory),
                "driver": driver,
                "compute_capability": capability,
            }
        )

    trace_roots: set[Path] = set()
    for pattern in ("*.npz", "*.pt"):
        for path in (repo / "artifacts").rglob(pattern):
            if "l40s_dsa_lock" in path.parts:
                continue
            parent = path.parent
            while parent.parent != repo / "artifacts" and not (
                parent.name in {"traces", "selection_details", "calibration", "validation", "test"}
                or parent.name.startswith("trace_capture_")
                or parent.name.startswith("warmup_")
            ):
                parent = parent.parent
            trace_roots.add(parent)

    result_dirs = sorted(
        {
            path.parent
            for suffix in ("*.csv", "*.json", "*.md")
            for path in (repo / "artifacts").rglob(suffix)
            if "l40s_dsa_lock" not in path.parts
        }
    )
    kernel_sources = [
        repo / "src" / "temporal_dsa" / "progressive_kernel.py",
        repo / "src" / "temporal_dsa" / "verifier_scoring.py",
        repo / "src" / "temporal_dsa" / "verifier.py",
        repo / "src" / "temporal_dsa" / "approx.py",
        repo / "scripts" / "benchmark_progressive_dsa.py",
        repo / "scripts" / "profile_progressive_dsa.py",
        repo / "scripts" / "benchmark_full64_topk_decomposition.py",
        repo / "scripts" / "benchmark_dsa_replay.py",
    ]
    payload = {
        "scope": "DeepSeek-V2-Lite research-sidecar DSA L40S lock",
        "measurement_host": "10.201.135.16:7021",
        "gpu_sku_exact": sorted(set(gpu["name"] for gpu in gpus)),
        "gpu_count_physical": len(gpus),
        "gpus": gpus,
        "measurement_visible_gpu_ids": [0],
        "protected_existing_job_gpu_ids": [2, 3],
        "driver": sorted(set(gpu["driver"] for gpu in gpus)),
        "measurement_runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "executable": __import__("sys").executable,
        },
        "system_cuda_toolkit": {
            "nvcc_path": command("bash", "-lc", "command -v nvcc"),
            "nvcc_version": command("nvcc", "--version"),
            "note": "System nvcc is not the CUDA runtime bundled with PyTorch.",
        },
        "nsight_systems": command("nsys", "--version"),
        "nsight_compute": command("bash", "-lc", "command -v ncu || true") or "unavailable",
        "source_checkout": {
            "path": str(repo),
            "is_git_repository": (repo / ".git").exists(),
            "git_base_commit_from_authoritative_local_clone": args.git_base_commit,
            "git_branch_from_authoritative_local_clone": args.git_branch,
            "authoritative_local_branch_status_during_measurement": (
                "lock-study changes in progress on exp/l40s-dsa-lock; no unrelated edits"
            ),
            "remote_snapshot_note": "Measurement checkout is a source snapshot without .git metadata.",
        },
        "bundle_materialization_runtime": {
            "python": "3.12.12",
            "torch": "2.9.1+cu128",
            "cuda": "12.8",
            "reason": "Prior locked sidecar environment contains pandas and safetensors.",
        },
        "hugging_face_provenance_audit": {
            "tool": "hf env",
            "huggingface_hub": "0.36.2",
            "cache": "/home/psa1582/.cache/huggingface/hub",
            "saved_token": False,
            "network_transfer_performed": False,
            "note": "This is provenance-only and separate from the measurement runtime.",
        },
        "trace_roots": sorted(str(path.relative_to(repo)) for path in trace_roots),
        "prior_result_directories": [str(path.relative_to(repo)) for path in result_dirs],
        "kernel_source_paths": [
            {"path": str(path.relative_to(repo)), "exists": path.exists()}
            for path in kernel_sources
        ],
        "existing_nsight_systems_reports": relative_paths(repo, "artifacts/**/*.nsys-rep"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gpu_count": len(gpus)}, sort_keys=True))


if __name__ == "__main__":
    main()
