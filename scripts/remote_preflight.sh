#!/usr/bin/env bash
set -euo pipefail

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1,2p' | grep -c 'L40')" -eq 2
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1,2p' | awk '$1 < 1024 {n++} END {print n+0}')" -eq 2
test -x "$HOME/.venvs/l40-regime/bin/python"
test -d "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader
"$HOME/.venvs/l40-regime/bin/python" -c 'import torch, transformers; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__})'

