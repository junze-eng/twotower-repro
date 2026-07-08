#!/usr/bin/env bash
# Idempotent installer. RunPod swaps containers and ONLY /workspace persists, so the pip
# env is wiped on every swap. Recover with one command: `bash setup/install.sh`.
# Wheels are cached on /workspace (PIP_CACHE_DIR) so re-install is fast (no re-download).
#
# It does NOT stop at "imports work" — it ends by SMOKE-testing that diffusion generates
# COHERENT text (src/diffusion_smoke.py). If that fails it tries the triton>=3.5 hypothesis
# and, if that doesn't help, reverts and tells you triton was not the cause.
#
# torch is LOCKED (it's the pod image); we never change it.
set -uo pipefail                      # NOT -e: we handle failures explicitly
cd "$(dirname "$0")/.."               # run from repo root (needs src/)

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

MAMBA_VER="${MAMBA_VER:-2.3.2.post1}"
CAUSAL_VER="${CAUSAL_VER:-1.6.2.post1}"   # torch2.8 wheels; this mamba/causal pair targets triton 3.4

TORCH_V=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
echo "torch: $TORCH_V (LOCKED - never modified)"
case "$TORCH_V" in
  2.8*) ;;
  *) echo "WARN: expected torch 2.8.x; the mamba/causal wheels below target torch2.8." ;;
esac

# --- detect wheel tags from the live interpreter ---
read -r PY TORCH CU ABI <<EOF
$(python - <<'PYEOF'
import torch, sys
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
tv = torch.__version__.split("+")[0].split(".")
print(py, f"torch{tv[0]}.{tv[1]}", "cu" + (torch.version.cuda or "0.0").split(".")[0],
      "cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")
PYEOF
)
EOF
echo "wheel tags: $PY $TORCH $CU $ABI"

MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VER}/mamba_ssm-${MAMBA_VER}+${CU}${TORCH}${ABI}-${PY}-${PY}-linux_x86_64.whl"
CAUSAL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_VER}/causal_conv1d-${CAUSAL_VER}+${CU}${TORCH}${ABI}-${PY}-${PY}-linux_x86_64.whl"

# --- base python deps (einops is REQUIRED by the modeling file!) ---
echo ">>> base deps"
pip install -q einops "transformers==4.57.1" accelerate safetensors sentencepiece \
    matplotlib imageio pillow datasets || { echo "base deps failed"; exit 1; }

# --- ABI-sensitive kernels (pip is idempotent: skips if already satisfied) ---
echo ">>> kernels: causal_conv1d $CAUSAL_VER, mamba_ssm $MAMBA_VER"
pip install -q "$CAUSAL_URL" || { echo "!! causal wheel 404 ($CU/$TORCH/$ABI/$PY) - see https://github.com/Dao-AILab/causal-conv1d/releases"; exit 1; }
pip install -q "$MAMBA_URL"  || { echo "!! mamba wheel 404 ($CU/$TORCH/$ABI/$PY) - see https://github.com/state-spaces/mamba/releases"; exit 1; }

python - <<'PYEOF' || { echo "!! kernel import failed"; exit 1; }
import causal_conv1d, mamba_ssm, einops, transformers, triton
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # forces symbol load
print(f"imports OK | triton {triton.__version__} | mamba {mamba_ssm.__version__} "
      f"| causal {causal_conv1d.__version__} | transformers {transformers.__version__}")
PYEOF

# --- coherence smoke drives pass/fail (needs 2 GPUs to hold both towers) ---
NGPU=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$NGPU" -lt 2 ]; then
  echo "only $NGPU GPU visible - env installed OK, but diffusion smoke needs 2 GPUs. Skipping."
  exit 0
fi

echo ">>> diffusion coherence smoke (stock stack, triton $(python -c 'import triton;print(triton.__version__)'))"
if python src/diffusion_smoke.py; then
  echo "PASS: diffusion coherent with the stock stack. Done."
  exit 0
fi

# --- stock stack produced GARBAGE: try the triton>=3.5 hypothesis, reversibly ---
TRITON_OLD=$(python -c "import triton;print(triton.__version__)")
echo "!! diffusion GARBAGE on triton $TRITON_OLD. Trying triton>=3.5 (--no-deps)..."
pip install -q --no-deps "triton>=3.5" || echo "  (no triton>=3.5 available or install failed)"
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "  torch broke under new triton -> reverting to $TRITON_OLD"
  pip install -q --no-deps "triton==$TRITON_OLD"
fi

echo ">>> re-smoke (triton $(python -c 'import triton;print(triton.__version__)'))"
if python src/diffusion_smoke.py; then
  echo "PASS: triton upgrade FIXED diffusion - keep triton $(python -c 'import triton;print(triton.__version__)')."
  exit 0
fi

echo "!! still GARBAGE after triton upgrade -> triton is NOT the cause. Reverting to $TRITON_OLD."
pip install -q --no-deps "triton==$TRITON_OLD" || true
cat <<'MSG'

================================================================================
Diffusion still garbage; triton is ruled out. Next isolation (run manually):
  Disable Mamba state seeding (initial_states=None) and re-check - this separates the
  seeding kernel path (causal_conv1d/chunk-scan) from cross-attention / AdaLN / weights.
  If disabling seeding gives grammatical text -> seeding path; else -> attn/adaLN/weights.
================================================================================
MSG
exit 1
