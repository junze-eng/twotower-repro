#!/usr/bin/env bash
# Install mamba-ssm + causal-conv1d wheels that EXACTLY match the pod's torch/cuda/python/abi.
# The #1 footgun is cxx11abi: a mismatch does not fail at install time, it fails at
# `import mamba_ssm` with `undefined symbol`. So we detect all four tags and build the URL.
set -euo pipefail

MAMBA_VER="${MAMBA_VER:-2.3.2.post1}"
CAUSAL_VER="${CAUSAL_VER:-1.6.2.post1}"   # 1.6.2.post1 has torch2.8 wheels; 1.5.0.post8 did not

# --- detect the four wheel tags from the live interpreter ---------------------
read -r PY TORCH CU ABI <<EOF
$(python - <<'PYEOF'
import torch, sys
py   = f"cp{sys.version_info.major}{sys.version_info.minor}"
tv   = torch.__version__.split("+")[0].split(".")
torch_tag = f"torch{tv[0]}.{tv[1]}"
cu   = "cu" + (torch.version.cuda or "0.0").split(".")[0]      # major only, e.g. cu12
abi  = "cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE"
print(py, torch_tag, cu, abi)
PYEOF
)
EOF

echo "Detected: python=$PY  $TORCH  $CU  $ABI"

MAMBA_WHL="mamba_ssm-${MAMBA_VER}+${CU}${TORCH}${ABI}-${PY}-${PY}-linux_x86_64.whl"
CAUSAL_WHL="causal_conv1d-${CAUSAL_VER}+${CU}${TORCH}${ABI}-${PY}-${PY}-linux_x86_64.whl"

MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VER}/${MAMBA_WHL}"
CAUSAL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_VER}/${CAUSAL_WHL}"

echo "mamba wheel : $MAMBA_URL"
echo "causal wheel: $CAUSAL_URL"

# --- python deps that never fight the ABI ------------------------------------
# transformers pinned to the version baked into the model's config.json (4.57.1);
# the custom modeling code imports DynamicCache / is_mamba_2_ssm_available against it.
pip install -q "transformers==4.57.1" einops accelerate safetensors sentencepiece
# offline rendering deps for the Exp0 GIF (CPU-only, harmless on the GPU box)
pip install -q matplotlib imageio pillow

# --- the two ABI-sensitive kernels -------------------------------------------
install_wheel () {
  local url="$1" name="$2"
  echo ">>> installing $name"
  if ! pip install "$url"; then
    cat >&2 <<MSG

!! $name wheel not found for this exact combo ($CU / $TORCH / $ABI / $PY).
   Browse the release assets and pick the closest match, or build from source:
     mamba : https://github.com/state-spaces/mamba/releases
     causal: https://github.com/Dao-AILab/causal-conv1d/releases
   Source fallback (needs matching CUDA toolkit + nvcc on PATH):
     pip install $name --no-build-isolation
MSG
    return 1
  fi
}
install_wheel "$CAUSAL_URL" causal_conv1d   # install causal-conv1d first; mamba imports it
install_wheel "$MAMBA_URL"  mamba_ssm

# --- verify the import actually resolves the symbols -------------------------
python - <<'PYEOF'
import causal_conv1d, mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # forces symbol load
print("OK: mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__)
PYEOF
echo "Environment ready."
