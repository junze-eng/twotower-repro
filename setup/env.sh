# Source this in every new shell / after every RunPod container swap:
#   source setup/env.sh
# Only /workspace survives a container swap, so caches live there.
export HF_HOME=/workspace/hf                 # 126GB weights persist here
export PIP_CACHE_DIR=/workspace/pip-cache     # wheels persist -> fast re-install
# export HF_TOKEN=hf_xxx                       # <-- fill in (needed to download gated weights)

echo "env set: HF_HOME=$HF_HOME  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "after a container swap, recover the env with:  bash setup/install.sh"
