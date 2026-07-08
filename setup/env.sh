# Source this in every new shell / after every RunPod container swap:
#   source setup/env.sh
# Only /workspace survives a container swap, so caches AND the venv live there.
export HF_HOME=/workspace/hf                 # 126GB weights persist here
export PIP_CACHE_DIR=/workspace/pip-cache     # wheels persist -> fast re-install
# export HF_TOKEN=hf_xxx                       # <-- set in your shell, do NOT commit it

# activate the persistent venv if it exists (built once by setup/install.sh)
if [ -f /workspace/venv/bin/activate ]; then
  source /workspace/venv/bin/activate
  echo "env set: HF_HOME=$HF_HOME | venv ACTIVE ($(python -V 2>&1)) -> no reinstall needed"
else
  echo "env set: HF_HOME=$HF_HOME | no venv yet -> run: bash setup/install.sh"
fi
