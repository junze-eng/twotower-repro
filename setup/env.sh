# Source this in every new shell:
#   source setup/env.sh
# Only the 126GB WEIGHTS persist on /workspace (network volume). The python env installs to
# the LOCAL container disk (a venv on the network FS hangs pip), so after a container swap
# you re-run `bash setup/install.sh` (~2 min) — you do NOT re-download the weights.
export HF_HOME=/workspace/hf                  # 126GB weights persist here
export HF_HUB_OFFLINE=1                        # weights+code are local; stay offline so HF never
                                              # re-pulls a new revision or re-downloads 126GB.
                                              # For the humaneval download, override inline:
                                              #   HF_HUB_OFFLINE=0 python src/prep_data.py --which humaneval ...
# export HF_TOKEN=hf_xxx                       # set in your shell; do NOT commit it

echo "env set: HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "after a container swap: bash setup/install.sh   (env is local+fast; weights stay on /workspace)"
