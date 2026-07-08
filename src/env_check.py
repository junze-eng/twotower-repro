"""Small-GPU environment check: validates the whole toolchain WITHOUT loading the 60GB
weights. A cheap card (even ~16-24GB) is enough — this imports and exercises the CUDA
kernels, then loads only config + tokenizer. Full generation still needs 2x80GB.

    python src/env_check.py
"""
import torch

from twotower import MODEL_NAME, MASK_TOKEN_ID


def main():
    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
    assert torch.cuda.is_available(), "no CUDA device visible"
    print("device:", torch.cuda.get_device_name(0))

    # 1. the two ABI-sensitive kernel packages must import AND resolve symbols
    import causal_conv1d
    import mamba_ssm
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update  # noqa
    from causal_conv1d import causal_conv1d_fn  # noqa: F401
    print("mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__)

    # 2. actually run a tiny causal_conv1d kernel to prove it executes on this GPU
    x = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16)
    _ = causal_conv1d_fn(x, w, None, activation="silu")
    print("causal_conv1d kernel ran OK")

    # 3. transformers + the custom modeling file must import (needs mamba-ssm present)
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    print("transformers", transformers.__version__)
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"config OK: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"pattern {cfg.hybrid_override_pattern[:12]}...")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"tokenizer OK: vocab {tok.vocab_size}, "
          f"decode({MASK_TOKEN_ID})={tok.decode([MASK_TOKEN_ID])!r}")

    print("\nenv_check PASSED — toolchain good. Full generation needs 2x80GB (smoke_test.py).")


if __name__ == "__main__":
    main()
