# coding=utf-8
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Two-tower NemotronH for HuggingFace — real separate context + denoiser weights.
#
# Checkpoint key layout (from converted safetensors):
#   context_tower.*        — context backbone (NemotronHModel)
#   context_lm_head.weight — context output head
#   denoiser_tower.*       — denoiser backbone (NemotronHModel)
#   lm_head.weight         — denoiser output head
#   t_embedder.*           — timestep embedder (optional, for mask_diffusion)
#   t_block.*              — timestep MLP (optional)
#   scale_shift_tables.*   — per-layer modulation bias (optional)
#
# Modes:
#   AR:             forward() + generate() — context_tower only
#   Mock-AR:        generate_mock_ar() — two-tower, S-2/KV[:-1] semantics
#   Mask-Diffusion: generate_mask_diffusion() — block-wise iterative denoising

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from .modeling_nemotron_h import (
        HybridMambaAttentionDynamicCache,
        NemotronHCausalLMOutput,
        NemotronHForCausalLM,
        NemotronHModel,
        NemotronHPreTrainedModel,
        repeat_kv,
    )
    from .configuration_nemotron_h import NemotronHConfig
except ImportError:
    from modeling_nemotron_h import (
        HybridMambaAttentionDynamicCache,
        NemotronHCausalLMOutput,
        NemotronHForCausalLM,
        NemotronHModel,
        NemotronHPreTrainedModel,
        repeat_kv,
    )
    from configuration_nemotron_h import NemotronHConfig

from transformers.generation import GenerationMixin


# ---------------------------------------------------------------------------
# Time conditioning (PixArt-alpha adaLN-single style)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal + MLP embedder for scalar timesteps in [0,1]."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 max_period: int = 1000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_scaled = t * self.max_period
        t_freq = self.timestep_embedding(t_scaled, self.frequency_embedding_size)
        return self.mlp(t_freq)


def _modulate(x, shift, scale):
    """Adaptive LN: x * (1 + scale) + shift. Broadcasts for (B,L,D) input."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_mod_params(t_emb, table):
    """(B, 3*D) + (3, D) -> (shift, scale, gate) each (B, D)."""
    B, D = t_emb.shape[0], table.shape[1]
    combined = table[None] + t_emb.reshape(B, 3, D)
    shift, scale, gate = combined.chunk(3, dim=1)
    return shift.squeeze(1), scale.squeeze(1), gate.squeeze(1)


# ---------------------------------------------------------------------------
# Bug-fixed cache
# ---------------------------------------------------------------------------

class FixedHybridCache(HybridMambaAttentionDynamicCache):
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        super().__init__(config, batch_size, dtype, device)
        self.conv_kernel_size = config.conv_kernel

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states[layer_idx].device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(
                self.conv_states[layer_idx].device
            )
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]


# ---------------------------------------------------------------------------
# Two-Tower CausalLM
# ---------------------------------------------------------------------------

class NemotronHTwoTowerForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    """Two-tower NemotronH with real separate context and denoiser weights.

    Modes:
        AR:             forward() + generate() — context_tower only
        Mock-AR:        generate_mock_ar() — S-2/KV[:-1] semantics
        Mask-Diffusion: generate_mask_diffusion() — block-wise confidence_unmasking
    """

    _tied_weights_keys = []

    def __init__(self, config: NemotronHConfig):
        super().__init__(config)
        self.context_tower = NemotronHModel(config)
        self.context_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.denoiser_tower = NemotronHModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        # Time conditioning (created unconditionally; weights loaded if present)
        H = config.hidden_size
        N = config.num_hidden_layers
        self.t_embedder = TimestepEmbedder(H)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(H, 3 * H, bias=True))
        self.scale_shift_tables = nn.ParameterList([
            nn.Parameter(torch.randn(3, H) / (H ** 0.5)) for _ in range(N)
        ])

        self.post_init()

    # ------------------------------------------------------------------
    # HF interface
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.context_tower.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.context_tower.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.context_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.context_lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, cache_position=None, position_ids=None,
        use_cache=True, **kwargs,
    ):
        empty_past_kv = past_key_values is None
        if not empty_past_kv:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        else:
            # FixedHybridCache (not the base class) so the Mamba mixer finds
            # conv_kernel_size during the cached forward (needed for AR generate).
            past_key_values = FixedHybridCache(
                self.config, input_ids.shape[0], self.dtype,
                device=next(self.context_tower.parameters()).device,
            )
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update({
            "position_ids": position_ids, "past_key_values": past_key_values,
            "use_cache": use_cache, "attention_mask": attention_mask,
            "logits_to_keep": self.config.num_logits_to_keep,
            "cache_position": cache_position,
        })
        return model_inputs

    # ------------------------------------------------------------------
    # Forward (context tower only, for HF generate)
    # ------------------------------------------------------------------

    def forward(
        self, input_ids=None, inputs_embeds=None, position_ids=None,
        cache_params=None, labels=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, use_cache=None,
        cache_position=None, attention_mask=None, **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        past_key_values = kwargs.pop("past_key_values", None)
        if past_key_values is not None and cache_params is None:
            cache_params = past_key_values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.context_tower(
            input_ids, cache_params=cache_params, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, use_cache=use_cache,
            cache_position=cache_position, attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        logits = self.context_lm_head(hidden_states.to(self.context_lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return NemotronHCausalLMOutput(
            loss=loss, logits=logits, cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Layer-by-layer forward with cache + optional time conditioning
    # ------------------------------------------------------------------

    def _forward_tower_with_cache(self, tower, lm_head, input_ids, cache,
                                  cache_position, t_emb=None):
        """Forward through tower with KV cache. If t_emb is provided, applies
        PixArt-style adaLN modulation (shift/scale after norm, gate on output)."""
        hidden = tower.embeddings(input_ids)
        causal_mask = tower._update_causal_mask(None, hidden, cache_position)

        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            hidden = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            mod = None
            if t_emb is not None:
                mod = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
                shift, scale, gate = mod
                hidden = _modulate(hidden, shift, scale)

            if block.block_type == "mamba":
                hidden = block.mixer(
                    hidden, cache_params=cache, cache_position=cache_position,
                )
            elif block.block_type == "attention":
                hidden, _, _ = block.mixer(
                    hidden, attention_mask=causal_mask,
                    past_key_value=cache, cache_position=cache_position,
                )
            elif block.block_type in ["mlp", "moe"]:
                hidden = block.mixer(hidden)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")

            if mod is not None:
                hidden = gate.unsqueeze(1) * hidden

            hidden = residual + hidden

        hidden = tower.norm_f(hidden)
        logits = lm_head(hidden.to(lm_head.weight.dtype)).float()
        return logits

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _make_cache(self, config, batch_size, dtype, device):
        return FixedHybridCache(config, batch_size, dtype, device)

    def _build_context_cache(self, prompt_ids):
        """Two-pass context prefill: S-2 and S-1 Mamba states + full KV."""
        B, S = prompt_ids.shape
        device = prompt_ids.device
        tower = self.context_tower
        pattern = self.config.hybrid_override_pattern

        cache_p1 = self._make_cache(self.config, B, self.dtype, device)
        cp_p1 = torch.arange(S - 1, device=device)
        self._forward_tower_with_cache(tower, self.context_lm_head,
                                       prompt_ids[:, :-1], cache_p1, cp_p1)

        mamba_s2 = {}
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                mamba_s2[i] = (cache_p1.conv_states[i].clone(),
                               cache_p1.ssm_states[i].clone())

        cache_p2 = self._make_cache(self.config, B, self.dtype, device)
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                cache_p2.conv_states[i] = cache_p1.conv_states[i].clone()
                cache_p2.ssm_states[i] = cache_p1.ssm_states[i].clone()
            elif pattern[i] == "*":
                cache_p2.key_cache[i] = cache_p1.key_cache[i].clone()
                cache_p2.value_cache[i] = cache_p1.value_cache[i].clone()

        cache_p2.has_previous_state = True
        cp_p2 = torch.arange(S - 1, S, device=device)
        logits = self._forward_tower_with_cache(tower, self.context_lm_head,
                                                prompt_ids[:, -1:], cache_p2, cp_p2)

        # "logits" = context tower's prediction at the last prompt position
        # (used by generate_ar). Diffusion/mock-AR ignore it.
        return {"ctx_cache": cache_p2, "mamba_s2": mamba_s2, "ctx_len": S, "logits": logits}

    def _extend_context_cache(self, new_tokens, cache_state, block_wise=True):
        """Extend context cache by new_tokens (B, L).

        block_wise=True (diffusion): Mamba advances via a single block chunk-scan
        (fast for a whole committed block; matches mcore).
        block_wise=False (AR / mock-AR): token-by-token single-step decode, the
        same kernels stock single-tower uses, so AR/mock-AR output matches stock.
        Also stores cache_state["logits"] (last-token prediction) when single-step.
        """
        ctx_cache = cache_state["ctx_cache"]
        pattern = self.config.hybrid_override_pattern
        ctx_len = cache_state["ctx_len"]
        tower = self.context_tower
        ctx_device = next(tower.parameters()).device
        L = new_tokens.shape[1]
        tokens = new_tokens.to(ctx_device)

        # Snapshot pre-extension Mamba states as the new S-2 (used by mock-AR).
        new_s2 = {}
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                new_s2[i] = (ctx_cache.conv_states[i].clone(),
                             ctx_cache.ssm_states[i].clone())
        cache_state["mamba_s2"] = new_s2

        ctx_cache.has_previous_state = True

        if not block_wise:
            # Single-step token-by-token extension (stock decode kernels).
            logits = None
            for j in range(L):
                cp = torch.tensor([ctx_len + j], device=ctx_device)
                logits = self._forward_tower_with_cache(
                    tower, self.context_lm_head, tokens[:, j:j+1], ctx_cache, cp,
                )
            cache_state["ctx_len"] = ctx_len + L
            cache_state["logits"] = logits
            return cache_state

        cache_position = torch.arange(ctx_len, ctx_len + L, device=ctx_device)
        hidden = tower.embeddings(tokens)
        causal_mask = tower._update_causal_mask(None, hidden, cache_position)

        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            if block.block_type == "mamba":
                d_conv = block.mixer.conv_kernel_size
                init_conv = ctx_cache.conv_states[layer_idx][..., -(d_conv - 1):]
                init_ssm = ctx_cache.ssm_states[layer_idx].contiguous()
                h, new_conv, new_ssm = self._denoiser_block_mamba(
                    block.mixer, h, init_conv, init_ssm, return_states=True,
                )
                ctx_cache.conv_states[layer_idx] = new_conv
                ctx_cache.ssm_states[layer_idx] = new_ssm
            elif block.block_type == "attention":
                # Standard cached attention appends block KV (causal within block).
                h, _, _ = block.mixer(
                    h, attention_mask=causal_mask,
                    past_key_value=ctx_cache, cache_position=cache_position,
                )
            elif block.block_type in ["mlp", "moe"]:
                h = block.mixer(h)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")

            hidden = residual + h

        cache_state["ctx_len"] = ctx_len + L
        return cache_state

    def _build_denoiser_cache_mock_ar(self, cache_state, device):
        """Mock-AR denoiser cache: Mamba S-2, Attention KV[:-1]."""
        ctx_cache = cache_state["ctx_cache"]
        mamba_s2 = cache_state["mamba_s2"]
        pattern = self.config.hybrid_override_pattern
        B = ctx_cache.conv_states[0].shape[0] if pattern[0] == "M" else ctx_cache.key_cache[0].shape[0]

        den = self._make_cache(self.config, B, self.dtype, device)
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                conv_s2, ssm_s2 = mamba_s2[i]
                den.conv_states[i] = conv_s2.to(device).clone()
                den.ssm_states[i] = ssm_s2.to(device).clone()
            elif pattern[i] == "*":
                k, v = ctx_cache.key_cache[i], ctx_cache.value_cache[i]
                if k.dim() == 4 and k.shape[2] > 0:
                    den.key_cache[i] = k[:, :, :-1, :].to(device).clone()
                    den.value_cache[i] = v[:, :, :-1, :].to(device).clone()
        den.has_previous_state = True
        return den

    def _build_denoiser_cache_diffusion(self, cache_state, device):
        """Diffusion denoiser cache: Mamba S-1 (latest), full Attention KV."""
        ctx_cache = cache_state["ctx_cache"]
        pattern = self.config.hybrid_override_pattern
        B = ctx_cache.conv_states[0].shape[0] if pattern[0] == "M" else ctx_cache.key_cache[0].shape[0]

        den = self._make_cache(self.config, B, self.dtype, device)
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                den.conv_states[i] = ctx_cache.conv_states[i].to(device).clone()
                den.ssm_states[i] = ctx_cache.ssm_states[i].to(device).clone()
            elif pattern[i] == "*":
                k, v = ctx_cache.key_cache[i], ctx_cache.value_cache[i]
                if k.dim() == 4 and k.shape[2] > 0:
                    den.key_cache[i] = k.to(device).clone()
                    den.value_cache[i] = v.to(device).clone()
        den.has_previous_state = True
        return den

    # ------------------------------------------------------------------
    # Denoiser step (shared by mock-AR and diffusion)
    # ------------------------------------------------------------------

    def _run_denoiser_step_mock_ar(self, input_ids, cache_state):
        """Mock-AR denoiser: pos=ctx_len-1, KV[:-1], Mamba S-2."""
        ctx_len = cache_state["ctx_len"]
        den_device = next(self.denoiser_tower.parameters()).device
        den_input = input_ids.to(den_device)
        den_cache = self._build_denoiser_cache_mock_ar(cache_state, den_device)
        cp = torch.tensor([ctx_len - 1], device=den_device)
        return self._forward_tower_with_cache(
            self.denoiser_tower, self.lm_head, den_input, den_cache, cp,
        )

    def _denoiser_block_attention(self, mixer, hidden, ctx_k, ctx_v):
        """Bidirectional denoiser self-attention over [context_KV | block_KV].

        Mirrors the mcore `_forward_attn_with_past` (is_causal=False, no mask):
        every block position attends to ALL context positions and ALL block
        positions (the noisy block is processed bidirectionally within itself).

        Args:
            mixer: NemotronHAttention module (provides q/k/v/o projections)
            hidden: (B, L, D) post-norm (and post-modulation) block hidden states
            ctx_k, ctx_v: context KV, each (B, num_kv_heads, ctx_len, head_dim)

        Returns: (B, L, D) attention output (before residual add)
        """
        bsz, q_len, _ = hidden.shape
        q = mixer.q_proj(hidden).view(bsz, q_len, mixer.num_heads, mixer.head_dim).transpose(1, 2)
        k = mixer.k_proj(hidden).view(bsz, q_len, mixer.num_key_value_heads, mixer.head_dim).transpose(1, 2)
        v = mixer.v_proj(hidden).view(bsz, q_len, mixer.num_key_value_heads, mixer.head_dim).transpose(1, 2)

        # Concatenate context KV (past) with current block KV on the sequence dim.
        k = torch.cat([ctx_k.to(k.dtype), k], dim=2)
        v = torch.cat([ctx_v.to(v.dtype), v], dim=2)

        # GQA: expand KV heads to match query heads.
        k = repeat_kv(k, mixer.num_key_value_groups)
        v = repeat_kv(v, mixer.num_key_value_groups)

        # Full (non-causal) attention: block sees all context + whole block.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, mixer.num_heads * mixer.head_dim
        )
        return mixer.o_proj(attn_output)

    def _denoiser_block_mamba(self, mixer, hidden, init_conv, init_ssm, return_states=False):
        """Chunk-scan the whole block through the Mamba mixer, seeded from the
        context state — mirrors mcore `forward_mamba_layer_with_states`
        (non-bidirectional). Uses the same mamba_ssm/causal_conv1d kernels as
        mcore, instead of HF's token-by-token single-step path (which is both a
        numerical mismatch and crashes in this env's causal_conv1d_update).

        Args:
            mixer: NemotronHMamba2Mixer
            hidden: (B, L, D) post-norm (and post-modulation) block hidden states
            init_conv: (B, conv_dim, d_conv-1) context conv state, or None
            init_ssm:  (B, nheads, headdim, d_state) context SSM state, or None
            return_states: also return the updated (conv_state[width d_conv], ssm_state)
                so the caller can advance a KV/Mamba cache (used by context extend).

        Returns: (B, L, D) mixer output (before adaLN gate / residual);
                 or (output, new_conv_state, new_ssm_state) if return_states.
        """
        from einops import rearrange
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        from causal_conv1d import causal_conv1d_fn

        d_inner = mixer.intermediate_size
        ngroups = mixer.n_groups
        d_state = mixer.ssm_state_size
        headdim = mixer.head_dim
        conv_dim = mixer.conv_dim
        d_conv = mixer.conv_kernel_size

        proj = mixer.in_proj(hidden)                       # (B, L, d_inner+conv_dim+nheads)
        z, xBC, dt = torch.split(proj, [d_inner, conv_dim, mixer.num_heads], dim=-1)

        # causal_conv1d_fn with initial_states requires channel-last layout:
        #  - input (B, conv_dim, L): use the transpose VIEW (stride(1)==1), no .contiguous()
        #  - initial_states (B, conv_dim, d_conv-1): force channel-last via the
        #    transpose->contiguous->transpose trick (mcore _run_denoiser_step).
        if init_conv is not None:
            init_conv = init_conv.transpose(-1, -2).contiguous().transpose(-1, -2)
        xBC_conv = causal_conv1d_fn(
            xBC.transpose(1, 2),                           # (B, conv_dim, L) channel-last view
            mixer.conv1d.weight.squeeze(1),
            mixer.conv1d.bias,
            activation=mixer.activation,
            initial_states=init_conv,
        ).transpose(1, 2)                                  # (B, L, conv_dim)

        x, B_proj, C_proj = torch.split(
            xBC_conv, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1
        )
        x = rearrange(x, "b s (h p) -> b s h p", p=headdim).contiguous()
        B_proj = rearrange(B_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
        C_proj = rearrange(C_proj, "b s (g n) -> b s g n", n=d_state).contiguous()

        # Run the SSM scan in fp32. With a long context the seeded SSM state gets
        # large (O(1e3)+); the bf16 chunk-scan then overflows to NaN, and because
        # the Triton kernel's reductions are not bit-deterministic this strikes
        # nondeterministically (a NaN on a block's first/all-masked step makes
        # every confidence NaN and force-commits an arbitrary token).
        # The scan spans only one block (<=16 tokens) so fp32 is essentially free,
        # and it is strictly more accurate. Cast back before the gated norm.
        _y_dtype = z.dtype
        A = -torch.exp(mixer.A_log.float())
        scan = mamba_chunk_scan_combined(
            x.float(), dt.float().contiguous(), A, B_proj.float(), C_proj.float(),
            mixer.chunk_size,
            D=mixer.D.float(), z=None,
            dt_bias=mixer.dt_bias.float(), dt_softplus=True,
            initial_states=(init_ssm.float() if init_ssm is not None else None),
            return_final_states=return_states,
        )
        if return_states:
            y, new_ssm = scan
        else:
            y = scan
        y = rearrange(y, "b s h p -> b s (h p)").to(_y_dtype)
        y = mixer.norm(y, z)                               # Mamba2 z-gated RMSNorm
        out = mixer.out_proj(y)
        if not return_states:
            return out
        # New conv state: HF cache stores the last d_conv raw xBC inputs (width
        # d_conv), most-recent at index -1. block_size >= d_conv here.
        L = xBC.shape[1]
        if L >= d_conv:
            new_conv = xBC[:, -d_conv:, :].transpose(1, 2).contiguous()
        else:
            hist = init_conv if init_conv is not None else xBC.new_zeros(xBC.shape[0], conv_dim, d_conv - 1)
            comb = torch.cat([hist.transpose(1, 2), xBC], dim=1)
            new_conv = comb[:, -d_conv:, :].transpose(1, 2).contiguous()
        return out, new_conv, new_ssm

    def _run_denoiser_step_diffusion(self, block_ids, cache_state, t=None, den_cache=None):
        """Diffusion denoiser forward over the FULL block (B, L) in one pass.

        Parity with mcore `_run_denoiser_step`:
          - Attention layers run BIDIRECTIONALLY within the block, attending to
            the full context KV cache + the whole noisy block (is_causal=False).
            A token-by-token causal pass would hide later block positions from
            earlier ones.
          - Mamba layers are causal/forward-only (bidirectional_mamba=False) and
            are chunk-scanned over the whole block from the context state (S-1),
            matching mcore's `forward_mamba_layer_with_states`.
          - Time conditioning (adaLN-single) is applied per layer. The modulate/norm
            ORDER depends on where mcore's norm lives: mamba & attention norms are
            FUSED into in_proj/linear_qkv (applied AFTER modulate) -> modulate THEN
            norm; MoE uses a separate pre_mlp_layernorm -> norm THEN modulate.
            Gate is applied to the mixer output in all cases.

        Args:
            block_ids: (B, L) tokens to denoise
            cache_state: context cache state
            t: (B,) timestep in [0,1], or None

        Returns: logits (B, L, V)
        """
        ctx_len = cache_state["ctx_len"]
        tower = self.denoiser_tower
        den_device = next(tower.parameters()).device
        den_input = block_ids.to(den_device)
        L = den_input.shape[1]

        # Time embedding -> per-layer modulation params (shift, scale, gate).
        t_emb = None
        if t is not None:
            t_dev = t.to(device=den_device, dtype=self.dtype)
            t_repr = self.t_embedder(t_dev)
            t_emb = self.t_block(t_repr)

        # Denoiser cache (context Mamba S-1 state + full context KV). It is
        # READ-ONLY here and identical for every step within a block, so the
        # caller should build it once per block and pass it in (avoids cloning +
        # cuda:0->cuda:1 copying the whole context cache on every NFE). Fall back
        # to building it if not provided.
        if den_cache is None:
            den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)

        hidden = tower.embeddings(den_input)

        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            mod = None
            if t_emb is not None:
                mod = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
                shift, scale, gate = mod

            # adaLN modulate vs norm ORDER depends on where mcore's norm lives:
            #   - mamba/attention: norm is FUSED into in_proj/linear_qkv and is
            #     applied AFTER the explicit modulate  -> modulate THEN norm.
            #   - moe/mlp: separate pre_mlp_layernorm applied BEFORE modulate
            #     -> norm THEN modulate.
            if block.block_type in ("mamba", "attention"):
                h = hidden
                if mod is not None:
                    h = _modulate(h, shift, scale)
                h = block.norm(h.to(dtype=block.norm.weight.dtype))
            else:  # mlp / moe
                h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
                if mod is not None:
                    h = _modulate(h, shift, scale)

            if block.block_type == "mamba":
                # Chunk-scan the whole block in one kernel launch, seeded from the
                # context Mamba state (matches mcore forward_mamba_layer_with_states).
                # HF conv_states are width d_conv; causal_conv1d_fn's initial_states
                # wants the d_conv-1 most-recent columns.
                d_conv = block.mixer.conv_kernel_size
                init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
                init_ssm = den_cache.ssm_states[layer_idx].contiguous()
                h = self._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
            elif block.block_type == "attention":
                ctx_k = den_cache.key_cache[layer_idx]
                ctx_v = den_cache.value_cache[layer_idx]
                h = self._denoiser_block_attention(block.mixer, h, ctx_k, ctx_v)
            elif block.block_type in ["mlp", "moe"]:
                h = block.mixer(h)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")

            if mod is not None:
                h = gate.unsqueeze(1) * h

            hidden = residual + h

        hidden = tower.norm_f(hidden)
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
        return logits

    # ------------------------------------------------------------------
    # Context-tower AR generation (single-tower baseline, cached)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_ar(self, input_ids, max_new_tokens=128, temperature=0.0,
                    top_k=None, top_p=None, eos_token_id=None):
        """Single-tower AR using ONLY the context tower, cached, 1 token/step.

        Equivalent to the stock single-tower model's greedy AR (the context tower
        is the frozen base), but routed through our own KV/Mamba cache machinery
        (single-step decode) — so it's O(N) cached and avoids HF generate()'s
        cache path that crashes on this env. This is the fair ST-AR baseline.
        """
        cache_state = self._build_context_cache(input_ids)
        logits = cache_state["logits"][:, -1, :].float()
        generated: List[torch.Tensor] = []

        for step in range(max_new_tokens):
            tok = self._sample_token(logits, temperature, top_k, top_p)
            generated.append(tok)
            if eos_token_id is not None and (tok == eos_token_id).any():
                break
            cache_state = self._extend_context_cache(tok, cache_state, block_wise=False)
            logits = cache_state["logits"][:, -1, :].float()

        return torch.cat([input_ids] + [g.to(input_ids.device) for g in generated], dim=1)

    # ------------------------------------------------------------------
    # Mock-AR generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_mock_ar(self, input_ids, max_new_tokens=128, temperature=0.0,
                         top_k=None, top_p=None, eos_token_id=None):
        """Two-tower mock-AR: S-2/KV[:-1] cache, 1 token/step."""
        B = input_ids.shape[0]
        generated: List[torch.Tensor] = []
        cache_state = self._build_context_cache(input_ids)

        for step in range(max_new_tokens):
            last_token = input_ids[:, -1:] if step == 0 else generated[-1]
            logits = self._run_denoiser_step_mock_ar(last_token, cache_state)
            logits = logits[:, -1, :].float()
            tok = self._sample_token(logits, temperature, top_k, top_p)
            generated.append(tok)
            if eos_token_id is not None and (tok == eos_token_id).any():
                break
            # Single-step context extension (stock kernels) so mock-AR matches stock.
            cache_state = self._extend_context_cache(tok, cache_state, block_wise=False)

        return torch.cat([input_ids] + [g.to(input_ids.device) for g in generated], dim=1)

    # ------------------------------------------------------------------
    # Mask-Diffusion generation
    # ------------------------------------------------------------------

    @staticmethod
    def _mdlm_forward(logits, xt, mask_token_id):
        """Constrain logits -> p(x0|xt): mask token gets -inf, decoded tokens
        get delta on their current value."""
        logits = logits.clone()
        logits[..., mask_token_id] = -1e12
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Fix unmasked positions: they must predict themselves with prob 1
        unmasked = (xt != mask_token_id)
        if unmasked.any():
            log_probs[unmasked] = -1e12
            log_probs[unmasked, :].scatter_(-1, xt[unmasked].unsqueeze(-1), 0.0)
        return log_probs

    @staticmethod
    def _gumbel_sample(log_probs):
        """Gumbel-max sampling from log probabilities."""
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(log_probs).clamp(min=1e-10)
        ))
        return (log_probs + gumbel_noise).argmax(dim=-1)

    @torch.no_grad()
    def generate_mask_diffusion(
        self,
        input_ids,
        max_new_tokens=128,
        block_size=16,
        steps_per_block=16,
        mask_token_id=3,
        temperature=0.0,
        top_k=None,
        confidence_threshold=0.9,
        eos_token_id=None,
        step_callback=None,
    ):
        """Block-wise mask diffusion with confidence_unmasking.

        Algorithm:
          1. Build context cache from prompt
          2. For each block:
             a. Init block_ids = all mask tokens
             b. For each denoising step:
                - Compute t_model = fraction of masked positions
                - Denoiser forward -> logits -> p(x0|xt) via _mdlm_forward
                - Predict tokens (greedy or gumbel)
                - Confidence = p(predicted|xt) from unscaled probs
                - Commit high-confidence predictions, remask low-confidence
             c. Extend context cache with final block
          3. Return full sequence

        Args:
            input_ids: (B, S) prompt
            max_new_tokens: total tokens to generate (must be divisible by block_size)
            block_size: tokens per diffusion block
            steps_per_block: denoising iterations per block
            mask_token_id: ID of the [MASK] token
            temperature: 0 = greedy argmax, >0 = gumbel sampling
            top_k: unused currently (kept for API compat)
            confidence_threshold: commit tokens above this confidence
            eos_token_id: stop on EOS

        Returns: (B, S + generated) full token sequence
        """
        B = input_ids.shape[0]
        device = input_ids.device
        assert max_new_tokens % block_size == 0, \
            f"max_new_tokens ({max_new_tokens}) must be divisible by block_size ({block_size})"
        num_blocks = max_new_tokens // block_size

        cache_state = self._build_context_cache(input_ids)
        context_ids = input_ids.clone()
        nfe = 0  # number of denoiser forward passes (network function evaluations)

        den_device = next(self.denoiser_tower.parameters()).device
        for block_idx in range(num_blocks):
            # Build the denoiser cache ONCE per block (context is fixed within a
            # block); reused by every denoising step to avoid per-NFE clone+copy.
            den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)

            # Initialize fully masked block
            xt = torch.full((B, block_size), mask_token_id, dtype=torch.long,
                            device=device)
            if step_callback is not None:
                step_callback(0, steps_per_block, xt, t=1.0, logits=None,
                              block_idx=block_idx)

            for step_idx in range(steps_per_block):
                # t_model = current mask fraction
                is_masked = (xt == mask_token_id)
                n_masked = is_masked.float().sum(-1).mean().item()
                if n_masked == 0:
                    break
                t_model = is_masked.float().mean()
                t_vec = t_model.expand(B).to(device)

                # Denoiser forward (logits come back on denoiser device, move to xt's device)
                logits = self._run_denoiser_step_diffusion(xt, cache_state, t=t_vec, den_cache=den_cache)
                nfe += 1
                logits = logits.to(device)

                # p(x0|xt) with constraints
                log_x_theta = self._mdlm_forward(logits, xt, mask_token_id)
                x_theta = log_x_theta.exp()

                # Predict: greedy or gumbel
                if temperature <= 0:
                    predicted = log_x_theta.argmax(dim=-1)
                else:
                    scaled_logits = logits.clone()
                    scaled_logits[..., mask_token_id] = -1e12
                    scaled_log = scaled_logits / temperature - torch.logsumexp(
                        scaled_logits / temperature, dim=-1, keepdim=True)
                    unmasked = (xt != mask_token_id)
                    if unmasked.any():
                        scaled_log[unmasked] = -1e12
                        scaled_log[unmasked, :].scatter_(-1, xt[unmasked].unsqueeze(-1), 0.0)
                    predicted = self._gumbel_sample(scaled_log)

                # Confidence from unscaled x_theta
                confidence = x_theta.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                confidence[~is_masked] = float('inf')

                # Determine how many to commit
                is_last_step = (step_idx == steps_per_block - 1)
                n_masked_int = is_masked.sum(-1)  # (B,)

                if is_last_step:
                    tokens_to_commit = n_masked_int
                else:
                    # Per-batch commitment logic (simplified for B=1 common case)
                    remaining_steps = max(1, steps_per_block - step_idx)
                    num_above = ((confidence > confidence_threshold) & is_masked).sum(-1)
                    tokens_to_commit = torch.where(
                        num_above > 0, num_above,
                        torch.ones_like(num_above),
                    )
                    min_commit = (n_masked_int.float() / remaining_steps).ceil().long()
                    tokens_to_commit = torch.clamp(
                        torch.max(tokens_to_commit, min_commit),
                        max=n_masked_int,
                    )

                # Apply predictions then remask low-confidence
                output = torch.where(is_masked, predicted, xt)
                num_to_remask = n_masked_int - tokens_to_commit  # (B,)

                for b in range(B):
                    if num_to_remask[b] > 0:
                        masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                        masked_conf = confidence[b, masked_indices]
                        _, sort_idx = masked_conf.sort()
                        remask_idx = masked_indices[sort_idx[:num_to_remask[b]]]
                        output[b, remask_idx] = mask_token_id

                if step_callback is not None:
                    step_callback(step_idx, steps_per_block, xt,
                                  t=float(t_model.detach().cpu()), logits=logits,
                                  block_idx=block_idx)

                xt = output

            # Block complete — extend context
            context_ids = torch.cat([context_ids, xt], dim=1)
            cache_state = self._extend_context_cache(xt, cache_state)

            if eos_token_id is not None and (xt == eos_token_id).any():
                break

        # Expose NFE (denoiser forward passes) for reporting, e.g. inference.py.
        self._last_nfe = nfe
        return context_ids

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_token(logits, temperature, top_k, top_p):
        if temperature is None or temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        if top_k is not None and top_k > 0:
            kth = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1).values[..., -1:]
            probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_p, idx = torch.sort(probs, descending=True, dim=-1)
            cum = sorted_p.cumsum(dim=-1)
            remove = torch.cat(
                [torch.zeros_like(cum[..., :1]), (cum > top_p)[..., :-1]], dim=-1,
            )
            sorted_p = sorted_p.masked_fill(remove.bool(), 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Multi-GPU placement
    # ------------------------------------------------------------------

    def place_towers_on_devices(self, ctx_device="cuda:0", den_device="cuda:1"):
        """Manual tower placement. Time conditioning goes with denoiser."""
        self.context_tower = self.context_tower.to(ctx_device)
        self.context_lm_head = self.context_lm_head.to(ctx_device)
        self.denoiser_tower = self.denoiser_tower.to(den_device)
        self.lm_head = self.lm_head.to(den_device)
        self.t_embedder = self.t_embedder.to(den_device)
        self.t_block = self.t_block.to(den_device)
        self.scale_shift_tables = nn.ParameterList([
            nn.Parameter(p.to(den_device)) for p in self.scale_shift_tables
        ])
        return self
