# TwoTower Loss and Layer-Similarity Discussion

This note summarizes two follow-up discussion threads for the Nemotron-Labs
TwoTower presentation:

1. how to interpret the masked-diffusion loss used by TwoTower, and
2. how to probe whether the denoising tower develops diffusion-style layer
   redundancy or remains close to an AR retrofit.

It is intended as presenter preparation material, not as a completed
experimental result.

## 1. Loss Function

TwoTower trains a denoising tower conditioned on a frozen AR context tower. For
each training example, the clean sequence is split into a clean context and a
target block:

```text
x = [context tokens | target block]
context = x_<b
target = x_b
```

The context tower reads the clean context and produces conditioning signals for
the denoising tower:

```text
clean context
  -> frozen AR context tower
      -> attention KV cache
      -> Mamba final states
```

The target block is randomly corrupted according to a sampled timestep or mask
ratio `t`:

```text
clean block:  [A, B, C, D, ...]
noisy block:  [A, MASK, C, MASK, ...]
```

The denoising tower receives:

```text
noisy block x_t
+ context attention KV
+ context Mamba states
+ timestep / mask ratio t
```

It predicts the original clean tokens in the target block. The loss is computed
on the masked positions:

```text
L = mean_{i: x_t[i] is MASK} -log p_theta(x_0[i] | x_t, context, t)
```

This is best described as a masked diffusion / MDLM-style denoising objective.
However, the paper makes an important engineering choice: the theoretical
`1/t` weighting term is omitted for stability. This weakens the connection to a
strict diffusion likelihood or ELBO objective and makes the practical training
objective closer to stable masked-token denoising conditioned on a strong AR
context.

This matters for the interpretation of the inference ablations. The model is
trained with time conditioning through AdaLN, but if freezing the real `t` at
inference has little or no effect, then the effective mechanism may be less
"time-conditioned diffusion refinement" and more:

```text
AR context conditioning
+ block-level parallel drafting
+ confidence-based iterative correction
```

The careful statement is not that TwoTower lacks a diffusion objective. It is
that the implemented objective and observed inference behavior both suggest a
pragmatic masked-denoising retrofit rather than a strong replacement of AR by a
fully diffusion-driven generation process.

## 2. Training vs Inference Data Flow

Training and inference share the same conditioning structure, but differ in how
the noisy block is obtained and how often the denoising tower is evaluated.

During training:

```text
ground-truth block
  -> random mask pattern sampled from t
  -> one denoising forward
  -> cross-entropy on masked positions
```

During inference:

```text
all-MASK block
  -> repeated denoising forwards
  -> confidence-based commit/remask
  -> completed block appended to context
```

The inference-only quantities are:

- `gamma`, the confidence threshold,
- `min_commit`, the progress lower bound,
- forced final-step commit, and
- NFE / tokens-per-NFE.

These are sampler mechanics rather than training loss terms.

## 3. AdaLN Time Conditioning

In the Hugging Face TwoTower code, the timestep used at inference is the current
mask fraction of the block, not simply the loop index:

```python
is_masked = (xt == mask_token_id)
t_model = is_masked.float().mean()
t_vec = t_model.expand(B)
```

For a block of 16 tokens:

```text
16 masks left -> t = 1.0
12 masks left -> t = 0.75
8 masks left  -> t = 0.5
1 mask left   -> t = 0.0625
```

The scalar `t` is passed through a sinusoidal timestep embedder and an MLP:

```text
t
  -> TimestepEmbedder
  -> t_block MLP
  -> 3 * hidden_size
```

Each layer receives shift, scale, and gate vectors:

```text
shift, scale, gate = f_layer(t)
```

The modulation has the form:

```text
h = h * (1 + scale) + shift
out = gate * mixer(h)
hidden = residual + out
```

The implementation applies the modulation in slightly different order depending
on layer type, matching Megatron-Core behavior:

```text
Mamba / attention: modulate -> norm -> mixer -> gate -> residual
MLP / MoE:         norm -> modulate -> mixer -> gate -> residual
```

The `freeze_time` ablation fixes the `t` passed into this AdaLN path while
leaving context conditioning, Mamba seeding, and commit/remask unchanged.

## 4. Layer-Similarity Motivation

The main open question from the layer-similarity direction is:

```text
Does TwoTower's denoising tower develop diffusion-style representational
redundancy, or does it remain close to an AR-initialized retrofit?
```

This question is motivated by arXiv:2603.07475, which compares layer-wise and
token-wise representational capacity in AR and diffusion LLMs. That work reports
that native diffusion LLMs exhibit more global representations, early-layer
redundancy, and lower recency bias, while AR-initialized diffusion LLMs retain
AR-like dynamics.

TwoTower is a natural follow-up case because it is explicitly an AR retrofit:

```text
context tower:   frozen pretrained AR backbone
denoising tower: separate tower trained for masked block denoising
```

If the denoising tower has strong diffusion-style redundancy, then TwoTower may
be more diffusion-like internally than its sampler behavior suggests. If it
does not, that would support the interpretation that TwoTower is mainly a
block-parallel AR retrofit.

## 5. Candidate Similarity Probes

### 5.1 Adjacent-Layer Similarity

Measure cosine similarity between adjacent layer hidden states:

```text
cos(h_l, h_{l+1})
```

Compare:

```text
context_tower on clean context
denoising_tower on masked / partially denoised block
```

Interpretation:

- higher adjacent-layer cosine suggests stronger layer redundancy,
- a denoising tower curve closer to native diffusion LLMs would weaken the
  "pure AR retrofit" interpretation,
- a denoising tower curve close to the context tower would strengthen it.

### 5.2 Token-Wise Similarity

Measure cosine similarity between neighboring token representations within each
layer:

```text
cos(h_l[token_i], h_l[token_{i+1}])
```

This can test whether representations are local and recency-biased, as in AR
models, or more globally smoothed, as reported for diffusion LLMs.

### 5.3 Context-vs-Denoiser Same-Layer Similarity

Compare same-layer activations between the two towers:

```text
cos(h_context_layer_i, h_denoiser_layer_i)
```

This requires care because the towers normally receive different inputs:

- context tower receives clean context,
- denoising tower receives a noisy block conditioned on context.

Possible setups:

1. feed the same clean tokens to both towers to measure structural similarity,
2. compare the real inference setting to measure functional divergence.

### 5.4 Cross-Step Denoising Similarity

For the same block and same position, compare hidden states across denoising
steps:

```text
cos(h_layer_step_t, h_layer_step_{t+1})
```

This is especially relevant for TwoTower because it has an iterative sampler.
High cross-step similarity would suggest that denoising steps make small
corrections; low similarity would suggest more substantial representational
changes across iterations.

### 5.5 MoE Routing Similarity

For MoE layers, compare expert selections:

```text
overlap(experts_step_t, experts_step_{t+1})
```

This complements activation similarity. If hidden states are stable but routing
churns, the denoising dynamics may be expressed through expert selection rather
than large hidden-state movement.

## 6. Adapting Existing Nemotron-Diffusion Code

The Nemotron-Labs-Diffusion probing script shared by Bodan Liu measures:

- token-wise cosine curves,
- adjacent-layer cosine curves,
- generated-token-only curves, and
- AR vs diffusion mode by toggling an attention flag.

That script assumes a single model structure:

```python
model.encoder.layers
model.encoder(input_ids=x, use_causal_mask=causal)
attn.diffusion_lm = True / False
```

TwoTower has a different structure:

```python
model.context_tower.layers
model.denoiser_tower.layers
model._run_denoiser_step_diffusion(...)
model.generate_mask_diffusion(...)
```

Therefore, the script cannot be copied directly. A TwoTower version should
capture hidden states separately from:

```text
context_tower
denoising_tower
```

and, ideally, at multiple denoising steps.

## 7. Suggested Main Discussion Question

The strongest way to frame this in the presentation is:

```text
If diffusion objectives induce layer redundancy, does TwoTower's denoising tower
show that redundancy, or does the AR-initialized/frozen-context design keep it
AR-like?
```

Possible outcomes:

```text
Denoiser shows diffusion-like redundancy:
    TwoTower may learn internal diffusion-style representations even though its
    sampler remains strongly AR-structured.

Denoiser remains close to context tower / AR curves:
    Supports the view that TwoTower is primarily a block-parallel AR retrofit.

Only specific modules diverge:
    The diffusion-specific behavior may live in time conditioning, cross-attn,
    MoE routing, or Mamba state usage rather than uniformly across all layers.
```

This is a good discussion item because it connects the loss function, inference
ablations, and the broader AR-vs-diffusion representation question without
requiring new experiments for the current report.
