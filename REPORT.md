# Nemotron-Labs-TwoTower 复现精读报告(骨架)

> 按七步框架搭好,已填入核实过的事实;`[DATA:…]` 是等实验跑完填的数。
> 核心亮点在 **§5 代码精读 & 复现**(找到并修了官方 kernel bug)。

---

## Step 1 — 版本
- 论文:*Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context*
  (Fitsum Reda, John Kamalu, Roger Waleffe, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro; NVIDIA)。arXiv 2606.26493。
- 两个 HF 仓库:`nvidia/Nemotron-TwoTower-30B-A3B-Base-BF16`(旧名,paper.pdf 挂这)与
  `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`(2 天前改名,加 "-Labs")。同一 base 模型。
- **没有 instruct/aligned 版**;第三方只有 MLX 量化。选讲:`-Labs` 版(最新)。

## Step 2 — 周边材料
- 代码:`modeling_nemotron_twotower.py`(962 行)、`modeling_nemotron_h.py`(骨干)、
  `configuration_nemotron_h.py`、**`inference.py`(184 行官方驱动)**。
- 权重:24 shard,BF16,~118GB,`model.safetensors.index.json`。
- 文档:README、bias/safety/privacy/explainability.md、`table_comparison.png`。
- HF 社区讨论 3 个(只有 1 个 CPU-path bug,与 GPU 无关)。
- 依赖:transformers 4.57.1、mamba_ssm、causal_conv1d、einops。硬件:2×A100/H100-80GB。

## Step 3 — 背景知识(帮同事快速对齐)
- **Mamba-2 / SSD**:线性状态空间递归,`state_t = decay·state_{t-1}+B·x`;chunk-scan 是它的硬件并行算法。
- **MoE**:128 routed experts,每 token 激活 6 + 1 shared;router 跑 fp64。
- **Mask diffusion(块扩散)**:块间自回归、块内并行去噪;置信度解码逐步 commit 高置信位、remask 低置信位。
- **Two-Tower**:冻结的 context 塔(复用 AR backbone)+ 可训练的 denoiser 塔,逐层协作。
- **AdaLN-single**(PixArt 风格):把当前 mask 比例 t 注入每层 scale/shift/gate。

## Step 4 — 数学 / 机制
- 去噪塔读上下文有**两条通道**:(A) cross-attention 拼接 context KV;(B) Mamba 用 context 的最终状态**播种**(initial_states)。
- 块内:attention 双向(`is_causal=False`),Mamba 因果/前向。
- 时间条件:`t_scaled = t·1000` → 正弦嵌入 → MLP → 每层调制。
- [DATA: adaLN / seeding / remask 消融对机制贡献的量化]

## Step 5 — 代码精读 & 复现(★核心)
### 架构参数(纯 config 算术,零 GPU)
- 每塔 52 层,pattern = **23 Mamba-2 / 23 MoE / 6 attention**。hidden 2688,vocab 131072。
- MoE:128 routed(激活6)+ 1 shared,intermediate 1856 / shared 3712。

| 模块(单塔) | 参数 |
|---|---|
| mamba2 ×23 | 0.891B |
| moe ×23(total) | 29.842B |
| attention ×6 | 0.140B |
| embed + lm_head | 0.705B |
| **单塔 total / activated** | **31.58B / 3.58B** |
| **双塔 total** | **63.2B** |

→ 参数几乎全在 MoE;30B-A3B 的账:总 63B、每 token 激活 ~3.6B/塔。

### 复现踩坑(见 TROUBLESHOOTING.md)
环境 8 坑全趟平并写进脚本:wheel 四元组、`--no-deps` 锁 torch(否则 torch 被升到 2.13)、
别升 triton(libcudart.so.13)、env 装本地盘(网络卷 pip 卡死)、Network Volume 持久、
MooseFS 残留锁、HF_TOKEN 非 ASCII、mamba_ssm 需 einops。

### ★ 找到并修复官方 diffusion bug
- **现象**:`generate_ar` 完美(149-token 数学题答对 39),但 `generate_mask_diffusion` /
  官方 `generate_mock_ar` 输出**词沙拉**,NFE 恒满、tokens/NFE=1.0(零并行)。
- **系统排查**(逐一排除):环境 / 权重加载 / cross-attn KV 送达 / Mamba 播种 / AdaLN /
  RoPE / attention 实现(sdpa vs eager)/ prompt 强度 / 采样 / t 方向 / cache 复用。
- **定位**:AR 用 `selective_state_update`(无 initial_states)→ 正常;去噪塔用
  `mamba_chunk_scan_combined` + `initial_states` → 垃圾。唯一 kernel 差异。
- **根因**:Mamba2 SSD kernel bug(**vLLM PR #21783**)——当 `block_size(16) < chunk_size(128)`,
  initial_states 的 decay 缩放算错、播种态影响被过度放大 → 去噪塌缩。
- **修复**:把 chunk-scan 换成**逐 token `selective_state_update`**(AR 用的正常 kernel),
  数学等价(去噪 Mamba 本就因果/前向)。**tokens/NFE 从 1.0(坏)→ 2.37+(修好)**,答案变正确。
- **为什么官方没发现**:内部用 mcore(不同 kernel),转 HF + 开源 mamba_ssm 才触发。
- **"还是扩散吗"**:是。并行在(1)块内双向 attention +(2)去噪循环一次预测全 16 位;
  Mamba 本是顺序递归,chunk-scan 与逐 token 只是两种实现。仅 wall-clock 变慢,NFE 不变。

## Step 6 — 实验 & 数据(我自己的)
> 全部用修好的模型;速度用 NFE/tokens-per-NFE 口径(硬件无关)。
- **并行度/速度**:tokens/NFE ≈ [DATA: 2.88~5.02,均值 ~3.5];NFE≪256(块提前收敛)。
- **动态发现**:steps/block 递减——**越后面的块收敛越快**(2-3 步),早期块 7-17 步。原创观察。
- **消融1 remask**(王牌):baseline vs disable_remask 的 质量 + NFE [DATA]。
  判据:关掉后质量不掉、NFE 更低 → 支持"加速来自块并行、非迭代精修"。
- **消融2/3/4**:去播种 / 固定 AdaLN t / top-k(4/6/8) [DATA]。
- **崩溃复现**:采样 block 8/16/32/64 的质量 [DATA:预期 >16 断崖]。
- **质量-速度 Pareto**:γ×T 扫描 [DATA]。
- **可视化**:生成过程 GIF(左上三角 + 块内 remask)+ before/after(修复前垃圾 vs 修复后)。

## Step 7 — topic 关系 / gap / 反思
- **关系**:相关线——LLaDA、block diffusion、Mamba-2 SSD、DiT/adaLN、MoE 路由。
- **研究 gap / 方向**:(a) 向量化正确 kernel(逐 token 慢);(b) 推理侧消融量化各机制贡献;
  (c) 跨步 MoE 路由稳定性(扩散独有);(d) base→instruct 后 diffusion 质量。
- **贡献**:可给 NVIDIA/HF 报这个 SSD kernel bug(最小复现:AR 正常 / diffusion 垃圾 + block<chunk)。
- **反思 / 避雷**:系统性二分排查 > 盲猜;环境坑全固化进脚本;权重放持久卷、env 装本地;
  单点 OK/多点崩 = 找状态污染或 kernel;拿"已知答案"当探针最快定位。
