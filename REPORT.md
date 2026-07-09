# Nemotron-Labs-TwoTower 深度精读报告

> 用途:组内 paper 精读范例(30-45 min)。展示:我做了什么、发现了什么、怎么解决的。
> `[DATA:…]` = 等本机实验跑完填(脚本在 `run_everything.sh`,数据在 `results/`)。
> **★ = 原创发现 / 亮点**(非"和 GPT 聊几句"能得到的)。

---

## Step 1 — 版本
- **论文**:*Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context*。
  arXiv **2606.26493**(有 v1、v2;abstract 与 §2.3 的 token 数不一致:2.1T vs 1.4T —— 一个待澄清点)。
  也在 alphaXiv 有页面。**没有 OpenReview**(arXiv-only research release)。
- **两个 HF 仓库**:`nvidia/Nemotron-TwoTower-...`(旧名,paper.pdf/inference.py 挂这)与
  `nvidia/Nemotron-Labs-TwoTower-...`(改名加 "-Labs",2 天前)。**权重、代码相同**。
- **选讲**:`-Labs` 版(最新)。**无 instruct/aligned 版**。

## Step 2 — 周边材料(找到 / 没找到)
| 类别 | 结果 |
|---|---|
| 代码 | ✅ modeling_nemotron_twotower.py(962行)/ modeling_nemotron_h.py / configuration / **inference.py(184行官方驱动)** |
| 权重 | ✅ 24 shard, BF16, ~118GB |
| 论文 | ✅ arXiv HTML + paper.pdf |
| 媒体 | ✅ MarkTechPost、TechTimes("2.42× 且无需重新预训练"是卖点) |
| Twitter/X | ✅ Tanishq Abraham(@iScienceLuvr)、@yesnoerror("~8% 数据即可升级现有模型") |
| 顶会评审 | ❌ 无 OpenReview |
| patent | ❌ 未找到 |
| 作者前作 | ✅ 见下 |
| HF 社区 | ✅ 3 讨论(仅 1 个 CPU-path bug,与 GPU 无关) |

**作者 & 前作(串起 lineage)**:Bryan Catanzaro / Mohammad Shoeybi / Mostofa Patwary = **Megatron-LM / Nemotron** 核心;
Roger Waleffe = *An Empirical Study of Mamba-based LMs*(NVIDIA 混合 Mamba,正是本文 backbone 的血脉);
Fitsum Reda = 视频/扩散背景(FILM 帧插值)——**把扩散经验带进 LLM**;John Kamalu = NVIDIA。

## Step 3 — 背景知识(帮同事对齐)
- **Mamba-2 / SSD**:线性状态空间递归 `state_t = decay_t·state_{t-1} + B_t·x_t`;chunk-scan 是它的硬件并行算法(与本文 bug 直接相关)。
- **MoE**:128 routed experts / token 激活 6 + 1 shared;router 跑 fp64(边界 top-k 一致)。
- **Mask diffusion(MDLM)**:把序列位置随机 mask、模型学去噪;推理时置信度解码逐步 commit。
- **块扩散(block diffusion)**:块间自回归、块内并行去噪 —— 兼顾 AR 的质量与扩散的并行。
- **Two-Tower**:冻结 context 塔(=AR backbone)+ 可训练 denoiser 塔,逐层 cross-attn 协作。
- **AdaLN-single**(PixArt 风格):把 mask 比例 t 注入每层 scale/shift/gate。

## Step 4 — 数学 / 机制
- **前向腐蚀**:线性 schedule `α_t = 1 − t`(t=mask 比例)。
- **损失(Eq.4)**:masked diffusion NLL。理论时间权重 `1/t` **被省略"for stability"**,改优化 masked 位置的平均负对数似然。
  → ★ 这是个可讨论的"assumption 未满足"点:省掉 `1/t` 破坏了 ELBO 的严格性,换来训练稳定;是工程 vs 理论的取舍。
- **块自回归分解**:`log pθ(x) = Σ_b log pθ(x_b | x_<b)`。
- **置信度解码**:每步预测所有 mask 位 + 置信度,>γ 的 commit,低的 remask(每步≥1 保底,末步全收)。
- **两条上下文通道**:(A) cross-attn 拼 context KV;(B) Mamba 用 context 最终状态**播种**(initial_states)。
- [DATA: adaLN / seeding / remask 消融的机制贡献量化 → 见 Step 6]

## Step 5 — 代码精读 & 复现(★核心)

### 5.1 架构 & 逐模块参数(纯 config 算术,零 GPU)
- Backbone = **Nemotron-3-Nano-30B-A3B**(两塔都从它初始化)。每塔 52 层:
  pattern **23 Mamba-2 / 23 MoE / 6 attention**,hidden 2688,vocab 131072。

| 模块(单塔) | 参数 | 占比 |
|---|---|---|
| mamba2 ×23 | 0.891B | 2.8% |
| **moe ×23 (total)** | **29.842B** | **94.5%** |
| attention ×6 | 0.140B | 0.4% |
| embed + lm_head | 0.705B | 2.2% |
| **单塔 total / activated** | **31.58B / 3.58B** | — |
| **双塔 total** | **63.2B** | — |

★ 参数几乎全在 MoE(94.5%);30B-A3B = 总 ~31.6B/塔、每 token 只激活 ~3.58B。双塔 63.2B(对上 HF 卡的 63B)。

### 5.2 训练手法(论文 §2.3)
- **只训 denoiser 塔;context 塔冻结**(no-grad,标准因果 AR mask 跑一遍供 KV/状态)。
- **数据**:两阶段课程(phase1 广度 → phase2 STEM/高质量),都取自 Nemotron-3-Nano 的 blend;
  **~1.4T tokens(§2.3)/ ~2.1T(abstract)**,而 backbone 预训练用了 **25T** →
  ★ **只用 backbone ~8% 的数据就加装了扩散能力**(这是"无需重预训练"卖点的量化)。
- **配方**:BF16、AdamW、WSD(warmup-stable-decay)LR,峰值 1e-4、末值 1e-6,阶段边界 reset。
  三阶段:phase1 S=32 → phase2 S=32 → phase2 **S=16**(最终)。
- **软硬协同**:训练在 **Megatron-LM**;adaLN 模块(1.5M 参数)**每个 TP rank 复制而非切分**(张量并行)。
  ★ **论文没给训练 GPU 数 / GPU-hours**(只说评测用 2×H100)——一个信息缺口。

### 5.3 ★★ 复现中发现并修复的官方 bug(pre 王牌)
**现象**:`generate_ar` 完美(149-token 数学题答对 39),但 `generate_mask_diffusion` /
官方 `generate_mock_ar` 输出**词沙拉**,NFE 恒满、tokens/NFE=1.0(零并行)。

**系统排查**(二分逐一排除,不盲猜):环境 / 权重加载 / cross-attn KV / Mamba 播种 / AdaLN /
RoPE / attention 实现(sdpa vs eager)/ prompt 强度 / 采样 / t 方向 / cache 复用 —— 全部排除。

**定位**:AR 用 `selective_state_update`(不传 initial_states)→ 正常;去噪塔用
`mamba_chunk_scan_combined` + `initial_states` → 垃圾。**唯一 kernel 差异**。

**根因(两层)**:
1. **config 层**:★ 论文明说"Mamba **chunk_size 匹配到 block size S=16**",但 HF `config.json` 里
   **`chunk_size=128`**(继承 backbone 长上下文默认值,转换时未对齐扩散块)。→ 造成 `block(16) < chunk(128)`。
2. **kernel 层**:该条件命中 Mamba2 SSD kernel bug(**vLLM PR #21783**)——block<chunk 时
   initial_states 的 decay 缩放算错、播种态被过度放大 → 去噪塌缩。

**修复**:把 chunk-scan 换成**逐 token `selective_state_update`**(AR 用的正常 kernel,等价于 chunk=1),
数学等价(去噪 Mamba 本就因果/前向)。**tokens/NFE 从 1.0(坏)→ 2.37+(修好)**,答案正确。
(注:仅把 chunk_size 改回 16 —— fixA —— 在开源 mamba_ssm 2.3.2 上**仍垃圾**,故彻底绕开 chunk-scan。)

**为什么官方没发现**:训练/推理在 mcore(chunk 匹配 S=16、且 mcore kernel 实现不同);
转成 HF + 开源 mamba_ssm 才同时踩中 config 未对齐 + 开源 kernel bug。→ 可给 NVIDIA/HF 报 issue(实打实贡献)。

**"这还是扩散吗?"** ★ 是。并行在(1)块内双向 attention +(2)去噪循环一次预测全 16 位、每步 commit 多个;
Mamba 本是**顺序递归**,chunk-scan 与逐 token 只是它的两种实现。仅 wall-clock 变慢,**NFE / 并行度不变**。

### 5.4 库 / custom / 改进空间(2026 视角)
- **现成库**:mamba_ssm(SSD kernel)、causal_conv1d、transformers、Megatron-LM。
- **custom**:两塔编排、cross-attn 拼接 context KV、Mamba 状态播种、AdaLN-single、置信度解码循环。
- **可精简/改进**:① config 应把 chunk_size 设为 block_size(修根因);② 逐 token 修复慢,可向量化正确 SSD 或用修好的 kernel;
  ③ 省掉的 `1/t` 权重可作为消融;④ MoE 占 94.5% 参数但每步激活 6/128 —— 推理时可调 top-k(见消融4)。

## Step 6 — 实验 & 数据

### 6.1 论文的关键结果(Tables 1-4,已核实)
- **Table 2 解耦**:tied+diffusion **−27.9/−27.8/−27.0**(Gen/Code/Math);解耦 TwoTower 仅 **−6.2/−10.5/−11.3**;
  continued-AR −10.5/−8.3/−17.8。→ ★ **解耦是关键**,绑定掉 26-28%。
- **Table 3 训练块**:S=16(phase2)= **77.10/74.56/85.45 @ 2.02×**;S=8 质量略高但仅 1.71×;S=32 phase2 2.25× 但质量低。→ **S=16 是甜点**。
- **Table 1 设计**:双向 Mamba"翻倍 SSM 算力却零/负收益"→ Mamba 保持因果;time conditioning(adaLN)+1~1.5;phase2 提升明显。
- **Table 4 采样块崩溃**(训练 S=16):采样 block **16→64**:HumanEval **76.40→19.85**、GSM8K **89.84→2.20**、MATH-500 **81.05→2.20**。
- 头条:**2.42× wall-clock、保留 98.7% AR 质量**;push 过 3× 质量明显下滑。

### 6.2 我自己的数据(修好的模型,NFE 口径硬件无关)
- **并行度**:tokens/NFE ≈ **[DATA:2.31~5.95,均值 ~3.3]**;NFE≪256(块提前收敛)。★ 我的加速比数据。
- ★ **动态发现1**:`steps/block` 递减 —— **越后面的块收敛越快**(2-3 步),早期块 7-17 步(后文有更多上下文)。
- ★ **动态发现2**:`nfe` 与题目难度正相关(简单题 nfe=43/tpn=5.95,难题 nfe=111/tpn=2.31)。
- **消融1 remask(王牌)**:baseline vs disable_remask 的质量+NFE [DATA]。判据:关掉后质量不掉、NFE 更低 → 支持"加速主来自块并行、非迭代精修"。
- **消融2/3/4**:去播种 / 固定 AdaLN t / top-k(4/6/8)[DATA]。
- **崩溃复现**:采样 block 8/16/32/64 [DATA:预期复现 Table 4 的断崖]。
- **Pareto**:γ×T 扫描 [DATA]。
- **可视化**:生成过程 GIF(左上三角 + 块内 remask)+ ★ **before/after**(修复前词沙拉 vs 修复后正确)。

## Step 7 — 关系 / gap / 反思
- **topic 关系**:LLaDA / block diffusion(块扩散谱系)、Mamba-2 SSD(架构)、DiT/PixArt adaLN(时间条件)、
  MDLM(sahoo/shi/ou 2024,损失)、Nemotron-3-Nano(backbone)、Mamba-in-Llama(Waleffe,前作)。
- **gap / 方向**:① config chunk_size 未对齐(已定位并修);② 向量化正确 SSD 修复(逐 token 慢);
  ③ 推理侧消融量化各机制贡献(论文只做训练侧);④ ★ 跨步 MoE 路由稳定性(扩散独有,AR 问不出);
  ⑤ base→instruct 后 diffusion 质量;⑥ 省掉的 `1/t` 权重的影响。
- **贡献**:向 NVIDIA/HF 报 SSD kernel + config chunk_size bug(最小复现:AR 正常/diffusion 垃圾 + block<chunk)。
- **反思 / 避雷**:★ 系统性二分排查 > 盲猜;"单点OK/多点崩"=状态污染或 kernel;拿"已知答案"当探针最快定位;
  环境 8 坑全固化进脚本(见 TROUBLESHOOTING.md):`--no-deps` 锁 torch、别升 triton、env 装本地盘、
  权重放 Network Volume、HF_TOKEN 别带非 ASCII。天生优势:肯逐层验证不跳步;习得能力:软硬件系统直觉。

---
### Sources
- 论文:https://arxiv.org/abs/2606.26493 · https://arxiv.org/html/2606.26493v2 · https://www.alphaxiv.org/abs/2606.26493
- 模型:https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16
- 媒体:https://www.marktechpost.com/2026/07/01/nvidia-releases-nemotron-labs-twotower/ · https://www.techtimes.com/articles/319531/20260702/
- X:https://x.com/iScienceLuvr/status/2070416130028794251 · https://x.com/yesnoerror/status/2070613578105647106
- kernel bug:https://github.com/vllm-project/vllm/pull/21783
