# HANDOFF — TwoTower 复现项目交接（给下一个 Claude / 下一场会话）

> 读完这一页你就能立刻接手。项目=深度复现 + 批判性分析 NVIDIA 的 **Nemotron-Labs-TwoTower**
> 扩散语言模型，产出一份给公司同事的**规范化 paper 精读报告**。用户是大一新手，重架构理解+实验、
> 战略性跳过数学推导。**只做推理侧实验，不训练。**

---

## 0. 一句话现状

深度报告 `REPORT_TwoTower.md`（含 6 张图 + 去噪 GIF）**已完成并推送**。GPU 侧最后一批实验
（`run_batch.sh`）**正在 RunPod 上跑**（~2h）。**下一步 = 等用户把 pod 输出下载到本地，然后你在本地
跑分析脚本出图、并进报告。GPU 已收工，不再开 pod。**

---

## 1. 核心论点（报告的灵魂）

> **TwoTower 本质是低成本的「块并行 AR 加速补丁」，而非扩散对自回归（AR）的真正替代。**

三支论点 + 已测证据：
- **a. 没真正抛弃 AR**：γ↑ 时 tokens/NFE→1（实测 7.4→**1.35**）；**b16 commit 顺序 Kendall τb=0.90**
  （远比 DiffusionGemma 的 0.43–0.60 更自回归）；每步 commit-batch 中位数 **1**（DG 是 13–26）。
- **b. 加速来自块内并行 + 置信度纠错，而非「扩散式时间精修」**：`freeze_time`（冻结 AdaLN 时间条件）
  **零影响**（100% clean-correct）；`remask OFF`→质量崩到 0%；`disable_seed`→0% 且 NFE 爆到 243。
  → 真机制是「上下文播种 + 并行草稿 + 置信度迭代纠错」，扩散的时间签名是惰性的。
- **c. 并行脆弱、上限低、被训练焊死**：采样 block>16 崩溃（e3：clean-correct 100/100/90/**50** @ block 8/16/32/64；
  朴素准确率的 90% 是假象，实为一半退化成重复垃圾）；论文实测仅 2.42×。

**诚实边界**：对象是唯一发布的 base checkpoint；承认 2.42× 在低 batch 场景有实用价值、训练成本低（~8%）；
批判的是「扩散替代 AR」的大叙事，不否定工程价值。

---

## 2. 已核实的架构/机制事实（写报告用，别再搞错）

- 模型 `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`，arXiv **2606.26493**，trust_remote_code。
  两份完整权重：**冻结的 AR 上下文塔 + 可训的扩散去噪塔**，逐层 cross-attention。
- Backbone = Nemotron-3-Nano：每塔 **52 层 = 23 Mamba-2 / 23 MoE / 6 attention**，hidden 2688，
  vocab 131072，context 262144（官方卡）。MoE **128 routed / 6 active / 1 shared**（不是 2）。
- 参数：单塔 total/activated **31.578B / 3.580B**，双塔 **63.2B**，**94.5% 在 MoE**（Mamba/attn 极轻）。
- `generate_mask_diffusion` **签名默认 temperature=0.0, confidence_threshold=0.9**
  （论文示例/实验常用 0.1/0.8），block_size=16, steps_per_block=16, mask_token_id=3。也有 `generate_ar`。
- **复现王牌 bug**：HF `config.json` 的 `chunk_size=128`（继承 backbone）遇上 block_size=16 →
  `block<chunk` 命中 Mamba2 SSD kernel 已知边界 bug（同类 vLLM PR #21783）→ 去噪塌成词沙拉、tokens/NFE=1.0。
  **修复 = 把 chunk-scan 换成逐 token `selective_state_update`**（AR 同款、数学等价），已 monkeypatch 进
  `src/twotower.py` 的 `load()`（`TWOTOWER_NOFIX=1` 保留 buggy 原版做 before/after）。修复只影响 wall-clock，
  不影响 NFE/tokens-per-NFE。

**谱系对比（报告第六部分素材）**：TwoTower 是 iLLaDA/LLaDA / DiffusionGemma 里**唯一物理解耦**、
**训练成本最低**（AR 塔冻结）的路线；用 `[MASK]`（经典 MDLM，非 DG 的 uniform-state 重噪）；
多一条 Mamba 状态播种通道，但**无 DG 的 self-conditioning**。DG 的 τb 0.43–0.60 / accept-batch 13–26
vs TwoTower 的 τb 0.90 / commit-batch 中位 1 → **两个独立团队都落在「批量提交但仍顺序」**，互为佐证。

---

## 3. 仓库文件地图（github junze-eng/twotower-repro）

| 文件 | 作用 | 跑在 |
|---|---|---|
| **`REPORT_TwoTower.md`** | ★ 最终深度报告（正文中文、图英文标注） | — |
| `RUNBOOK.md` | 端到端 6 阶段流程 | — |
| `run_batch.sh` | 一键跑完剩余 GPU 实验（①-⑦） | pod |
| `HANDOFF.md` | 本文件 | — |
| `src/twotower.py` | 加载 + **修复 monkeypatch** + NFE 工具 | pod |
| `src/exp0_capture.py` | 抓逐位置提交轨迹 **+ 置信度** | pod |
| `src/run_all.py` | AR / e1-e3 / 崩溃生成 | pod |
| `src/ruler_lite.py` | 长上下文 needle 探针 | pod |
| `src/ablation_topk.py` / `ablation_remask.py` / `ablation_denoiser.py` | 消融 | pod |
| `src/prep_data.py` | 造 GSM8K-mini（离线合成）/ HumanEval-mini | pod |
| `analyze_commit.py` | **τb + commit-batch + 真三角热图 + 真 GIF**（读 exp0 npz） | 本地 |
| `score_all.py` | AR 对比 / HumanEval **pass@1（沙箱执行）** / RULER 汇总 | 本地 |
| `make_figs.py` / `make_gif.py` | 6 张核心图 + 去噪 GIF（读 jsonl/pkl） | 本地 |
| `figs/` | 已出的 6 图 + `denoising.gif` | — |
| `setup/env.sh` `setup/install.sh` | 环境（已固化 `HF_HUB_OFFLINE=1`） | pod |

---

## 4. 数据在哪

- **本地分析工作区**：`C:\Users\whaletech007\work\megatron-lm\all_results\`
  - 已有：Jul9 的 `e1/e2/e3.jsonl`、`abl_remask/abl_denoiser.jsonl`、`trace_main/trace_moe.pkl`、
    `trace.npz`（旧空 frames，弃用）、`make_figs.py`/`make_gif.py`、`figs/`（6 图+GIF）、`REPORT_TwoTower.md`。
  - `trace_tri_b16.npz`（τb=0.90 那次）在本地 `megatron-lm\` 和 `Downloads\`。
- **仓库**：脚本 + 报告 + `figs/` 的权威副本。
- **pod（临时）**：`/workspace/twotower-repro/results/` 正在生成新数据；`/workspace/hf` 是 126GB 权重。

⚠️ 注意：`make_figs.py`/`make_gif.py` 从**自己所在目录**读数据 → 在 `all_results\` 里跑它们（数据在那）。
`analyze_commit.py`/`score_all.py` 用 `--npz`/路径参数 → 从仓库拷到 `all_results\` 或直接指路径。

---

## 5. run_batch.sh 会产出什么（等这些下载回来）

在 pod 上跑，产出到 `results/`：
1. `trace_tri_b64.npz` — 块 64 崩溃版三角
2. `ar.jsonl` — AR baseline（GSM8K）
3. `he_collapse.jsonl` + `he_ar.jsonl` — HumanEval 崩溃 + AR 代码基线
4. `ruler.jsonl` — 长上下文 needle（2K/8K/16K/32K，diff+ar）
5. `abl_topk.pkl` — top-k MoE 消融
6. `trace_buggy_demo.npz` — before/after 的 buggy（词沙拉）版
7. `trace_tri_aggr.npz` — aggressive-γ（γ0.5/steps4）三角，测 τb 在最大并行下是否仍高

外加用户还会下载：`trace_tri_fact.npz`、`trace_tri_code.npz`（τb×任务型用）、`data/humaneval_mini.jsonl`（pass@1 需要）。

**刻意跳过的可选实验（不影响论点 a/b/c —— 别当成遗漏再劝用户重开 pod）**：
- **confidence→correctness**（DG 第 6 发现的 TwoTower 版）：需新写「多 prompt 置信度采集 + 相关性」分析，纯加分项。
- **τb 多 prompt 统计**：现每任务型 n=1（b16=math / fact / code / aggr），足够讲清任务依赖性；要更硬统计再补。
- **温度敏感性**：全用 `temp=0.0` 贪心（论文默认 0.1）——标准可复现设置，论点不依赖。
用户已明确不再开 RunPod。以上只在用户**主动要求**时才做，否则视为已完成。

---

## 6. 数据到位后你（下一个 Claude）要做的分析

全在本地、无 GPU。数据放进 `all_results\` 后：

```bash
# 1) 并行验证 / commit 顺序（对每个 exp0 npz 跑一次）：τb + commit-batch + 真三角 + 真 GIF
python analyze_commit.py --npz trace_tri_b16.npz  --outdir figs_commit_b16
python analyze_commit.py --npz trace_tri_fact.npz --outdir figs_commit_fact
python analyze_commit.py --npz trace_tri_code.npz --outdir figs_commit_code
python analyze_commit.py --npz trace_tri_b64.npz  --outdir figs_commit_b64
python analyze_commit.py --npz trace_tri_aggr.npz --outdir figs_commit_aggr
#   → 收集每个的 [stats] 里的 tau_b / commit_batch，做「τb × 任务型 / γ」对照表（对标 DG）

# 2) AR 对比 + HumanEval pass@1 + RULER
python score_all.py --ar results/ar.jsonl --diff e2.jsonl \
    --he results/he_collapse.jsonl --he-ar results/he_ar.jsonl \
    --he-prompts data/humaneval_mini.jsonl --ruler results/ruler.jsonl --outdir figs

# 3) （已做过，如需重出）6 张核心图 + 去噪 GIF
python make_figs.py ; python make_gif.py

# 4) before/after：analyze_commit 跑 trace_buggy_demo.npz，和 b16 并排（buggy tokens/NFE≈1、词沙拉 vs 修复后正确）
```

**已知结果（无需重算，报告已用）**：b16 τb=0.90、commit-batch 中位 1；e3 崩溃 clean-correct 100/100/90/50；
消融 remask_OFF=0% / disable_seed=0%(NFE243) / freeze_time=100%；γ→tok/nfe 7.4→1.35；MoE churn ~3.5/6。

⚠️ 打分口径：**degeneration-aware** —— 答案「正确 **且** 未退化成短周期重复」才算 clean-correct
（这把 block64 从虚高 90% 还原成真实 50%）。评测集是 15 题**离线合成算术（synthmath，非官方 GSM8K）**。

---

## 7. 分析完 → 并进报告 `REPORT_TwoTower.md`

- τb×任务型表 + 真三角 + 真 GIF → §7 论点 A（替换现在的「示意」GIF；加 τb 对标 DG 的表）。
- AR 对比数字 → §8 把「引论文 2.42×」换成自测的 retention/speedup（注明 wall-clock 因慢速修复偏悲观，用 tokens/NFE）。
- HumanEval pass@1 崩溃曲线 → §7 论点 C 代码侧。
- RULER → 新增「长上下文能力继承自冻结 AR 塔」小节。
- before/after → §6 bug 那节的视觉。
- aggressive-γ τb → §7 论点 A 补一句「即使最激进并行，τb 仍高」。

---

## 8. 已应用的核实纠正（用户很在意准确性，别退回错误版本）

1. `generate_mask_diffusion` 默认是 **temp=0.0 / γ=0.9**（0.1/0.8 是示例值，非默认）。
2. 256K context 来自官方卡（仓库 config 未收录），已按官方卡陈述。
3. mask id=3 由 model card 确认；「复用 [INST] slot」无独立佐证，已删。
4. bug 根因表述软化为「block<chunk 命中 kernel 边界 bug，换 kernel 修复」，**不**强断言「initial_states 被过度放大」
   （因为 seed-off 探针也仍垃圾，没单独隔离）。
5. 论点 b **据实验重构**：不是「迭代没用」，而是「时间条件惰性（freeze_time 证据）」。
6. tokens/NFE 用实测区间（1.35–7.4），非早期占位值。

---

## 9. 血泪教训 / 避雷（RunPod + 操作）

- **2 张 A100 同时只够一份模型**（~63G/卡）；**双开生成任务=OOM**。一个终端串行跑，监控用 `watch -n 2 nvidia-smi`（只读）。
- **新 shell 必须** `source setup/env.sh`（已自动带 `HF_HUB_OFFLINE=1`）。否则 HF 上网重拉新 revision + **重下 126GB**。
- **只有 Network Volume（us-md-1）持久**；起 pod 必须挂**同一块下满 126GB 的卷**（`du -sh /workspace/hf` 应 ~120G+）。
  容器盘/pip 环境停机即清 → 换容器后 `bash setup/install.sh`（~2min）。
- **别反复 Ctrl-C 打断下载**（MooseFS 会留 `.lock`；恢复 `rm -rf /workspace/hf/hub/.locks`）。
- 用户终端**长命令粘贴会断行、多行块会丢 `cd`** → 用 `run_batch.sh` 这类脚本文件跑，别手粘长命令。
- 环境细节见 `TROUBLESHOOTING.md`；完整流程见 `RUNBOOK.md`。

---

## 10. 用户剩余动作（就三步，GPU 已收工）

1. 等 `run_batch.sh` 打印 `ALL DONE`；`grep FAILED results/logs/batch.log` 有失败趁 pod 在补跑。
2. 补跑 ⑦（若批是加 ⑦ 之前启动的）：`python src/exp0_capture.py --block-size 16 --max-new 64 --steps 4 --gamma 0.5 --out results/trace_tri_aggr.npz`
3. 下载 `results/*` + `data/humaneval_mini.jsonl` +（已有）`trace_tri_fact/code.npz` 到 `all_results\` → 对 Claude 说「跑分析」。
