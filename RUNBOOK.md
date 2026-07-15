# RUNBOOK — 从零跑完 TwoTower 全部实验

一趟端到端流程:重开 pod → 环境 → 采集/生成(GPU)→ 下载 → 本地分析出图 → 并入报告。
**铁律**:每批命令都在 `/workspace/twotower-repro` 里跑;整段一起粘,别单挑几行(会丢 `cd`)。

产出:并行验证(τb + commit-batch + 真三角/真 GIF)、AR baseline、HumanEval 崩溃、长上下文扫描,
外加已有的 6 张图 + 消融。脚本全在仓库,`git pull` 即最新。

---

## Phase 0 — pod + 权重(最容易翻车的一步)

- 起 pod 前:**RunPod 控制台确认你有一块下满 126GB 的 Network Volume(us-md-1)**,起 pod 时**挂它到 `/workspace`**。卷和 pod 是分开的——挂错成空卷 = 白重下。
- 起 pod 后先自检权重:
  ```bash
  du -sh /workspace/hf/hub/models--nvidia--Nemotron-Labs-TwoTower-30B-A3B-Base-BF16   # 要 ~120G+
  ```
  若只有几 G → 卷没挂对,先解决再往下;若确需重下,`HF_HUB_OFFLINE=0 huggingface-cli download nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` 后台下,别反复 Ctrl-C。

## Phase 1 — 环境(容器换了 pip 环境就没了,~2 分钟重建)

```bash
cd /workspace/twotower-repro
git pull origin main
source setup/env.sh
export HF_HUB_OFFLINE=1
python -c "import transformers, mamba_ssm, causal_conv1d; print('env OK')" || bash setup/install.sh
```
看到 `env OK`(或 install.sh 末尾 `imports OK` / `PASS`)再继续。

## Phase 2 — 数据(全新卷才需要)

```bash
cd /workspace/twotower-repro
[ -f data/gsm8k_mini.jsonl ] || python src/prep_data.py --which gsm8k --n 15 --out data/gsm8k_mini.jsonl
HF_HUB_OFFLINE=0 python src/prep_data.py --which humaneval --n 20 --out data/humaneval_mini.jsonl   # 需联网
```

## Phase 3 — 采集/生成(GPU,按优先级)

```bash
cd /workspace/twotower-repro

# ① ★★ 并行验证 + 真三角/真 GIF:抓逐位置提交轨迹(现已捕获 frames + 逐位置置信度)
python src/exp0_capture.py --block-size 16 --max-new 64  --steps 16 --gamma 0.8 --out results/trace_tri_b16.npz
python src/exp0_capture.py --block-size 64 --max-new 128 --steps 16 --gamma 0.8 --out results/trace_tri_b64.npz
#   自检:输出里 [cb] first xt row width= 必须 >0；saved ... frames=(F, W) 的 W 必须 >0

# ② ★ AR baseline(自测加速/质量保留)
python src/run_all.py --exp ar --prompts data/gsm8k_mini.jsonl --out results/ar.jsonl --limit 15

# ③ ★ HumanEval 代码侧崩溃 + AR 代码基线
python src/run_all.py --exp e3 --prompts data/humaneval_mini.jsonl --out results/he_collapse.jsonl --limit 10
python src/run_all.py --exp ar --prompts data/humaneval_mini.jsonl --out results/he_ar.jsonl --limit 10

# ④ 长上下文 needle 扫描(tokens/NFE + 检索质量;32K 若 OOM 会自动记录跳过)
python src/ruler_lite.py --lengths 2048 8192 16384 32768 --out results/ruler.jsonl --ar

# ⑤(可选)top-k MoE 消融 —— 脚注级,别指望推进论点
python src/ablation_topk.py --prompts data/gsm8k_mini.jsonl --out results/abl_topk.pkl --limit 10

# ⑥ ★ 层间冗余:AR 塔 vs 扩散塔(对应 arXiv 2603.07475;图直接在 pod 上出)
#   纯数学自检(无需模型/GPU,任意机器):
python src/layer_similarity.py --selftest
#   真跑(2 卡):写 results/layer_sim/*.csv + summary.json + figs/fig_layer_redundancy.png
python src/layer_similarity.py --prompts data/gsm8k_mini.jsonl --out results/layer_sim \
    --num-prompts 8 --block-size 16 --plot
```

## Phase 4 — 下载 + 停 pod

把这些拉回本地(放进本地 `results/` 或分析目录),然后**停 pod**:
```
results/trace_tri_b16.npz  trace_tri_b64.npz  ar.jsonl  he_collapse.jsonl  he_ar.jsonl  ruler.jsonl  abl_topk.pkl
results/layer_sim/   figs/fig_layer_redundancy.png   # 层间冗余(⑥,--plot 已在 pod 出图)
data/humaneval_mini.jsonl   # pass@1 打分需要它拿函数签名
```

## Phase 5 — 本地分析出图(无 GPU;需 numpy matplotlib imageio)

```bash
# 并行验证:τb + commit-batch + 真三角 + 真 GIF
python analyze_commit.py --npz results/trace_tri_b16.npz --outdir figs
python analyze_commit.py --npz results/trace_tri_b64.npz --outdir figs_b64   # 崩溃版对照

# 已有的 6 张图 + 消融 GIF（把下载的 jsonl / trace_*.pkl 放在脚本同目录）
python make_figs.py
python make_gif.py

# AR 对比 + HumanEval pass@1(沙箱子进程执行生成代码 vs 单测)+ RULER 长上下文汇总
python score_all.py --ar results/ar.jsonl --diff results/e2.jsonl \
    --he results/he_collapse.jsonl --he-ar results/he_ar.jsonl \
    --he-prompts data/humaneval_mini.jsonl --ruler results/ruler.jsonl --outdir figs
```
`analyze_commit.py` 会打印 `[stats] {tau_b, commit_batch_mean/median/max, ...}` 并写出
`fig_commit_order.png`、`fig_commit_batch.png`、`denoising_real.gif`、`commit_stats.json`。
AR / HumanEval / RULER 的打分我这边接一个小脚本汇总(pass@1 需沙箱执行,本地做)。

## Phase 6 — 并入报告

- τb / commit-batch → 报告 §7 论点 A（并行验证,对标 DG 的 τb 0.43–0.60 与 accept batch 13–26）。
- 真三角 + 真 GIF → 替换 §7 里「示意」版 GIF。
- AR baseline → §8 把「引论文 2.42×」换成自测数字。
- HumanEval → §7 论点 C 代码侧。
- RULER → 新增「长上下文:能力继承自冻结 AR 塔」一节。

---

### 脚本索引
| 脚本 | 作用 | 跑在 |
|---|---|---|
| `src/exp0_capture.py` | 采集逐位置提交轨迹 + 置信度 | GPU |
| `src/run_all.py` | AR / e1-e3 / 崩溃生成 | GPU |
| `src/ruler_lite.py` | 长上下文 needle 探针 | GPU |
| `src/ablation_topk.py` | top-k MoE 消融 | GPU |
| `analyze_commit.py` | τb / commit-batch / 真三角 / 真 GIF | 本地 |
| `make_figs.py` / `make_gif.py` | 6 张图 + 去噪 GIF | 本地 |
| `score_all.py` | AR 对比 / HumanEval pass@1 / RULER 汇总 | 本地 |
