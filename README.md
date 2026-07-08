# TwoTower 复现实验 (Nemotron-Labs-TwoTower-30B-A3B)

推理侧复现 NVIDIA Nemotron-Labs-TwoTower 论文,产出"自己的数据"用于组内 paper 精读汇报。
真正的 CUDA 计算跑在 RunPod 的 **2×H100-80GB** 上(两塔 diffusion 需要双卡;单卡只能 AR)。

## 里程碑

- **M0 环境 + 冒烟测试** ← 现在这里。跑通 = 能 load、能出一次不 NaN 的 diffusion 生成、能读到 NFE。
- M1 参数量表(computed,免费,不下 60GB)
- M2 生成过程可视化(Exp 0)
- M3 速度面 NFE/wall-clock(Exp 1,零评分)
- M4 崩溃复现(Exp 3,HumanEval)
- M5 质量-速度 Pareto(Exp 2,GSM8K)

## 两段式:先小卡验环境,再双卡跑实验

**小卡只能验工具链(import + 内核 + config/tokenizer),不能生成** —— 一座塔 ~30GB,
diffusion 要 2×80GB。所以先用便宜卡把环境钉死,再换双 H100 跑。

### 阶段一:小卡(或任意有 CUDA 的机器)验环境

```bash
export HF_HOME=/workspace/hf        # 持久卷上的 HF 缓存
export HF_TOKEN=hf_xxx              # 权重是 gated,先在 HF 页面同意 license
cd twotower-repro
bash setup/install.sh               # 自动探测 torch/cuda/py/abi,装匹配的 mamba-ssm + causal-conv1d
python src/env_check.py             # import + 跑一次 CUDA 内核 + 载 config/tokenizer(不载权重)
python src/param_count.py           # 纯算术参数表,连 GPU 都不用(本地也能跑)
```

### 阶段二:双 H100 跑生成

```bash
python src/smoke_test.py                                              # M0 端到端(触发 ~60GB 下载)
python src/exp0_capture.py --out results/trace.npz                   # E0 采集一条去噪轨迹
python src/run_all.py --exp e1 --prompts data/speed_prompts.jsonl --out results/e1.jsonl  # 速度面
python src/run_all.py --exp e3 --prompts data/humaneval.jsonl --out results/e3.jsonl       # 崩溃(需 HumanEval prompts)
python src/run_all.py --exp e2 --prompts data/gsm8k.jsonl     --out results/e2.jsonl       # Pareto(需 GSM8K prompts)
```

### 阶段三:本地处理产出(无 GPU)

```bash
python src/exp0_render.py --npz results/trace.npz --out results/exp0.gif   # 渲染 GIF
# eval/*.py 打分 + plot.py 画图(待写)
```

`smoke_test.py` 通过判据:AR 和 diffusion 都出非空文本、无 NaN、打印 `NFE` 与 `tokens/NFE`。
`run_all.py` 可断点续跑:重跑会跳过 jsonl 里已完成的 (prompt_id, config_key)。

## 已知坑(边跑边补,给同事的避雷表)

- mamba-ssm / causal-conv1d 的 wheel 四元组 (cu / torch / cxx11abi / cp) 必须全对;**cxx11abi 错了是 import 时 `undefined symbol`,不是安装报错**。`install.sh` 已自动匹配。
- 权重 ~59GB/卡,两塔 diffusion 必须双卡;单卡 OOM 时降到 `--ar-only`。
- Mamba scan 在长上下文 + BF16 会溢出成 NaN(官方代码已在生成时强制 fp32),留意 step_callback 的 NaN 告警。
- `mask_token_id=3` 已由 HF 卡确认,但 `smoke_test.py` 仍会用 tokenizer 复核。
