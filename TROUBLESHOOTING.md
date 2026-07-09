# 复现踩坑记录 (TROUBLESHOOTING)

TwoTower 推理复现在 RunPod 2×A100-80GB 上的真实踩坑清单。给 presentation 的 "Step 7 避雷表"
和后来的同事用。按"环境坑"和"模型坑"两类记。

## 环境坑(全部已在 setup/install.sh 修好)

### 1. mamba-ssm / causal-conv1d 的 wheel 四元组必须精确匹配
- 现象:`bash setup/install.sh` 报 404;或 import 时 `undefined symbol`。
- 根因:预编译 wheel 名 = `cu{CUDA大版本} × torch{大.小} × cxx11abi{TRUE/FALSE} × cp{python}`,
  任一不匹配就 404 或符号错。**cxx11abi 错了不是安装报错,是 import 时 undefined symbol**。
- pod 实际:`cp312 / torch2.8 / cu12 / cxx11abiTRUE`,A100-80GB。
- 修:install.sh 从 live 解释器探测四元组再拼 URL。

### 2. causal-conv1d 版本要覆盖 torch2.8
- 现象:`causal_conv1d 1.5.0.post8` 的 torch2.8 wheel 不存在 → 404。
- 修:升到 `1.6.2.post1`(有 cu12torch2.8cxx11abiTRUE cp312 wheel)。mamba_ssm 2.3.2.post1 本就有 torch2.8 wheel。

### 3. ⚠️ 最严重:装 kernel 不加 --no-deps,torch 被从 2.8 升到 2.13
- 现象:装完 `import mamba_ssm` 报 `undefined symbol: _ZN3c104cuda...`;`import torch` 显示 2.13.0 + 一堆 CUDA-13 包。
- 根因:`pip install causal-conv1d(wheel)` **不带 --no-deps** 时,它的依赖链把 torch 升级了,
  拖来 CUDA-13 全家桶(cuda-toolkit-13、triton-3.7、tilelang...),而 mamba 的 .so 是按 torch2.8 编的 → ABI 崩。
- 修:**装 kernel 一律 `--no-deps`**;装完断言 torch 版本未变。
- 恢复:`pip install --force-reinstall torch==2.8.0 ... --index-url .../cu128`,再 `--no-deps` 重装 kernel。

### 4. 别手动升 triton
- 现象:`pip install triton>=3.5` 后 `OSError: libcudart.so.13: cannot open shared object file`。
- 根因:triton 3.5+ 拉 CUDA-13 运行时,和 pod 的 CUDA 12.8 冲突。
- 真相:mamba_ssm 2.3.2 的 torch2.8 wheel 就是按 **triton 3.4** 编的。pip 会警告
  "requires triton>=3.5.0" 但那是**无害警告**,triton 3.4 实际能跑(kernels OK 已验证)。
- 修:install.sh 删掉所有 triton 折腾,锁死不动。

### 5. python 环境不能装在 /workspace 网络卷上
- 现象:`python -m venv /workspace/venv` + pip 装包,卡住 20+ 分钟不动。
- 根因:`/workspace` 是 MooseFS 网络卷(`mfs#us-md-1.runpod.net`),对"海量小文件"(python 包)
  极慢;对"大文件"(权重)没问题。
- 修:python 环境装到**本地容器盘**(系统 python,~2分钟),换容器后重跑 install.sh 即可。
  **只有 126GB 权重放 /workspace/hf**(大文件,持久)。

### 6. RunPod 持久化:只有 Network Volume 持久,Container Disk / 本地盘停机即清
- 现象:重启 pod 后 `/workspace` 空了、权重没了、要重下 126GB。
- 根因:早期 `/workspace` 挂的是临时本地盘 `/dev/md1`(停机清空);真正持久的是
  **Network Volume**(`mfs#...`,us-md-1 区)。
- 修:创建 Network Volume(us-md-1 同区),起 pod 时挂到 /workspace。权重一次下完永久复用。
  **前提:下次起 pod 必须同 region + 挂同一块卷。**

### 7. 网络卷上强杀进程会留残留文件锁
- 现象:下载卡在 `Still waiting to acquire lock on .../.locks/....lock (elapsed: 20s)`。
- 根因:上一个被 Ctrl-C / kill 的 HF 下载进程,在 MooseFS 上没释放 `.lock`。
- 修:`pkill -9 -f huggingface; rm -rf /workspace/hf/hub/.locks`,再后台重下(会续传)。
- 纪律:**别反复 Ctrl-C 打断下载**;用 `nohup ... &` 后台下 + `watch du -sh` 看进度。
  下载慢时进度条不刷新 ≠ 卡住(hf_transfer 只在每个 shard 完成时跳)。

### 8. HF_TOKEN 带非 ASCII 字符 → UnicodeEncodeError
- 现象:下载报 `UnicodeEncodeError: 'latin-1' codec can't encode ... position 10-11`。
- 根因:粘贴 token 时带进了全角引号/空格等非 ASCII;HTTP header 只接受 latin-1。
- 修:`huggingface-cli login` 交互输入(内部 strip),或手打 export 别粘贴。
  查:`echo -n "$HF_TOKEN" | od -c` 看有没有非 ASCII 字节。
- 教训:token 明文别贴到任何对话里;泄露了立刻去 HF settings revoke 重发。

## 模型坑(进行中)

### 9. 🔴 diffusion 生成垃圾(当前主阻塞)
- 现象:`generate_ar` 输出**正常连贯**(如 "Paris...");但 `generate_mask_diffusion` 输出**词沙拉**
  (", best best the the minds...")、NFE 恒等于 max(64/64)、tokens/NFE=1.0(零并行)、
  diffusion 比 AR 还慢。
- 已排除:
  - 代码旧版 — 用的是 HF 官方最新 snapshot(35b6498...)。
  - 权重不匹配 — 同 snapshot;check_weights.py 确认去噪塔权重已加载(std≈0.0176,absmax=25,
    无 "newly initialized" 警告,与 context 塔统计一致)。
  - 官方参数 — 官方示例参数(temp=0.1, thr=0.8)也垃圾。
  - triton — 无关(见坑 4)。
  - AdaLN 时间条件 — diagnose_diffusion.py 的 adaln-off(t=None)仍垃圾。
  - Mamba 状态播种 — seed-off(initial_states=None)仍垃圾。
- 顶层模块命名(供定位):`context_tower / context_lm_head / denoiser_tower / lm_head /
  t_embedder / t_block / scale_shift_tables(52)`。
- 下一步:`src/probe_logits.py` 抓去噪塔第一步原始 logits,区分:
  - 前向坏(上下文没进去噪塔 → cross-attn / cache)vs
  - 后处理坏(_mdlm_forward / 置信度 / commit)。
- 待查代码:`_mdlm_forward`、`_denoiser_block_attention`、`_build_denoiser_cache_diffusion`。
