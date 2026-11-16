# Windows 本机实验环境

本机基线采用 Windows 原生单 GPU PyTorch + Transformers + PEFT + bitsandbytes。环境名称固定为 `opdistill`，用于后续训练模块开发、数值与梯度检查及小规模实验。模型、数据集、正式训练脚本和评测集仍需按实验方案确定。

## 已验证配置（2026-11-16）

| 项目 | 本机实际版本 |
| --- | --- |
| 系统 | Windows 11 24H2，build 26100，64 位 |
| GPU / 内存 | RTX 4080 Laptop GPU，12,282 MiB 显存；约 32 GB 系统内存 |
| NVIDIA 驱动 | 621.01（沿用并验收） |
| CUDA Toolkit / MSVC | 12.4.131 / 19.37.32822（沿用并成功编译运行 sm_89 内核） |
| Python / PyTorch | 3.11.18 / 2.6.0+cu124 |
| PyTorch CUDA runtime / cuDNN | 12.4 / 9.1.0 |
| Transformers / PEFT / bitsandbytes | 4.57.6 / 0.17.1 / 0.48.2 |
| Accelerate / Datasets | 1.10.1 / 4.1.1 |
| JupyterLab / pytest | 4.4.10 / 8.4.2 |

离线验收的真实 JSON 见 [环境验收记录](environment-verification-windows.json)。`pip check` 返回 `No broken requirements found.`；CUDA 矩乘与 CPU 参考数值比较、梯度与 BF16、随机微型 Qwen3 + LoRA 前向/反向/参数更新及 NF4 线性层输入梯度均通过。科学计算、数据、Jupyter 和 TensorBoard 主库导入通过。记录中的随机 loss 是环境冒烟检查值，不是正式模型或蒸馏实验指标。Python 的 `platform` 字段可能按兼容接口显示 Windows-10；系统名称以实际 Windows 查询为 Windows 11。

## 安装布局

- Conda 根安装：沿用 `C:\ProgramData\anaconda3`，不修改 base 的 Python 包。
- 环境：`D:\opdistill\envs\opdistill`。
- Conda 包缓存：`D:\opdistill\conda-pkgs`。
- pip 缓存：`D:\opdistill\pip-cache`。
- Hugging Face 缓存：`D:\opdistill\hf-cache`。
- 临时目录：`D:\opdistill\tmp`。

用户 Conda 配置仅增加 `D:\opdistill\envs` 搜索路径，保留原有目录。缓存和 TEMP/TMP 通过该环境的 Conda 环境变量配置，在退出环境时恢复；新开终端后运行：

```powershell
conda activate opdistill
python scripts/verify_environment.py
jupyter lab
```

Jupyter 中选择 `Python (opdistill)`；kernel 仅注册在该环境的 `share/jupyter/kernels/opdistill`，Python 路径已核对为环境内的 `python.exe`，不写用户全局 kernel。

本机 Windows PowerShell 5.1 已有 Conda hook；Anaconda Prompt 也可直接激活。若使用尚未初始化的 PowerShell 7，本机现有 Conda 24.11 与新版原生参数传递存在兼容问题，可仅在当前终端加载以下 hook，再激活，无需修改全局 PATH 或升级 base：

```powershell
$PSNativeCommandArgumentPassing = 'Legacy'
(& 'C:\ProgramData\anaconda3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate opdistill
```


## 复现

固定主依赖见 `configs/requirements-windows.txt`，完整 pip 版本锁见 `configs/requirements-windows-lock.txt`，Conda 基础规格见 `configs/environment-windows.yml`。

在项目根目录的新机器上，先安装兼容 NVIDIA 驱动、CUDA Toolkit 12.4、Conda 和 Visual Studio C++ 工具，然后运行：

```powershell
powershell -File scripts/setup_windows.ps1 -Root D:\opdistill
```

脚本面向尚不存在的环境。已装好的环境直接激活和验收，避免覆盖。先从 PyTorch 官方 cu124 索引安装 `torch==2.6.0+cu124`，再从 PyPI 按锁文件安装其余依赖，防止被替换为其他 CUDA 构建。CUDA Toolkit 的 MSVC 验收另行执行：

```powershell
powershell -File scripts/verify_cuda_toolkit.ps1
```

驱动版本、系统 Toolkit 和 PyTorch CUDA runtime 是三项不同信息：`nvidia-smi` 显示驱动支持能力，`nvcc --version` 显示系统 Toolkit，`torch.version.cuda` 显示 PyTorch 构建使用的 runtime。

## 空间与下载缓存

安装结束时按文件大小统计：环境约 6.02 GB、保留 PyTorch wheel 2.532 GB、Conda 缓存约 0.898 GB、pip 缓存约 0.287 GB；HF 缓存为空。D 盘剩余约 104.92 GB，C 盘剩余约 23.55 GB。初始 C 盘仅约 5.39 GB，因此选择 D 盘；本任务没有清理 C 盘，期间空闲量变化按实测记录，不归因为本任务操作。目录大小合计包含可能的硬链接重复计数，不等于精确磁盘分配量。

完整 wheel 保留于 `D:\opdistill\wheels\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl`。已验证其大小为 2,532,350,702 字节，官方 SHA256 为 `6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb`，来源为 PyTorch 官方 cu124 索引指向的 `download-r2.pytorch.org`。恢复下载复用了已保存分片，没有重新下载整包。安装器在新机器会从官方源下载；本机如需重新安装同一 wheel，可离线复用：

```powershell
python -m pip install --no-index --no-deps 'D:\opdistill\wheels\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl'
```

## 下载与兼容性

本机安装时 `repo.anaconda.com` 出现 TLS EOF，采用官方 conda-forge。间歇性下载错误经过重试；全程保留 TLS 和包完整性验证。官方 PyTorch wheel 一次下载校验失败后，通过官方地址分段重下，要求范围、长度及最终官方 SHA256 全部通过后才安装。

原生 Windows 基线未安装 vLLM、FlashAttention 或 DeepSpeed。未来需要这些组件时，单独建立 Linux/WSL 或远程 GPU 实验环境并记录其依赖，不能把此基线等同于上述推理/分布式栈已验收。

本机 RTX 4080 Laptop GPU 的可用显存约 12 GB，现有方案的完整教师/学生组合、序列长度和 batch size 仍须实际显存测试；不依据环境冒烟测试推断正式训练预算或性能。

bitsandbytes 使用 Windows 原生 wheel，验收仅覆盖 NF4 Linear4bit 前向和输入梯度；真实教师模型量化的加载、精度与显存仍待模型实验。

## 官方来源

- [PyTorch 2.6.0 / CUDA 12.4 官方安装矩阵](https://pytorch.org/get-started/previous-versions/)
- [Qwen3-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)
- [NVIDIA CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
- [conda-forge 软件源](https://conda.anaconda.org/conda-forge)
- [Python 包索引 PyPI](https://pypi.org/)
- [bitsandbytes 官方安装与平台支持](https://huggingface.co/docs/bitsandbytes/installation)
