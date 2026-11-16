# 公共接口

## 环境安装与验收 CLI

当前尚无项目 Model、Dataset、Distillation、Trainer 或 Evaluator 的公共接口；确定设计并实现时再补充。

| 接口 | 输入 | 输出与失败行为 |
| --- | --- | --- |
| `scripts/setup_windows.ps1 [-Root D:\opdistill]` | 根目录；已可用的 Conda、NVIDIA 驱动 | 在 `<Root>\envs\opdistill` 创建环境、设置作用域缓存、安装锁定依赖并验收；环境已存在或命令失败时抛错停止 |
| `python scripts/verify_environment.py [--output <路径>]` | 已激活的 opdistill；至少一块支持 BF16 的 CUDA GPU | stdout 输出 JSON；可选写出 UTF-8 JSON；依赖冲突、CUDA、数值/梯度或参数更新检查失败时非零退出 |
| `scripts/verify_cuda_toolkit.ps1 [-BuildDirectory D:\opdistill\tmp]` | CUDA 12.4 的 nvcc、VS C++ 工具及 sm_89 GPU | 在指定目录产生临时二进制，编译并运行32元素加法内核；失败时抛错停止 |

环境验收 JSON 包括 `status`、`scope`、`python`、`python_executable`、`platform`、`cudnn`、`gpu`、`gpu_memory_bytes`、`torch_cuda_runtime`、`nf4_linear_forward_input_backward`、`bf16`、`cuda_matmul_backward`、`random_qwen3_lora_optimizer_step`、`qwen3_logits_shape`、`qwen3_loss`、`pip_check` 和 `packages`。其中随机 Qwen3 输入为 int64 `[2,16]` token IDs，logits 为 BF16 `[2,16,256]`；模型随机初始化且不下载权重，LoRA 可训练参数必须有有限梯度且 optimizer step 后至少一项更新。另外检查 bitsandbytes NF4 Linear4bit 的 BF16 输入 `[2,64]` 到输出 `[2,32]` 的前向和输入梯度有限且非零。该 JSON 仅用于环境验收，不是正式实验结果格式。
