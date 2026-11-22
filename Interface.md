# 公共接口

## 环境安装与验收 CLI

下表描述环境安装与验收接口。单轮 SFT 基线接口见下文；在线蒸馏及统一测试系统接口尚待设计。

| 接口 | 输入 | 输出与失败行为 |
| --- | --- | --- |
| `scripts/setup_windows.ps1 [-Root D:\opdistill]` | 根目录；已可用的 Conda、NVIDIA 驱动 | 在 `<Root>\envs\opdistill` 创建环境、设置作用域缓存、安装锁定依赖并验收；环境已存在或命令失败时抛错停止 |
| `python scripts/verify_environment.py [--output <路径>]` | 已激活的 opdistill；至少一块支持 BF16 的 CUDA GPU | stdout 输出 JSON；可选写出 UTF-8 JSON；依赖冲突、CUDA、数值/梯度或参数更新检查失败时非零退出 |
| `scripts/verify_cuda_toolkit.ps1 [-BuildDirectory D:\opdistill\tmp]` | CUDA 12.4 的 nvcc、VS C++ 工具及 sm_89 GPU | 在指定目录产生临时二进制，编译并运行32元素加法内核；失败时抛错停止 |

环境验收 JSON 包括 `status`、`scope`、`python`、`python_executable`、`platform`、`cudnn`、`gpu`、`gpu_memory_bytes`、`torch_cuda_runtime`、`nf4_linear_forward_input_backward`、`bf16`、`cuda_matmul_backward`、`random_qwen3_lora_optimizer_step`、`qwen3_logits_shape`、`qwen3_loss`、`pip_check` 和 `packages`。其中随机 Qwen3 输入为 int64 `[2,16]` token IDs，logits 为 BF16 `[2,16,256]`；模型随机初始化且不下载权重，LoRA 可训练参数必须有有限梯度且 optimizer step 后至少一项更新。另外检查 bitsandbytes NF4 Linear4bit 的 BF16 输入 `[2,64]` 到输出 `[2,32]` 的前向和输入梯度有限且非零。该 JSON 仅用于环境验收，不是正式实验结果格式。

## 单轮 Function Calling SFT 基线

在仓库根目录使用 `python -m scripts.run_sft_baseline --config <JSON 配置> --data <JSON 或 JSONL> --output <空目录>`。运行要求支持 BF16 的 CUDA GPU、可加载指定版本的 Qwen 模型与 tokenizer，并使用现有 Windows 依赖环境。目录已经存在且非空时拒绝覆盖；失败的运行可能留下部分记录，重试需使用新的空目录。

`configs/sft_baseline_4b_smoke.json` 包含 `model_id`、`model_revision`、`dataset_source`、`dataset_revision`、`seed`、`validation_fraction`、`test_fraction`、`max_length`、`max_steps`、`gradient_accumulation_steps`、`learning_rate`、`lora_rank`、`lora_alpha`、`lora_dropout`、`lora_targets`、`quantization`、`gradient_checkpointing`、`max_validation_samples`、`max_eval_samples` 和 `max_new_tokens`。该配置指定官方 xLAM-60k 原始 JSON 及版本 `26d14ebfe18b1f7b524bd39b404b50af5dc97866`，8 步和有限样本数仅用于流程试跑。随机模型和合成数据使用仓库外的单独配置。使用其他数据时需另建配置，记录真实来源和版本，正式运行前应根据实验设计确定预算。

原始行包括 `id`（可省略）、`query`、`tools`、`answers`。`tools` 和 `answers` 可以是 JSON 字符串或对应数组；支持单个或多个有序调用。数据模块 `read_rows(path)` 返回原始行，`prepare_splits(rows, *, seed, validation_fraction, test_fraction)` 返回划分字典和统计清单。规范化行包含 `id`、`query`、OpenAI 风格 `tools`、有序 `calls`，每个调用包含 `name` 与对象形式的 `arguments`。不符合处理要求的行和完全重复行被计数，统计清单保留排除原因。

xLAM 的参数类型支持简单类型、`optional` 修饰及安全解析的嵌套 List、Dict、Tuple、Union、Optional。解析不执行类型表达式；Callable、set 等不支持的类型使所在行被排除并计数。显式 `required` 布尔值决定参数是否必需，没有声明时不推断必需。只有与转换后类型兼容且可序列化的显式默认值会保留，不兼容的默认值被省略并在 `normalization_diagnostics_valid_rows` 中计数；类型文本中声明默认值也计数，但不推断其值。

划分将规范化 query 相同或完整工具集合签名相同的行放进同一连通分组，分组少于三个时报错。大于两个留出集目标大小的组优先放入训练集，同时保证三个划分各有分组。实际分组大小可能使比例偏离配置，清单保留实际分组数、最大组大小和样本计数；此规则未检测语义近重复，单个工具仍可出现在不同划分。各划分使用独立的固定种子顺序，仅检查满足试跑数量所需的样本，训练最多选 `max_steps * gradient_accumulation_steps` 条，验证和测试分别使用各自的样本上限。检查到的超长完整样本被排除而非截断，清单 `tokenization` 按划分记录总数、已检查数、保留及超长 ID、未检查数；训练、验证或测试集检查后为空时停止。训练标签对输入提示设为 `-100`，仅对 assistant 响应计算损失。

`score_output(text, sample)` 接受由完整 `<tool_call>...</tool_call>` 块组成的文本，返回 `structure_valid`、`function_name_match`、`arguments_match` 和 `joint_match` 布尔值。名称和参数分别按调用顺序精确比较，联合匹配要求两者均匹配；全空白文本仅在目标调用为空时视作有效的空调用序列。格式检查验证调用块及 JSON 对象结构，不验证函数是否属于工具列表、参数是否符合工具 schema、工具执行结果或参数语义正确性，也未实现 BFCL 官方评测。

成功运行的输出目录包含 `status.json`、`config.json`、`provenance.json`、`split_manifest.json`、`train_log.jsonl`、`adapter/`、`predictions.jsonl` 和 `metrics.json`。`status.json` 在开始时记录 `incomplete`，所有步骤成功后更新为 `complete`。其中 provenance 记录 Git commit 与 dirty 状态、输入输出路径、工作目录、命令参数、数据 SHA-256、Python、平台、GPU、CUDA 及依赖版本，配置文件记录预算和模型版本。日志记录更新步、损失、梯度范数与耗时；预测逐行保存原始输出、目标、各项布尔分数、生成 token 数、是否到达 EOS 及是否达到 token 上限；metrics 汇总验证损失、评测分子/分母/比例、峰值显存与耗时。dirty 状态不包含代码差异，正式结果应从已提交且可追溯的代码运行。

可选配置 `evaluate_base_model` 为布尔值，缺省为 `false`。设为 `true` 时，训练前禁用 LoRA adapter，对同一组测试样本按相同生成参数评测原始量化模型，记录 `base_predictions.jsonl`；训练后重新加载保存的 adapter 并生成 `predictions.jsonl`。`metrics.json` 的 `base_metrics` 汇总原始模型指标，`paired_joint` 给出联合精确匹配的 `improved`、`regressed`、`unchanged_correct`、`unchanged_incorrect`、`unchanged` 和 `denominator`；逐条比较前检查 ID 和目标调用一致。`validation` 仍为 SFT 后 teacher-forced 损失，配对模式下耗时和显存峰值包含训练前的原始模型评测。配置 `configs/sft_baseline_06b_paired.json` 设定 0.6B、64 更新步、累积 4、最多 16 条验证和 64 条测试样本。
