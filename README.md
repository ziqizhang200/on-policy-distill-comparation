# 基于在线策略蒸馏的大语言模型后训练方法研究

哈尔滨工业大学本科毕业设计项目。当前开题报告设置三个研究问题：第一项联合设计决策状态的监督位置与位置条件化散度目标；第二项研究保留关键决策 token 和尾部质量的稀疏教师反馈；第三项设计统一测试系统，将同前缀配对回放与独立闭环训练连接起来，追踪结构化调用错误和分项资源成本。毕业论文拟围绕三项研究分别成章。以上均为待实验验证的方案。

## 当前状态

已建立统一参考文献库；截至 2026-10-07，`Papers/references.bib` 含 45 条 BibTeX 条目，新增条目的核验状态仍需逐条复查。开题答辩已通过，正式报告按“决策状态与散度目标的联合方法—稀疏反馈方法—统一测试系统”组织研究内容，尚待导师审阅与实验验证。尾部类别及学生感知支持集均有先行研究，本文的联合方案目前只是待验证假设。

已完成并验收本机 Windows 单 GPU 实验环境 `opdistill`；安装布局、固定依赖和验收方式见下文及环境说明。

已增加单轮 Function Calling 的 SFT 基线试跑实现。开题报告中的候选组合为 Qwen3-0.6B LoRA 学生和冻结的 Qwen3-4B 教师。周报第 6 周记录了量化后的 4B LoRA SFT 可继续运行，但目前未在项目相关目录找到该次运行的配置、日志或检查点，训练完成状态和质量指标尚待核验。当前 4B 配置用于继续基线流程试跑，后续比较 OPD 与 SFT 时仍需受控设计同一 0.6B 学生的 SFT 对照。

## 项目结构

- `configs/`：Windows Conda 基础环境、主依赖及完整 pip 版本锁。
- `scripts/`：环境安装、CUDA Toolkit 编译验收、PyTorch/Qwen3/LoRA/NF4 验收和 SFT 基线试跑入口。
- `src/`、`tests/`：基线数据处理、结构化评测及其测试。
- `docs/`：本机环境与安装验收记录。
- `Papers/`：统一文献库、导读、核验记录及本地论文资料。
- `Interface.md`：安装验收及当前基线相关接口。
- `STAGE.md`：长期进展、决策和后续目标。

## 环境安装与使用

现有机器的 Conda 环境位于 `D:\opdistill\envs\opdistill`。环境、包缓存、Hugging Face 缓存和临时文件均优先放 D 盘，避免 C 盘空间不足。以下命令在 Anaconda Prompt 或已初始化的 Windows PowerShell 5.1 中使用；PowerShell 7 的当前会话兼容设置见环境说明。

```powershell
conda activate opdistill
python scripts/verify_environment.py
```

新机器环境安装规格、官方软件下载源、Toolkit 编译验证命令及完整记录见 [Windows 环境说明](docs/environment-windows.md)。安装器用于创建新环境：

```powershell
powershell -File scripts/setup_windows.ps1 -Root D:\opdistill
```

本机采用 Windows 原生 PyTorch + Transformers + PEFT + bitsandbytes，不包含 vLLM、FlashAttention 或 DeepSpeed。相关 Linux 推理/分布式栈需另行建立与验收。

## 数据、训练、评估与复现

本地数据输入接受 xLAM 格式的 JSON 数组或 JSONL。候选数据集 `Salesforce/xlam-function-calling-60k` 的官方仓库要求接受访问条件：原始文件的匿名 HTTP 请求返回 401，已登录并获得访问权限后可从网页下载。仓库内配置指定官方 `xlam_function_calling_60k.json` 及版本 `26d14ebfe18b1f7b524bd39b404b50af5dc97866`；使用其他数据时必须另建配置，填写真实来源和版本。保留原始文件及校验信息，数据和权重放在 Git 仓库外。

在项目根目录使用已验收的 Windows 环境运行试跑。以下路径是示例，`--data` 必须指向取得授权的真实数据文件，输出目录须为空：

```powershell
$env:HF_HOME = 'D:\opdistill\hf-cache'
& 'D:\opdistill\envs\opdistill\python.exe' -m scripts.run_sft_baseline --config configs/sft_baseline_4b_smoke.json --data 'D:\opdistill\data\xlam_function_calling_60k.json' --output 'D:\opdistill\runs\sft-4b-pilot-001'
```

默认配置固定了 4B 模型版本、NF4、LoRA、随机种子与 8 个更新步，只适合流程试跑。随机模型及合成数据的 CUDA 流程检查使用仓库外的单独配置。运行生成配置、环境和数据哈希、划分与超长样本记录、训练日志、LoRA adapter、原始预测和指标，详情见 [公共接口](Interface.md)。该流程不执行真实工具；指标是生成文本的结构和与标注的有序精确匹配。正式比较前需固定数据版本、划分、模型版本、预算和评测，并记录干净的代码版本。环境验收中的随机微型 Qwen3 和 NF4 线性层只检查计算栈，不作为模型质量或显存预算结论。

配置 `configs/sft_baseline_06b_paired.json` 在同一批留出样本上比较量化的原始 0.6B 模型和 SFT 后模型，设置 64 个更新步、每步累积 4 个样本，验证最多 16 条、测试最多 64 条。将上述命令的 `--config` 改为该路径，并使用新的空输出目录即可运行。输出包含两组原始预测、各自指标，以及联合精确匹配从错误变正确、正确变错误和保持不变的配对样本计数。该配置为小规模试验，尚需结合实际运行记录解释。

## 文献与协作入口

- [文献导读与项目书引用映射](Papers/README.md)
- [统一 BibTeX 文献库](Papers/references.bib)
- [项目进展与后续目标](STAGE.md)
- [公共接口](Interface.md)
- [开题报告修改与研究方案交接](Papers/OPENING_REPORT_HANDOFF.md)
- [协作与研究规范](AGENTS.md)

`Papers/notes/` 保存文献核验记录。论文原始资料和模板沿用本地保存规则；部分正式报告与答辩输出文件已纳入版本控制，具体以 Git 跟踪状态为准。

## 4B 教师 SFT 试验

`configs/sft_teacher_4b_paired.json` 固定 Qwen3-4B 版本，以 NF4 和 LoRA 训练 64 个更新步；在同一批最多 32 条验证样本上记录原始模型与重载保存的 adapter 后模型的生成结果，验证损失最多使用其中 16 条。测试集不用于此阶段的生成评估。运行方式与前述 SFT 命令相同，替换配置并指定新的空输出目录。该结果用于教师非退化检查，不代表泛化提升。

运行中 `progress.json` 是实时更新的标准 JSON，包含阶段、步数、损失、梯度和显存；完成时还生成 `train_history.json`、`base_predictions.json` 和 `predictions.json` 数组，保留原有 JSONL。失败时进度文件记录错误。

## 在线蒸馏试跑（待执行）

`python -m scripts.run_opd_pilot --config configs/opd_pilot_06b_4b.json --data <数据文件> --adapter <已完成的0.6B SFT adapter目录> --baseline <对应SFT运行目录> --teacher-run <已完成的4B教师SFT运行目录> --output <空目录>`。在 CUDA BF16 环境中使用；输入、模型及 teacher adapter 均须可访问。教师运行必须通过完成状态、64步有限训练、保存后重载、32条验证集配对结果不退化等检查，否则停止。学生控制组与 OPD 从相同已保存 SFT adapter、相同训练 ID 和优化步数开始；两组训练均禁用 dropout，监督 token 预算不相同。评估使用从验证集确定性选出的16条，排除已有学生与教师验证 ID，不使用测试集；先在同一批样本上重评起始学生。生成的工具调用仅作为文本解析，绝不执行。

试跑输出为仓库外的空目录，`progress.json` 在每步或每条评测后原子更新；`status.json` 标示完成或失败。`selection.json`、`provenance.json`、各分支 `trajectories.json`、`predictions.json`、`metrics.json` 和 adapter 可用于追溯；中途失败可能留下不完整记录。该入口尚未在真实师生联合负载下验收，不代表蒸馏或显存结果。

配置 `configs/opd_validation_06b_4b.json` 将每组更新步数设为 16、rollout 上限设为 128，并在同一验证选择顺序中跳过前 16 个合格样本后选取 64 条；此配置用于后续验证，实际结果以运行记录为准。
