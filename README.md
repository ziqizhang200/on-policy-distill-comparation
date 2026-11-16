# 基于在线策略蒸馏的大语言模型后训练方法研究

哈尔滨工业大学本科毕业设计项目。计划在统一模型、任务和训练预算下，比较在线策略蒸馏中不同 KL 方向及全词表、Top-K、K1 监督粒度的效果、稳定性与资源开销。

## 当前状态

已建立包含 35 条文献的统一参考文献库，覆盖项目书原有 9 条。已完成并验收本机 Windows 单 GPU 实验环境，环境名为 `opdistill`；安装布局、固定依赖和验收方式见下文及环境说明。

项目尚无正式训练、评估实现和实验结果。当前模型方案为 Qwen3-0.6B LoRA 学生、Qwen3-4B 教师，面向 Function Calling；模型版本、数据版本和训练预算仍需受控实验设计与真实显存测试。

## 项目结构

- `configs/`：Windows Conda 基础环境、主依赖及完整 pip 版本锁。
- `scripts/`：环境安装、CUDA Toolkit 编译验收及 PyTorch/Qwen3/LoRA/NF4 验收。
- `docs/`：本机环境与安装验收记录。
- `Papers/`：统一文献库、导读、核验记录及本地论文资料。
- `Interface.md`：当前安装与验收 CLI；训练模块接口尚待设计。
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

尚未下载项目模型权重或数据集，尚未提供正式训练与评估命令。后续将在统一配置中记录模型和数据版本、随机种子、Git Commit、依赖、硬件及预算；正式实验结果待实验验证。环境验收中的随机微型 Qwen3 和 NF4 线性层只检查本机计算栈，不作为模型质量、性能或显存预算结论。

## 文献与协作入口

- [文献导读与项目书引用映射](Papers/README.md)
- [统一 BibTeX 文献库](Papers/references.bib)
- [项目进展与后续目标](STAGE.md)
- [公共接口](Interface.md)
- [协作与研究规范](AGENTS.md)

`Papers/notes/` 保存文献核验记录。论文原始资料、模板和输出目录沿用本地保存规则；仅将需要维护的项目文档和正式研究产物按版本控制规则提交。
