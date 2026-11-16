# 基于在线策略蒸馏的大语言模型后训练方法研究

哈尔滨工业大学本科毕业设计项目。当前开题报告设置三个研究问题：第一项联合设计决策状态的监督位置与位置条件化散度目标；第二项研究保留关键决策 token 和尾部质量的稀疏教师反馈；第三项设计统一测试系统，将同前缀配对回放与独立闭环训练连接起来，追踪结构化调用错误和分项资源成本。毕业论文拟围绕三项研究分别成章。以上均为待实验验证的方案。

## 当前状态

已建立统一参考文献库；截至 2026-10-07，`Papers/references.bib` 含 45 条 BibTeX 条目，新增条目的核验状态仍需逐条复查。开题答辩已通过，正式报告按“决策状态与散度目标的联合方法—稀疏反馈方法—统一测试系统”组织研究内容，尚待导师审阅与实验验证。尾部类别及学生感知支持集均有先行研究，本文的联合方案目前只是待验证假设。尚未提供本项目的训练、评估及实验复现命令，也没有本项目实验结果。

已完成并验收本机 Windows 单 GPU 实验环境 `opdistill`；安装布局、固定依赖和验收方式见下文及环境说明。

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
- [开题报告修改与研究方案交接](Papers/OPENING_REPORT_HANDOFF.md)
- [协作与研究规范](AGENTS.md)

`Papers/notes/` 保存文献核验记录。论文原始资料和模板沿用本地保存规则；部分正式报告与答辩输出文件已纳入版本控制，具体以 Git 跟踪状态为准。
