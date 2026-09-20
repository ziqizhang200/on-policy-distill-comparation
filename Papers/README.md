# 项目参考文献

本目录为“基于在线策略蒸馏的大语言模型后训练方法研究”维护统一参考文献。截至 **2026-09-20**，已收录 **23 条**：项目书原有 9 条全部覆盖，补充 14 条。包含学术论文、预印本、模型技术报告，以及 1 条官方技术博客；不将它们统称为已同行评审论文。

正式书目唯一来源是 [references.bib](references.bib)。下表给出主题、用途和阅读优先级，作者、准确题名、年份、出版信息、DOI/arXiv 标识与访问日期见 BibTeX。核验记录保存实际访问来源和阅读范围，元数据核实不等于全文精读，更不代表本项目已复现实验。

## 阅读顺序与分类索引

优先级：**A** 为直接服务当前实验设计的必读文献；**B** 为理论、后训练或模型背景；**C** 为确定评测方案时阅读的候选来源。以下用途是本项目的研究规划，不是已有实验结果。

### 蒸馏与分布偏移基础

| 引用键 | 文献 | 年份与引用类型 | 优先级 | 本项目用途 |
|---|---|---|---|---|
| `hinton2015distilling` | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | 2015，arXiv | B | 教师软目标、温度与传统 KD 基线；workshop 年份存在来源差异，本库引用预印本。 |
| `kim-rush-2016-sequence` | [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/) | 2016，EMNLP | A | 静态教师生成序列的离线对照。 |
| `ross11dagger` | [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a) | 2011，AISTATS | B | DAgger 与学生访问状态上的专家监督，解释分布偏移。 |
| `lin-etal-2020-autoregressive` | [Autoregressive Knowledge Distillation through Imitation Learning](https://aclanthology.org/2020.emnlp-main.494/) | 2020，EMNLP | A | ImitKD，在线状态分布与模仿学习式蒸馏的先行工作。 |
| `huszar2015not` | [How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?](https://arxiv.org/abs/1511.05101) | 2015，arXiv | B | 自生成前缀与目标一致性；广义散度背景。 |
| `wen-etal-2023-f` | [f-Divergence Minimization for Sequence-Level Knowledge Distillation](https://aclanthology.org/2023.acl-long.605/) | 2023，ACL | A | f-divergence、序列目标与逐 token 近似。 |

### 在线策略蒸馏与失败模式

| 引用键 | 文献 | 年份与引用类型 | 优先级 | 本项目用途 |
|---|---|---|---|---|
| `agarwal2024gkd` | [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) | 2024，ICLR | A | GKD：学生采样比例、FKL/RKL/JSD 的统一比较。 |
| `gu2024minillm` | [MiniLLM: Knowledge Distillation of Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html) | 2024，ICLR | A | 序列 RKL 与策略梯度；和 GKD 的梯度处理分别定义。 |
| `pmlr-v235-ko24c` | [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://proceedings.mlr.press/v235/ko24c.html) | 2024，ICML | A | skew KL 与学生样本复用，作为效率比较参照。 |
| `ko2025distillm2` | [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs](https://proceedings.mlr.press/v267/ko25a.html) | 2025，ICML | B | 教师/学生数据与对比目标的配合。 |
| `lu2025onpolicydistillation` | [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 2025，官方博客 | A | 学生 rollout 与逐 token RKL 反馈的工程基线。 |
| `li2026rethinkingopdphenomenology` | [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016) | 2026，arXiv v2 | A | 教师兼容性、Top-K 重叠与熵差；全词表/Top-K/采样 token 比较。 |
| `fu2026rethinkingopdoneshot` | [Rethinking On-Policy Distillation of Large Language Models II: One Training Example](https://arxiv.org/abs/2609.04172) | 2026，arXiv v1 | A | 提示数量、状态覆盖和训练步效率的消融依据。 |
| `fu2026revisitingopd` | [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562) | 2026，arXiv v2 | A | teacher Top-K、前缀漂移、特殊 token 与训练稳定性。 |
| `zhu2026hybridpolicydistillation` | [Hybrid Policy Distillation for LLMs](https://arxiv.org/abs/2604.20244) | 2026，引用 arXiv v2；ICML 名录已确认 | A | FKL/RKL 配合、混合策略与时间/显存开销。 |
| `song2026survey` | [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) | 2026，arXiv v4，持续更新 | B | 研究分类与扩展检索入口；算法细节回溯原始论文。 |

### 模型报告与后训练背景

| 引用键 | 文献 | 年份与引用类型 | 优先级 | 本项目用途 |
|---|---|---|---|---|
| `qwen3technicalreport` | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | 2025，技术报告 | B | 候选模型家族与后训练背景；团队署名依据官方模型卡。 |
| `deepseekai2026deepseekv4` | [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) | 2026，技术报告 | B | 多教师、全词表 OPD 与工程效率；不意味着本项目采用该大模型。 |
| `xiaomillmcore2026mimov2flash` | [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780) | 2026，技术报告 v2 | B | 多教师 OPD 与领域教师反馈。 |
| `schulman2017ppo` | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 2017，arXiv | B | 策略更新与在线训练背景，不当作蒸馏损失的同义词。 |
| `rafailov2023dpo` | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) | 2023，NeurIPS | B | 离线偏好优化背景，区分偏好对与教师分布监督。 |

### 评测候选的原始来源

| 引用键 | 文献 | 年份与引用类型 | 优先级 | 本项目用途 |
|---|---|---|---|---|
| `cobbe2021gsm8k` | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) | 2021，arXiv | C | GSM8K 原始出处；实际评测仍须确定数据版本、答案抽取和采样设置。 |
| `hendrycks2021math` | [Measuring Mathematical Problem Solving With the MATH Dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html) | 2021，NeurIPS Datasets and Benchmarks | C | 较难数学推理候选；MATH 与 MATH500 子集不能混用。 |

建议先读 GKD、MiniLLM 和项目书中的 Rethinking／Revisiting OPD，再读 DistiLLM、HPD 及基础引用。每篇核心方法后续建立统一比较表：采样策略、监督支持集、KL 方向、梯度估计、tokenizer 假设、生成预算和稳定化措施。

## 项目书原有九条的覆盖关系

来源为根目录项目书《基于On-Policy Distillation的LLM后训练方法研究.docx》，本次只提取文字，未修改原文或检查其排版。原作者缩写及 et al. 属于正常参考文献简写，以下是信息补全与引用版本选择。

| 项目书编号 | 文献简称 | 正式引用键 | 补全与版本说明 |
|---|---|---|---|
| 1 | Thinking Machines OPD | `lu2025onpolicydistillation` | 补 Kevin Lu 与团队署名、DOI；明确官方博客类型。 |
| 2 | Rethinking OPD I | `li2026rethinkingopdphenomenology` | 完整作者与 v2；保留原 arXiv 编号。 |
| 3 | Rethinking OPD II | `fu2026rethinkingopdoneshot` | 完整作者与 2026-09-03 的 v1。 |
| 4 | Revisiting OPD | `fu2026revisitingopd` | 完整作者与 v2。 |
| 5 | HPD | `zhu2026hybridpolicydistillation` | 完整四作者；arXiv 及 ICML 官方下载名录均指向该工作，尚未核实论文集卷页，暂引用预印本。 |
| 6 | OPD Survey | `song2026survey` | 两作者，v4；页面标注 Ongoing Work。 |
| 7 | Qwen3 | `qwen3technicalreport` | 补 arXiv:2505.09388；沿用官方支持的 Qwen Team 团体署名。 |
| 8 | DeepSeek-V4 | `deepseekai2026deepseekv4` | 用官方 DeepSeek-AI 团体署名；编号与页面提交月份不一致，仅保留原文证据，不猜测原因。 |
| 9 | MiMo-V2-Flash | `xiaomillmcore2026mimov2flash` | 用官方 Xiaomi LLM-Core Team 团体署名，v2。 |

## 引用追溯与核验记录

已实际核对的引用链：

- [Rethinking OPD I 的相关工作与参考文献](https://arxiv.org/html/2604.13016v1) → GKD、MiniLLM、Hinton KD、Sequence-Level KD。
- [GKD 正式论文第 9 页相关工作及参考文献](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf) → DAgger、ImitKD、f-DISTILL；其散度部分还引用 Huszár。
- DistiLLM、DistiLLM-2、PPO、DPO 和评测文献通过主题检索补充，不把主题相关性冒写为已经核对的直接引用关系。

逐条来源、版本差异和阅读范围见：[原方法文献 1—5](notes/verified_seed_methods.md)、[原综述与报告 6—9](notes/verified_seed_reports.md)、[基础文献](notes/verified_foundations.md)、[核心方法与评测](notes/verified_core_methods.md)。[ICML 2026 官方名录](https://icml.cc/Downloads/2026)也列出 HPD；单篇海报链接本次未成功读取，未补写未核实卷页。

## 后续实验设计需要先确认的事项

- K1 的 sampled-token 估计与 Top-1 argmax 选择不同。Top-K 必须说明支持集来自教师、学生还是交集，以及是否重归一化。
- GKD 的固定采样轨迹 token 损失与 MiniLLM 的序列目标/策略梯度须分别定义；仅写“RKL”不足以复现实验。
- Rethinking OPD I 的子集重归一化公式在单元素支持集上产生零 KL，而正文另有 Top-1 实验讨论。这是本次公式检查发现的**待查实现问题**，不是已确认的论文错误；需对照作者代码后解释。
- 教师兼容性、tokenizer、特殊 token、rollout 长度、温度及训练预算可能影响比较。文献结论需在本项目设置下重新验证，尚不能预写为毕业论文实验结论。

## 维护约定

1. 新文献先访问原始论文或正式出版页，核实题名、作者、年份、venue 与标识；无法确认的条目留在待检索记录中，不进入正式库。
2. 优先选可核验的正式会议/期刊版本；预印本和技术博客明确标注类型。同一工作不同版本不重复计数，arXiv DOI 不作为会议出版 DOI。
3. 使用稳定引用键；团队作者用花括号保护。更新版本时同步修改本索引、BibTeX 与核验记录，并保留实质差异。
4. 论文从本目录的单一 `references.bib` 引用，不复制第二份独立维护的库。模板特殊要求需另行确认。
5. 当前完成书目核验与部分章节阅读；全文精读、代码对照、实验复现分别记录，不互相代替。
