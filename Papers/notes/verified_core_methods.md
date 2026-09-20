# 核心方法与评测文献核验记录

核验日期：2026-09-20。本文件记录主 Agent 已实际打开并核对的 7 篇核心方法、后训练与评测来源；正式书目字段统一维护于 [references.bib](../references.bib)。阅读范围均明确区分元数据/摘要与全文精读。

## 1. Agarwal et al. (2024)：On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

### 核验结果

- **题名**：On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
- **作者**：Rishabh Agarwal；Nino Vieillard；Yongchao Zhou；Piotr Stanczyk；Sabela Ramos Garea；Matthieu Geist；Olivier Bachem。正式会议元数据使用 `Ramos Garea`；与 arXiv 页面作者显示存在差异，入库以会议页为准。
- **年份与 venue**：2024；International Conference on Learning Representations (ICLR 2024)。
- **arXiv**：2306.13649（v3，2024）
- **DOI**：未核验到独立会议 DOI；不把 arXiv DataCite DOI当作会议 DOI。
- **已打开验证 URL**：https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html；https://arxiv.org/abs/2306.13649

### 正式引用

正式引用键：`agarwal2024gkd`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

GKD 直接研究学生自生成错误上的在线蒸馏，适合作为本项目在线采样比例、教师反馈和散度选择的核心比较对象。需明确其在线样本策略与是否对采样过程反传，不能只按普通 KD 基线解读。

### 阅读范围

已核对会议元数据、摘要及 PDF 引言、算法 1、第 9 页相关工作与参考文献；未将摘要和局部阅读表述为全文精读。

## 2. Gu et al. (2024)：MiniLLM: Knowledge Distillation of Large Language Models

### 核验结果

- **题名**：MiniLLM: Knowledge Distillation of Large Language Models
- **作者**：Yuxian Gu；Li Dong；Furu Wei；Minlie Huang
- **年份与 venue**：2024；International Conference on Learning Representations (ICLR 2024)。
- **arXiv**：2306.08543。最新 arXiv 版本改题为 “MiniLLM: On-Policy Distillation of Large Language Models”；项目正式引用采用 ICLR 2024 会议题名，并记录该版本差异。
- **DOI**：未核验到独立会议 DOI。
- **已打开验证 URL**：https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html；https://arxiv.org/abs/2306.08543

### 正式引用

正式引用键：`gu2024minillm`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

MiniLLM 以 reverse KL 为核心进行序列级/策略级蒸馏，是 GKD 及本项目讨论 on-policy 分布匹配时的重要近邻。应把其序列目标、策略梯度与 GKD 的 token 级损失分开记录，避免将两者误写成同一算法。

### 阅读范围

已核对 ICLR 会议摘要、元数据及与 GKD 相关工作段落；未声称已完整复现或精读最新 arXiv 版本的全部推导。

## 3. Ko et al. (2025)：DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs

### 核验结果

- **题名**：DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs
- **作者**：Jongwoo Ko；Tianyi Chen；Sungnyun Kim；Tianyu Ding；Luming Liang；Ilya Zharkov；Se-Young Yun
- **年份与 venue**：2025；*Proceedings of the 42nd International Conference on Machine Learning*，PMLR volume 267，页码 31044–31062。
- **arXiv**：2503.07067
- **DOI**：未在 PMLR 页面核验到独立出版社 DOI。
- **已打开验证 URL**：https://proceedings.mlr.press/v267/ko25a.html；https://arxiv.org/abs/2503.07067

### 正式引用

正式引用键：`ko2025distillm2`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

DistiLLM-2 用对比式目标联合建模教师与学生生成数据，为项目分析“数据策略—损失函数”耦合提供近年方法参照。它可帮助比较纯 on-policy、off-policy replay 与对比学习式蒸馏的效率和质量取舍。

### 阅读范围

已核对 PMLR/arXiv 元数据与摘要，确认作者、题名、会议、卷页和方法定位；尚未把该条目当作全文精读或本项目已复现实验。

## 4. Schulman et al. (2017)：Proximal Policy Optimization Algorithms

### 核验结果

- **题名**：Proximal Policy Optimization Algorithms
- **作者**：John Schulman；Filip Wolski；Prafulla Dhariwal；Alec Radford；Oleg Klimov
- **年份与 venue**：2017；arXiv 预印本（未核验到独立正式会议出版版）。
- **arXiv**：1707.06347（v2）
- **DOI**：10.48550/arXiv.1707.06347（arXiv DataCite DOI）
- **已打开验证 URL**：https://arxiv.org/abs/1707.06347

### 正式引用

正式引用键：`schulman2017ppo`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

PPO 提供截断策略比率和稳定策略更新的经典背景，可用于解释在线策略优化中的更新约束与分布变化。它是后训练方法论参照，不应被直接当作本项目在线蒸馏的等价基线或蒸馏损失。

### 阅读范围

已核对 arXiv 元数据与摘要，阅读范围限于 PPO 的目标函数定位、策略更新稳定性和实验背景；未声称已完成全文理论证明核验。

## 5. Rafailov et al. (2023)：Direct Preference Optimization: Your Language Model is Secretly a Reward Model

### 核验结果

- **题名**：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **作者**：Rafael Rafailov；Archit Sharma；Eric Mitchell；Christopher D Manning；Stefano Ermon；Chelsea Finn。此处按会议元数据作者顺序记录；PDF/arXiv 页面存在 Stefano Ermon 与 Christopher D. Manning 的顺序差异，后续入库需保持会议页版本并在记录中保留差异。
- **年份与 venue**：2023；*Advances in Neural Information Processing Systems 36*（NeurIPS 2023）。
- **arXiv**：2305.18290
- **DOI**：10.52202/075280-2338
- **页码**：本次未核验，留空，不猜填。
- **已打开验证 URL**：https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html；https://arxiv.org/abs/2305.18290

### 正式引用

正式引用键：`rafailov2023dpo`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

DPO 是离线偏好优化方法，可作为后训练背景中“从偏好数据直接优化策略”的参照。项目应将其与在线策略蒸馏分开：DPO 的监督来自偏好对，不能直接等同于教师生成分布上的在线 token 蒸馏。

### 阅读范围

已阅读 NeurIPS 官方元数据与摘要，核对题名、会议年份、作者顺序和 DOI；未核验页码，也未声称已完成全文算法和实验表精读。

## 6. Cobbe et al. (2021)：Training Verifiers to Solve Math Word Problems

### 核验结果

- **题名**：Training Verifiers to Solve Math Word Problems
- **作者**：Karl Cobbe；Vineet Kosaraju；Mohammad Bavarian；Mark Chen；Heewoo Jun；Lukasz Kaiser；Matthias Plappert；Jerry Tworek；Jacob Hilton；Reiichiro Nakano；Christopher Hesse；John Schulman
- **年份与 venue**：2021；arXiv 预印本（GSM8K 原始来源；本次未核验独立会议出版版）。
- **arXiv**：2110.14168（v2）
- **DOI**：10.48550/arXiv.2110.14168（arXiv DataCite DOI）
- **已打开验证 URL**：https://arxiv.org/abs/2110.14168

### 正式引用

正式引用键：`cobbe2021gsm8k`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

该文是 GSM8K 数据集与 verifier 训练的原始来源，可为项目后续数学推理评测及奖励/验证器设计提供出处。当前只能把 GSM8K 作为评测候选，不能写成项目已经采用或已完成的实验设置。

### 阅读范围

已核对 arXiv 元数据与摘要，确认作者完整名单、版本、题名和数据集定位；未将数据集细节、模型配置或结果扩展为本项目事实。

## 7. Hendrycks et al. (2021)：Measuring Mathematical Problem Solving With the MATH Dataset

### 核验结果

- **题名**：Measuring Mathematical Problem Solving With the MATH Dataset
- **作者**：Dan Hendrycks；Collin Burns；Saurav Kadavath；Akul Arora；Steven Basart；Eric Tang；Dawn Song；Jacob Steinhardt
- **年份与 venue**：2021；*Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*，volume 1。
- **arXiv**：2103.03874
- **DOI**：未在本次核验的 NeurIPS Datasets and Benchmarks 官方页确认独立 DOI；保留 arXiv 标识。
- **已打开验证 URL**：https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html；https://arxiv.org/abs/2103.03874

### 正式引用

正式引用键：`hendrycks2021math`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

MATH 是较难数学问题求解的原始评测来源，可作为项目检验蒸馏后模型推理能力的候选数据集。论文中的 MATH 与常用 MATH500 子集必须分开记录，评测结果不能混写或互相替代。

### 阅读范围

已核对 NeurIPS Datasets and Benchmarks 官方元数据与摘要，确认作者、题名、年份、venue 和 arXiv ID；未声称已精读数据集全部类别或评测脚本。
