# 基础文献核验笔记

核验日期：2026-09-20。以下条目均实际打开了原始论文、出版方页面或官方元数据页；BibTeX 只填写已由这些页面确认的字段。`阅读范围`记录本次实际阅读到的元数据、摘要和正文范围。

## 1. Hinton et al. (2015)：Distilling the Knowledge in a Neural Network

### 核验结果

- **题名**：Distilling the Knowledge in a Neural Network
- **作者**：Geoffrey Hinton；Oriol Vinyals；Jeff Dean。arXiv 正文与元数据显示 `Jeff Dean`；Google Research 页面显示 `Jeffrey Dean`，此处保留论文元数据中的 `Jeff Dean`。
- **年份与 venue**：2015；Google Research 的正式出版条目标为 *NIPS Deep Learning and Representation Learning Workshop (2015)*。arXiv 元数据的 comments 写作 *NIPS 2014 Deep Learning Workshop*，两处年份标注不一致；本库引用 2015 年 arXiv 版本，保留此差异，不推断会议活动年份。
- **arXiv**：1503.02531
- **DOI**：10.48550/arXiv.1503.02531（arXiv DataCite DOI；未发现独立出版社 DOI）
- **已打开验证 URL**：
  - https://arxiv.org/abs/1503.02531
  - https://arxiv.org/html/1503.02531
  - https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/

### 正式引用

正式引用键：`hinton2015distilling`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

该工作奠定教师—学生蒸馏范式，展示用教师软目标和温度平滑传递“暗知识”可以让小模型保留大模型能力。项目可将其作为离线蒸馏基线，并与在线策略蒸馏的序列级目标比较。

### 阅读范围

已阅读 arXiv 元数据、摘要、正文引言、§2 Distillation、§2.1 logits 匹配，以及 §6 Soft Targets as Regularizers；重点核对教师软目标、温度参数、硬/软目标组合及压缩动机，未据此扩展未核验的实验字段。

## 2. Kim and Rush (2016)：Sequence-Level Knowledge Distillation

### 核验结果

- **题名**：Sequence-Level Knowledge Distillation
- **作者**：Yoon Kim；Alexander M. Rush
- **年份与 venue**：2016；EMNLP 2016（*Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*），页码 1317–1327，Austin, Texas。
- **DOI**：10.18653/v1/D16-1139
- **arXiv**：1606.07947
- **已打开验证 URL**：
  - https://aclanthology.org/D16-1139/
  - https://aclanthology.org/D16-1139.pdf
  - https://arxiv.org/abs/1606.07947

### 正式引用

正式引用键：`kim-rush-2016-sequence`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

SeqKD 把教师 beam search 生成的序列作为静态训练集，首次系统展示序列级蒸馏在 NMT 中的效果，是本项目比较在线策略蒸馏时的关键离线基线。其静态状态分布也正好对应 exposure bias 的来源。

### 阅读范围

已阅读 ACL 元数据与 PDF 摘要、§1 Introduction、§2.2 Knowledge Distillation、§3 的 word-level/sequence-level 蒸馏说明、§6 Related Work 和 §7 Conclusion，核对 beam 生成数据集、序列级目标与加速结论。

## 3. Ross, Gordon, and Bagnell (2011)：A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

### 核验结果

- **题名**：A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning
- **作者**：Stéphane Ross；Geoffrey J. Gordon；J. Andrew Bagnell。PMLR BibTeX 将第三作者显示为 Drew Bagnell；PDF 作者行采用 `J. Andrew Bagnell`，两者指同一作者，本文按 PDF 完整署名记录。
- **年份与 venue**：2011；*Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*，PMLR volume 15，页码 627–635，Fort Lauderdale, FL, USA。
- **arXiv**：1011.0686（arXiv 首次提交 2010-11-02；会议出版年份为 2011）
- **DOI**：未发现独立出版社 DOI；保留 PMLR 与 arXiv 标识。
- **已打开验证 URL**：
  - https://proceedings.mlr.press/v15/ross11a
  - https://proceedings.mlr.press/v15/ross11a/ross11a.pdf
  - https://arxiv.org/abs/1011.0686

### 正式引用

正式引用键：`ross11dagger`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

DAgger 将模仿学习转化为无悔在线学习，通过聚合学生自身访问的状态并查询专家动作来缓解分布偏移。它为本项目的在线策略蒸馏提供理论原型：训练数据应逐步覆盖学生实际生成的前缀。

### 阅读范围

已阅读 PMLR 元数据与摘要、PDF/HTML 的 §1 Introduction、§2 监督模仿学习背景、§3 Dataset Aggregation（DAgger）和 §4 No-Regret Online Learning 理论分析，核对二次误差、状态分布、专家混合与线性 regret 结论。

## 4. Lin et al. (2020)：Autoregressive Knowledge Distillation through Imitation Learning（ImitKD）

### 名称校正

检索时的候选名 “Imitation Learning for Non-Autoregressive Neural Machine Translation” 并非 Lin 等人的论文。ACL 官方页显示该题作者为 Bingzhen Wei、Mingxuan Wang、Hao Zhou、Junyang Lin、Xu Sun，发表于 ACL 2019（P19-1125，DOI 10.18653/v1/P19-1125），主题是非自回归翻译。Lin 等人关于 imitation-based knowledge distillation 的正确原始正式论文是下面的 EMNLP 2020 论文；项目引用应采用本条。

- **候选标题核验 URL**：https://aclanthology.org/P19-1125/

### 核验结果

- **题名**：Autoregressive Knowledge Distillation through Imitation Learning
- **作者**：Alexander Lin；Jeremy Wohlwend；Howard Chen；Tao Lei
- **年份与 venue**：2020；EMNLP 2020（*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*），页码 6121–6133，Online。
- **DOI**：10.18653/v1/2020.emnlp-main.494
- **arXiv**：2009.07253
- **已打开验证 URL**：
  - https://aclanthology.org/2020.emnlp-main.494/
  - https://aclanthology.org/2020.emnlp-main.494.pdf
  - https://arxiv.org/abs/2009.07253

### 正式引用

正式引用键：`lin-etal-2020-autoregressive`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

ImitKD 将自回归学生视作策略、教师视作 oracle，让学生探索自身生成状态并由教师逐 token 修正，直接把 DAgger 式在线分布对齐用于蒸馏。它是本项目研究在线策略蒸馏最直接的先行工作与实现基线。

### 阅读范围

已阅读 ACL 元数据与 PDF 摘要、§1 Introduction、§2.1 自回归蒸馏、§2.2 Distillation as Imitation Learning、§2.3 SeqKD as Behavioral Cloning、§3.1–§3.4 ImitKD 算法与采样/替换策略，以及 §4 Related Work；同时阅读候选 Wei 等 ACL 2019 页面的元数据与摘要以完成名称排除。

## 5. Huszár (2015)：How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?

### 核验结果

- **题名**：How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?
- **作者**：Ferenc Huszár
- **年份与 venue**：2015；arXiv 预印本（未核验到独立会议或期刊正式出版版）。
- **arXiv**：1511.05101
- **DOI**：10.48550/arXiv.1511.05101（arXiv DataCite DOI）
- **已打开验证 URL**：
  - https://arxiv.org/abs/1511.05101
  - https://arxiv.org/html/1511.05101

### 正式引用

正式引用键：`huszar2015not`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

该文从目标函数角度指出 scheduled sampling 在自生成前缀下并不一致，揭示训练分布与生成分布错配的根源。它支持项目对在线采样、教师纠错和 exposure bias 的动机分析，但不能直接当作蒸馏算法结果。

### 阅读范围

已阅读 arXiv 元数据与摘要、§1 Introduction、§2 自回归序列模型、§3 symptoms、§4 Scheduled Sampling（含 §4.1 KL 推导）、§5 Diagnosis、§6 Generalised Adversarial Training 与 §7 Conclusions。

## 6. Wen et al. (2023)：f-Divergence Minimization for Sequence-Level Knowledge Distillation

### 核验结果

- **题名**：f-Divergence Minimization for Sequence-Level Knowledge Distillation
- **作者**：Yuqiao Wen；Zichao Li；Wenyu Du；Lili Mou
- **年份与 venue**：2023；ACL 2023，*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*，页码 10817–10834，Toronto, Canada。
- **DOI**：10.18653/v1/2023.acl-long.605
- **arXiv**：2307.15190
- **已打开验证 URL**：
  - https://aclanthology.org/2023.acl-long.605/
  - https://aclanthology.org/2023.acl-long.605.pdf
  - https://arxiv.org/abs/2307.15190

### 正式引用

正式引用键：`wen-etal-2023-f`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

f-DISTILL 把序列级蒸馏统一表述为广义 f-divergence 最小化，分析 KL 的 mode averaging 与 reverse KL 的 mode collapse，并给出 JS/TVD 及逐步分解。它为项目比较不同蒸馏损失及其与在线数据策略的耦合提供理论基线。

### 阅读范围

已阅读 ACL 元数据与 PDF 摘要、§1 Introduction、§2.1 经典 KD 与缺陷、§2.2 f-DISTILL（KL/RKL/JS/TVD）、§2.3 高效近似与离线教师采样，以及 §3.1 实验设置；核对方法命名、分解结论和数据集范围。

## 7. Ko et al. (2024)：DistiLLM: Towards Streamlined Distillation for Large Language Models

### 核验结果

- **题名**：DistiLLM: Towards Streamlined Distillation for Large Language Models
- **作者**：Jongwoo Ko；Sungnyun Kim；Tianyi Chen；Se-Young Yun
- **年份与 venue**：2024；ICML 2024，*Proceedings of the 41st International Conference on Machine Learning*，PMLR volume 235，页码 24872–24895。
- **arXiv**：2402.03898（v2，2024-07-03）
- **DOI**：10.48550/arXiv.2402.03898（arXiv DataCite DOI）；PMLR 页面未列独立出版社 DOI。
- **已打开验证 URL**：
  - https://proceedings.mlr.press/v235/ko24c.html
  - https://arxiv.org/abs/2402.03898
  - https://arxiv.org/html/2402.03898

### 正式引用

正式引用键：`pmlr-v235-ko24c`；书目字段统一维护于 [references.bib](../references.bib)。

### 项目关联

DistiLLM 面向自回归 LLM，联合提出 skew KLD 与自适应 off-policy 学生生成样本调度，在缓解训练—推理错配的同时降低持续 on-policy 生成成本。它是本项目在线策略蒸馏效率、散度选择和 replay 设计的直接近年参照。

### 阅读范围

已阅读 PMLR 元数据与摘要、arXiv 元数据/摘要、§1 Introduction、§2.1–§2.2 背景与现有蒸馏缺陷、§3.1 skew KLD、§3.2 adaptive off-policy、§4 实验概览及 §5 分析入口；重点核对损失、学生生成样本与 replay 调度，不据此补写未阅读的附录参数。
