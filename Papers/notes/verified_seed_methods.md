# 原始文献核验：在线策略蒸馏方法种子文献

核验日期：2026-09-20（研究时间截点）

本笔记保存原始发布页面的核验依据；正式书目字段统一维护于 [references.bib](../references.bib)。第 1 条使用 Thinking Machines Lab 官方文章及其文章内置引用；第 2—5 条核对 arXiv 摘要页与 HTML 正文。HPD 的 ICML 2026 信息另经[官方下载名录](https://icml.cc/Downloads/2026)确认；单篇海报页本次未成功读取，未补写会议论文集卷期或页码，当前引用 arXiv v2。

实现阅读边界（项目待查，不是对原文的错误断言）：sampled-token 目标是在学生分布上采样一个 token，并不等于取 Top-1 argmax；Top-$K$ 目标必须同时记录支持集来自教师还是学生，以及是否在支持集内重归一化。按第 2 条 HTML §2.2 式 (5)，若 $k=1$ 且 $p/q$ 都在单元素支持集内重归一化，子集 KL 在数学上恒为 0；同文 §6.3 又讨论 Top-1 训练，因此应结合原文上下文和代码核查后再形成实现结论，不能直接写成论文错误。

## 核验总表

| 编号 | 精确题名 | 作者（完整） | 年份 | venue / 类型 | ID | 核验状态 |
|---|---|---|---:|---|---|---|
| 1 | *On-Policy Distillation* | Kevin Lu；Thinking Machines Lab | 2025 | Thinking Machines Lab: Connectionism（官方文章） | DOI `10.64434/tml.20251026` | 已核实 |
| 2 | *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* | Yaxuan Li；Yuxin Zuo；Bingxiang He；Jinqian Zhang；Chaojun Xiao；Cheng Qian；Tianyu Yu；Huan-ang Gao；Wenkai Yang；Zhiyuan Liu；Ning Ding | 2026 | arXiv 预印本，cs.LG（并列 cs.AI、cs.CL） | `arXiv:2604.13016`，当前 v2 | 已核实 |
| 3 | *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* | Zixuan Fu；Bingxiang He；Yuxin Zuo；Haohuan Huang；Jinqian Zhang；Ruhang Xiao；Cheng Qian；Qinyu Luo；Huan-ang Gao；Yudong Wang；Zhiyuan Liu；Ning Ding；Chaojun Xiao | 2026 | arXiv 预印本，cs.AI（并列 cs.CL） | `arXiv:2609.04172`，v1 | 已核实 |
| 4 | *Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes* | Yuqian Fu；Haohuan Huang；Kaiwen Jiang；Jiacai Liu；Zhuo Jiang；Yuanheng Zhu；Dongbin Zhao | 2026 | arXiv 预印本，cs.LG（并列 cs.AI、cs.CL） | `arXiv:2603.25562`，当前 v2 | 已核实 |
| 5 | *Hybrid Policy Distillation for LLMs* | Wenhong Zhu；Ruobing Xie；Rui Wang；Pengfei Liu | 2026 | arXiv 预印本，cs.CL；arXiv Comments 标注 ICML 2026 | `arXiv:2604.20244`，当前 v2 | 已核实 |

## 1. On-Policy Distillation

### 来源与书目信息

- 官方原始页面：[Thinking Machines Lab — On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- 页面显示：Kevin Lu 与 Thinking Machines Lab 合作，2025-10-27 发布；页面引用部分将日期写为 2025 年 10 月。
- 原文指定的 DOI：[`10.64434/tml.20251026`](https://doi.org/10.64434/tml.20251026)。DOI 字符串中的日期与网页显示的 10 月 27 日不同，均原样保留，不据此推断新的发布日期。

正式引用键：`lu2025onpolicydistillation`；书目字段统一维护于 [references.bib](../references.bib)。

### 摘要/关键节阅读范围

已阅读官方页面的 “On-policy distillation — best of both worlds”、 “Implementation” 下的 “Loss function: reverse KL” 和 “Pseudocode”、 “Distillation for reasoning”、 “Distillation for personalization”、 “On-policy learning as a tool for continual learning”、 “Conclusion” 与 “Citation”。关键原文位置可由页面内目录直接定位。

### 与本项目的关联（约 50—100 字）

该文给出在线策略蒸馏的工程基线：学生生成自身轨迹，教师在学生访问的前缀上提供逐 token 评分，并以 reverse KL 作为优势信号。它为逐 token RKL 提供工程基线；其他 KL 方向与 Top-K 粒度的比较需结合 GKD、DistiLLM 和相关失败模式论文。

### 信息补全与版本说明

用户条目“Lu,K & Thinking Machines Lab. Oct2025”可补全为 **Kevin Lu and Thinking Machines Lab**；精确网页发布日期是 **2025-10-27**，原文引用的 venue 是 **Thinking Machines Lab: Connectionism**，不是已核验的会议或期刊。原文 BibTeX 还给出 DOI `10.64434/tml.20251026`。

## 2. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe

### 来源与书目信息

- 摘要/元数据原始页：[arXiv:2604.13016](https://arxiv.org/abs/2604.13016)
- 全文原始页：[arXiv HTML v2](https://arxiv.org/html/2604.13016v2)
- DOI：[10.48550/arXiv.2604.13016](https://doi.org/10.48550/arXiv.2604.13016)
- arXiv 摘要页显示：2026-04-14 提交，2026-04-15 修订为 v2；引用标识为 `arXiv:2604.13016 [cs.LG]`，另列 cs.AI、cs.CL。

正式引用键：`li2026rethinkingopdphenomenology`；书目字段统一维护于 [references.bib](../references.bib)。

### 摘要/关键节阅读范围

已阅读摘要以及 HTML 全文 §2.2 “On-Policy Distillation”（sampled-token、full-vocabulary、top-$k$ 三种粒度）、§3 “Phenomenology of On-Policy Distillation”、§4 “Mechanism of On-Policy Distillation”、§5 “Practical Recipe”和 §6 “Discussion”。重点记录：思维模式一致性、教师必须带来学生尚未获得的新能力、学生访问状态上的高概率 token 对齐、off-policy cold start 与 teacher-aligned prompt selection，以及长轨迹深度增加时奖励质量下降。

### 与本项目的关联（约 50—100 字）

该文把 OPD 的有效性拆成教师—学生思维模式兼容性和教师新能力两项条件，并明确比较 sampled-token、全词表和 Top-K 监督。其 overlap ratio、熵差及共享高概率 token 分析可转化为本项目的稳定性指标，cold start 与提示选择也可作为控制变量或消融项。

### 信息补全与版本说明

用户条目只列 “Li Y, Zuo Y, He B et al.”，原始 arXiv 页确认完整作者共 11 人：**Yaxuan Li、Yuxin Zuo、Bingxiang He、Jinqian Zhang、Chaojun Xiao、Cheng Qian、Tianyu Yu、Huan-ang Gao、Wenkai Yang、Zhiyuan Liu、Ning Ding**。年份为 2026；当前页面为 v2（2026-04-15），本次引用已核验的 arXiv 版本。

## 3. Rethinking On-Policy Distillation of Large Language Models II: One Training Example

### 来源与书目信息

- 摘要/元数据原始页：[arXiv:2609.04172](https://arxiv.org/abs/2609.04172)
- 全文原始页：[arXiv HTML v1](https://arxiv.org/html/2609.04172v1)
- DOI：[10.48550/arXiv.2609.04172](https://doi.org/10.48550/arXiv.2609.04172)
- arXiv 摘要页显示：2026-09-03 提交，当前 v1；引用标识为 `arXiv:2609.04172 [cs.AI]`，另列 cs.CL。

正式引用键：`fu2026rethinkingopdoneshot`；书目字段统一维护于 [references.bib](../references.bib)。

### 摘要/关键节阅读范围

已阅读摘要以及 HTML 全文 §3 “One-Shot OPD”、§4.1 “State Coverage of One-Shot OPD”、§4.2 “Diversity Expands State Coverage”、§5 “Algorithm Perspective: Slow Alignment”、§6 “One-Shot Extends to Multi-Teacher OPD”和 §7 “Discussion”。原文报告单一 query 到第 300 步覆盖 full-data OPD 状态空间的 71.5%，16 个语义多样 query 达到 98.9%；同时指出对齐速度会持续变慢。

### 与本项目的关联（约 50—100 字）

该文把数据量问题转化为学生 rollout 产生的状态覆盖问题，说明少量 query 也能提供广泛 token 监督，而吸收监督的算法速度成为瓶颈。它适合支撑本项目的效率实验：固定训练步数、比较 query 数与语义多样性，并将覆盖率、收敛速度和稳定性分开记录。

### 信息补全与版本说明

用户条目 “Fu Z, He B, Zuo Y et al.” 省略了 10 位作者；完整作者为 **Zixuan Fu、Bingxiang He、Yuxin Zuo、Haohuan Huang、Jinqian Zhang、Ruhang Xiao、Cheng Qian、Qinyu Luo、Huan-ang Gao、Yudong Wang、Zhiyuan Liu、Ning Ding、Chaojun Xiao**，共 13 人。原始页显示提交日为 **2026-09-03**，截至核验日只有 v1；venue 只能确认是 arXiv 预印本，不能补写未核实的会议名称。

## 4. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

### 来源与书目信息

- 摘要/元数据原始页：[arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
- 全文原始页：[arXiv HTML v2](https://arxiv.org/html/2603.25562v2)
- DOI：[10.48550/arXiv.2603.25562](https://doi.org/10.48550/arXiv.2603.25562)
- arXiv 摘要页显示：2026-03-26 提交，2026-04-27 修订为 v2；引用标识为 `arXiv:2603.25562 [cs.LG]`，另列 cs.AI、cs.CL。

正式引用键：`fu2026revisitingopd`；书目字段统一维护于 [references.bib](../references.bib)。

### 摘要/关键节阅读范围

已阅读摘要、§2.1 “From reverse-KL to token-level OPD”、§2.2 “Why sampled-token OPD is brittle in practice”、§3 “Method”、§4 “Experiments”和 §5 “Conclusion”。关键结论包括：token-level OPD 相对 sequence-level reverse KL 有偏但最坏方差界更紧；采样 token 信号不平衡、学生漂移前缀上的教师指导不可靠、tokenizer/特殊 token 不匹配；方法使用 teacher top-$K$ local support matching、截断 reverse KL、top-$p$ rollout 与 special-token masking。

### 与本项目的关联（约 50—100 字）

该文直接对应本项目的稳定性主线：它把 sampled-token OPD 的长轨迹不稳定归因于信号失衡、前缀漂移和 tokenizer 差异，并以 teacher Top-K 局部支持替代单 token 比较。项目可据此实现全词表、Top-K、K1 等公平消融，分别记录方差、熵塌缩、训练曲线和任务得分。

### 信息补全与版本说明

用户条目只列 “Fu Y, Huang H, Jiang K et al.”；原始页完整作者为 **Yuqian Fu、Haohuan Huang、Kaiwen Jiang、Jiacai Liu、Zhuo Jiang、Yuanheng Zhu、Dongbin Zhao**，共 7 人。标题中的大小写与连字符应保留为 *Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes*；当前引用版本为 v2（2026-04-27），venue 仍应写 arXiv 预印本，不应把页面中的 arXiv 分类误写成会议 venue。

## 5. Hybrid Policy Distillation for LLMs

### 来源与书目信息

- 摘要/元数据原始页：[arXiv:2604.20244](https://arxiv.org/abs/2604.20244)
- 全文原始页：[arXiv HTML v2](https://arxiv.org/html/2604.20244v2)
- DOI：[10.48550/arXiv.2604.20244](https://doi.org/10.48550/arXiv.2604.20244)
- arXiv 摘要页显示：2026-04-22 提交，2026-08-08 修订为 v2；引用标识为 `arXiv:2604.20244 [cs.CL]`，Comments 字段为 **ICML 2026**。

正式引用键：`zhu2026hybridpolicydistillation`；书目字段统一维护于 [references.bib](../references.bib)。

### 摘要/关键节阅读范围

已阅读摘要、§1 “Introduction”、§3.2 “KD via KL Divergence”（FKLD 与 RKLD）、§4 “Hybrid Policy Distillation”、§5 “Experiments”、§6 “Ablation Study”、§7.1 “Computational Efficiency”，以及 Appendix B “Limitation”和 Appendix C “Gradient of Jensen-Shannon Divergence”。重点记录：HPD 用 negative-$K_1$ token-level reweighting 同时利用 forward/reverse KL，结合离线数据与轻量近似 on-policy sampling；全文还报告了显存、训练时间和同 tokenizer 限制。

### 与本项目的关联（约 50—100 字）

该文覆盖项目计划中的 FKL、RKL、JSD 和效率比较：HPD 把方向互补的 KL 转成 token 级权重，兼顾 mode coverage 与 mode seeking，并用少量学生采样降低全词表匹配成本。其显存、时延、消融和同 tokenizer 限制可用于设计统一的稳定性—效率评测表。

### 信息补全与版本说明

项目书使用前三位作者加 et al. 的简写；完整作者为 **Wenhong Zhu、Ruobing Xie、Rui Wang、Pengfei Liu**。精确题名为 *Hybrid Policy Distillation for LLMs*；arXiv 当前版本是 v2。ICML 官方名录已确认该题目，但正式论文集卷页尚未核实，BibTeX 保持 `@misc` 并引用 arXiv DOI。

## 未核实条目

本次五项均已在指定原始页面找到并完成元数据核验；没有需要移入“未核实”区的条目。正式书目字段见 [references.bib](../references.bib)。
