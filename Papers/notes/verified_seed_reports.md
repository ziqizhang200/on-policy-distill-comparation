# 参考文献 6 至 9 核验记录

核验日期：2026-09-20

元数据统一维护于 [references.bib](../references.bib)；本记录保存核验依据、版本差异和阅读范围。综述阅读至元数据与摘要，模型报告的正文阅读范围见各条说明。

## 6. A Survey of On-Policy Distillation for Large Language Models

- 项目书原条目：`Song M, Zheng M. A survey of on-policy distillation for large language models[J]. arXiv preprint arXiv:2604.00626, 2026.`
- 核验题名：A Survey of On-Policy Distillation for Large Language Models。
- 作者：Mingyang Song；Mao Zheng。arXiv 页面列出这两位作者，未发现需要使用团队署名的情况。
- 出版类型：arXiv 预印本；页面备注为 `Ongoing Work`，未给出期刊或会议名称。
- 日期与版本：v1 提交于 2026-04-01；当前页面版本为 v4，最后修订于 2026-06-18。年份保留为 2026。
- 正式引用：

  正式引用键：`song2026survey`；书目字段统一维护于 [references.bib](../references.bib)。

- 与本项目的中文关联：该综述把 OPD 表述为学生采样轨迹上的 f-divergence 优化，并按优化对象、监督信号来源和稳定化方式梳理研究，同时讨论失败模式及其与 KL 约束强化学习的关系。可用于绪论、相关工作和实验变量设计；综述页面标注为持续更新，不能把它当作已正式发表的期刊或会议论文。
- 已访问 URL：
  - https://arxiv.org/abs/2604.00626

## 7. Qwen3 Technical Report

- 项目书原条目：`Qwen Team. Qwen3 Technical Report. 2025.`
- 核验题名：Qwen3 Technical Report。
- 作者署名：正式 BibTeX 可采用官方团队署名 `Qwen Team`。Qwen3-8B 官方模型卡的 Citation 区块明确给出该署名、arXiv 编号和年份；arXiv 页面同时列出 60 位个人作者，若学校格式要求完整个人作者，可据 arXiv 页面或 Qwen 官方仓库的 citation 区块展开。
- 出版类型：arXiv 技术报告/预印本，分类为 `cs.CL`。
- 日期与版本：arXiv v1 提交于 2025-05-14；页面未显示后续版本，年份为 2025。
- 正式引用（官方团队署名）：

  正式引用键：`qwen3technicalreport`；书目字段统一维护于 [references.bib](../references.bib)。

- 与本项目的中文关联：Qwen3 是项目书中可考虑的开源模型/部署对象，官方模型卡确认其训练阶段包含预训练与后训练，并提供 vLLM 部署信息。该报告主要用于模型和工程背景；当前核验未把它作为 OPD 算法结论的直接来源。
- 已访问 URL：
  - https://arxiv.org/abs/2505.09388
  - https://huggingface.co/Qwen/Qwen3-8B
  - https://github.com/QwenLM/Qwen3

## 8. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

- 项目书原条目：`Xu A, Lin B, Xue B, et al. Deepseek-v4: Towards highly efficient million-token context intelligence[J]. arXiv preprint arXiv:2606.19348, 2026.`
- 核验题名：DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence。
- 作者署名：建议采用官方团队署名 `DeepSeek-AI`。arXiv 页面将 `DeepSeek-AI` 列为作者首项，并列出个人作者列表；本记录不把项目书中的前三位个人作者误当成完整作者列表。
- 出版类型：arXiv 技术报告/预印本，分类为 `cs.CL`。
- 日期与版本：arXiv 页面显示 v1，提交日期为 2026-04-26，年份为 2026。需要保留一个日期记录：编号为 `2606.19348`，但页面显示提交日期为 2026-04-26；本记录只报告两项原文信息，不推断或修正该不一致。
- 正式引用（官方团队署名）：

  正式引用键：`deepseekai2026deepseekv4`；书目字段统一维护于 [references.bib](../references.bib)。

- 与本项目的中文关联：报告第 5.1.2 节明确描述多教师 OPD，将学生自身生成轨迹上的教师分布用于能力合并；报告采用反向 KL，并指出全词表 logits 蒸馏可降低 token 级 KL 估计的梯度方差、改善训练稳定性。第 5.2.2 节还直接讨论全词表 OPD 的教师调度。因此该报告与本项目的 KL 方向、监督粒度、稳定性和资源效率比较具有直接关联，但报告中的工程结论仍应在本项目配置下独立验证。
- 已访问 URL：
  - https://arxiv.org/abs/2606.19348
  - https://arxiv.org/html/2606.19348

## 9. MiMo-V2-Flash Technical Report

- 项目书原条目：`Xiao B, Xia B, Yang B, et al. Mimo-v2-flash technical report[J]. arXiv preprint arXiv:2601.02780, 2026.`
- 核验题名：MiMo-V2-Flash Technical Report。
- 作者署名：建议采用 arXiv 给出的官方团队署名 `Xiaomi LLM-Core Team`。arXiv 页面以该团队为作者组名，并列出完整个人作者列表；因此项目书中仅列 Xiao、Xia、Yang 不能视为完整作者列表。
- 出版类型：arXiv 技术报告/预印本，分类为 `cs.CL`。
- 日期与版本：v1 提交于 2026-01-06；当前页面为 v2，最后修订于 2026-01-08；年份为 2026。
- 正式引用（官方团队署名）：

  正式引用键：`xiaomillmcore2026mimov2flash`；书目字段统一维护于 [references.bib](../references.bib)。

- 与本项目的中文关联：报告第 4.1 节提出多教师在线策略蒸馏（MOPD），采用三阶段后训练流程；学生从自身演化分布采样，并从领域教师获得基于 KL 散度的 token 级监督。该报告可作为多教师 OPD 和后训练工程实现的相关工作与对照背景，不能直接替代本项目对不同 KL 方向和监督粒度的受控实验。
- 已访问 URL：
  - https://arxiv.org/abs/2601.02780
  - https://arxiv.org/html/2601.02780
  - https://github.com/XiaomiMiMo/MiMo

## 核验边界

- 四条记录均依据 arXiv 题名页或 HTML 正文核验；Qwen Team 的团体署名额外依据 Qwen/Qwen3-8B 官方模型卡的 citation 区块。
- 本文件未把 arXiv 预印本写成期刊或会议论文，也未为 DeepSeek-V4 的编号与提交日期不一致添加推测性解释。
- 本文件记录可追溯的来源与阅读边界；不代表已复现论文实验。
