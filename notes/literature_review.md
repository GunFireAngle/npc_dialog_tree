# 文献笔记（Literature Review）

> 诚信提示：以下为阅读线索与论点整理，**条目信息必须逐条到 DBLP / Google Scholar 核对后**才可写入 `paper/refs.bib`。
> 未核实的内容一律标注 `[待核实]`，不得凭记忆填写卷期页码。

## 1. 研究脉络

| 脉络 | 代表工作 | 与本课题的关系 |
|---|---|---|
| 思维链 / 树式推理 | Wei et al., Chain-of-Thought Prompting (NeurIPS 2022)；Yao et al., Tree of Thoughts (NeurIPS 2023) | "树"在 LLM 里通常是搜索结构；本课题把树用在**内容生成**上，节点是台词而非推理状态 |
| 生成式智能体与角色扮演 | Park et al., Generative Agents (UIST 2023)；Mirowski et al., Co-Writing Screenplays and Theatre Scripts with Language Models (CHI 2023) | 证明 LLM 能维持角色记忆与人格；但其评测多为定性案例，缺少"深度"这一自变量 |
| 游戏对话生成 | 关键词：`LLM game dialogue generation`、`NPC dialogue tree`、`procedural dialogue` `[待核实具体条目]` | 工业上常用"节点树 + 手工写死"，LLM 自动生成多轮分支树的系统化评测较少 |
| 长文本一致性衰减 | Liu et al., Lost in the Middle (TACL 2024)；关键词：`long-form generation coherence degradation` `[待核实]` | 提供"越长越容易跑偏"的理论预期，本课题把"长"操作为"树深度" |
| 结构化 / 受约束生成 | 关键词：`constrained decoding`、`grammar-guided generation`、`JSON schema constrained decoding` `[待核实]` | 我们的"逐层批量扩写 + 解析失败修复重试"是工程折中，可作为 baseline 对照 |
| LLM-as-a-judge | Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (NeurIPS 2023) | 代理评分的方法依据；本课题明确区分"代理评分"与"真人量表" |
| 参数高效微调 | Hu et al., LoRA (ICLR 2022)；Qwen2.5 Technical Report (arXiv 2412.15115) | 8GB 显存下唯一可行的微调路线 |

## 2. 三个量表的概念来源

- **一致性（consistency）**：叙事学中的"故事世界连贯性"＋游戏设计的"设定圣经（story bible）"约束。
  可操作化为：与 NPC 档案中的身份、物品、经历是否矛盾；分支之间是否自相矛盾。
- **角色性格稳定性（persona stability）**：角色扮演研究中的 "persona drift"。
  可操作化为：说话方式、价值观、禁忌是否始终一致；深层节点是否退化为通用助手口吻。
- **玩家选择有意义性（choice meaningfulness）**：游戏研究中的 "meaningful choice"（选择需有可感知的差异与后果）。
  可操作化为：同一节点两个选项是否代表不同立场/意图，而不是换词重述。

## 3. 研究缺口（本课题的落点）

1. 现有 LLM 叙事工作多评"故事好不好"，少评"**结构规模**如何影响质量"，缺少把深度作为自变量的受控实验。
2. 游戏对话的可玩性要求"每个节点都能被点击"，这比自由文本生成多了**结构合法性**约束，现有评测很少同时报告结构指标与内容指标。
3. 缺少"小模型 + 结构微调"在中文 NPC 对话上的可行性与代价分析（8GB 消费级显卡场景）。

## 4. 假设

- H1：随深度增加，三个量表分数下降，且**性格稳定性**下降最快（长程人格维持比事实一致更难）。
- H2：结构合法性（JSON 合规、节点完整）随深度显著下降，一次性整树生成比逐层扩写更早崩塌。
- H3：LoRA 结构微调主要提升结构合法性与选项意图差异，对一致性/性格稳定性的提升有限。

## 5. 待核实清单

- [ ] 上述每条 `[待核实]` 的准确作者、年份、会议/期刊、DOI
- [ ] 游戏对话生成方向近三年（2023-2026）是否有直接可比的"深度-质量"实验
- [ ] LLM-as-judge 与真人评分的一致性系数（本课题需自行报告 Spearman）
