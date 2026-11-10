# NPC 对话树生成器：大模型自动生成多轮对话树及"深度 vs 质量"权衡

课程：软件 B（2026 秋）· 选题：大模型热门研究问题第 7 题
研究问题：给本地模型一个游戏世界观设定，生成分支数为 2、深度为 2/4/6 的 NPC 对话树，
用一致性、角色性格稳定性、玩家选择有意义性三个量表评测，分析"深度 vs 质量"的权衡。

## 1. 环境

- Windows + Python 3.12，NVIDIA RTX 4060 Laptop（8 GB 显存）
- 基座模型：`Qwen2.5-1.5B-Instruct`（本地推理 + LoRA 结构微调）
- 依赖：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate fastapi "uvicorn[standard]" pandas scipy matplotlib
```

## 2. 目录

```
npc_research/
├── paper/                 # 论文（LaTeX）、figures/、refs.bib
├── experiment/
│   ├── run_exp.py         # 主实验：生成 → 自动指标 → LLM 评审 → 导出 CSV
│   ├── analyze.py         # 统计检验 + 300dpi 论文图
│   ├── src/               # schema / worlds / prompts / engine / builder / metrics / judge
│   ├── data/trees/        # 每棵树一个 JSON（含树、指标、评审分，可断点续跑）
│   ├── results/           # results.csv、trees.csv、stats.csv、summary.md
│   └── plots/             # fig1~fig4（pdf + png）
├── app/
│   ├── backend/server.py  # FastAPI：生成 / 评分 / 导出
│   ├── frontend/index.html# 试玩 + 对话树可视化 + 三量表盲评
│   └── data/ratings.csv   # 人工盲评结果
├── training/
│   ├── build_dataset.py   # 程序化合成"结构对齐"语料
│   ├── train_lora.py      # LoRA 微调
│   ├── data/              # train.jsonl / val.jsonl
│   └── output/lora/       # 适配器权重
├── models/Qwen2.5-1.5B-Instruct/
├── notes/                 # 文献笔记、提示词留档
└── logs/                  # experiment.log、train_log.csv、安装日志
```

## 3. 复现步骤

```powershell
# 0) 项目根目录
cd npc_research

# 1) 下载基座模型（任选镜像）
pip install -U "huggingface_hub[cli]"
$env:HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

# 2) 冒烟测试（不占显存，用 mock 引擎验证全链路）
python experiment/run_exp.py --mock --models base --depths 2 --seeds 42 --limit 1 --judge-repeats 1

# 3) 构造训练数据并微调
python training/build_dataset.py --n 1600
python training/train_lora.py --epochs 1 --lr 2e-4    # 190 步，loss 0.260→0.025，约 63 分钟

# 4) 主实验：4 世界观 × 深度 2/4/6 × 2 seed × {base, ft}，含 3 次评审校准（评审仅作对照）
python experiment/run_exp.py --models base,ft --depths 2,4,6 --seeds 42,1337 --judge-repeats 3

# 4b) 只重算指标 / 只重导 CSV（不重新生成，不重跑模型）
python experiment/run_exp.py --recompute-metrics
python experiment/run_exp.py --judge-mode holistic --judge-limit 24   # 其它评审口径（可选）

# 5) 消融：一次性生成整棵树（深度 6，前两个世界观，seed 42）
python experiment/run_exp.py --ablation --models base

# 6) 分析与出图
python experiment/analyze.py

# 6.5) 干预对照：A1 节拍表 + A2 去重重采样 + A4 秘密门控（写入 trees_v2/，不覆盖主实验）
python experiment/run_exp.py --models base,ft --depths 2,4,6 --seeds 42 --skip-judge --tag v2
python experiment/analyze.py --compare v2   # 产出 compare_v2.csv 与 fig5_improvement_v2

# 7) 启动演示系统
python -m uvicorn app.backend.server:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

显存注意：8 GB 卡上训练必须用 `logits_to_keep=k`（见 `training/train_lora.py`）只对答案位置
计算 logits，否则 max-len 1408 会把显存吃满；生成/训练前先关闭占用 GPU 的程序
（实测游戏会抢占约 2 GB，直接导致 OOM）。

## 4. 关键设计

- **深度定义**：深度 d 表示一条对话路径上 NPC 说话的轮数；分支固定 2；完整树节点数 = 2^d − 1
  （d=2 → 3 节点，d=4 → 15，d=6 → 63）。
- **两种生成策略**：`levelwise`（逐节点扩写，主实验）与 `oneshot`（一次生成整棵树，消融）。
  逐节点策略让每次调用只携带该节点自己的历史路径，避免多节点同批生成时的"串味"。
- **对话策略规划层（tactic planning）**：扩写前先由确定性规划器根据"层级 + 玩家选项意图"
  决定本句的对话行为（戒备 / 试探 / 回避 / 松动 / 揭示 / 交易 / 强硬 / 收尾），
  再把该策略作为约束交给模型实现为台词。这样 NPC 在不同分支上不会说同一种话，
  也让"深层是否仍在推进剧情"可被自动指标检验。
- **深层规划干预（v2）**：针对"越深越空"的三个确定性补丁，均不改模型权重：
  - **A1 节拍表**：`prompts.plan_beats()` 先确定整棵树每层的内容目标（该层要推进哪条 fact、
    第几层解锁秘密），再把**当前层的目标**写进该节点的提示行（`…｜本层目标：…`）。
    实测把整张六行节拍表塞进提示词会压垮 1.5B（输出退化为短句），因此只给一行；
  - **A2 去重重采样**：台词/选项与已用内容相似度超阈值（均 0.90，只抓近乎逐字重复）、
    或把玩家上一句原样当成 NPC 台词时，带黑名单重采样，每层预算为节点数的 60%；
  - **A4 秘密门控**：秘密只在最后一层解锁，叶节点强制"揭示"策略；若某个叶节点一句都没沾到
    秘密，用 `force_reveal` 重写一次（整树预算 8 次），避免所有叶子收敛到同一段往事。
  实验通过 `--tag v2` 写入独立目录，`analyze.py --compare v2` 输出 fig5 与对照表。
- **串味修复**：玩家最后一句从 history 中单独拎出并以"玩家刚说：… → 现在轮到 NPC"渲染，
  显著降低 1.5B 模型整句复述玩家台词（角色错位）的比例。
- **训练**：LoRA（r=16）在程序化合成的 1600 条"结构对齐"语料上微调，学习 JSON 契约、
  双选项意图差异、角色口吻与长度约束。语料非人写，论文中如实披露为 silver data。
- **自动指标**（CPU，确定性）：结构完整度、重复句率、distinct-2、选项语义距离、
  选项意图区分率、语域命中率、助手口吻泄漏率、设定冲突启发式计数、句长统计。
- **确定性量表执行器（论文主指标）**：把三个 1–5 量表落成 9 项子检查，
  `score = 1 + 4 × 通过率`，全部由 CPU 代码在已存 JSON 上计算，可逐行复现。
  - 一致性 = 无事实冲突 + 无跨世界观专名 + 无世界外术语；
  - 角色性格稳定性 = 逐字重复率 < 0.10 + 语域命中率 ≥ 0.30 + 无助手口吻；
  - 玩家选择有意义性 = 选项意图区分率 ≥ 0.50 + 选项重复率 < 0.05 + 选项意图可识别率 ≥ 0.80。
- **LLM-as-judge 仅作校准对照**：同一量表交给 1.5B 模型、盲评、每棵 3 次重复，
  结果 24 棵全部给 5.0/5.0/5.0（方差 0），与确定性量表分（2.78~5.00）的相关系数无定义。
  即 1.5B 不具备执行量表的能力，论文中作为负面结果报告，主指标改用执行器。
  真人盲评页面已就绪，用于后续人工验证。
- **消融（oneshot）**：用同样提示一次性生成 d=6 的 63 节点树，实际只产出 1 / 6 个节点，
  完整度 0.016 / 0.095，结构合法 0/2（耗时 37s / 23s，token 预算 7168 并未成为瓶颈）。
  这说明结构约束必须由 harness 承担，而不是指望模型一次写对。

## 5. 主要结果

完整数字见 `experiment/results/{trees.csv,summary.md,stats.csv,compare_v2.csv}` 与 `experiment/plots/`。

- **主实验 48 棵树**（4 世界观 × 深度 2/4/6 × 2 seed × 2 系统）：结构合法 48/48；
  一致性量表饱和在 5.00 不随深度变化；选择有意义性 3.17→1.33、角色稳定性 3.67→2.67（d2→d6）。
  oneshot 消融在 d6 只产出 1/63 与 6/63 个节点，结构合法 0/2。
- **干预对照 v2（24 对，seed 42）**：综合分 3.70→3.91（p=.048）、角色稳定性 3.33→3.78（p=.007）、
  人物接地率 .239→.477（p<.001）、重复句率 .365→.312（p=.069）、收益随深度递增（+0.11/+0.22/+0.28）。
  但 d6 每棵树触发 34.1 次去重重采样只修好 4.9 次，叶子真正兑现秘密仅 10–21%。
- **三条提示词/去重敏感性的负面结果**（`paper/main.tex` §Interventions 有记录）：
  把整张六行节拍表注入提示词，同一棵树综合分 4.11→2.78；把规则从 8 条剪到 5 条，
  平均句长 27→19.6、综合分 3.67→3.22；按句首三字判"同模板"直接把整句塞进黑名单，
  ash d4 综合分 2.78（同配置 v2 为 3.67）。1.5B 对提示词长度、约束数量与黑名单内容都很敏感，
  最终保留"每节点一行目标 + 完整 8 条规则 + 只拦近乎逐字重复"的配置（即 v2，
  `experiment/results/trees_v3.csv` 保留该次探针的两棵树以留证）。

## 6. 学术规范

- 提示词全部留档于 `notes/prompts/prompt_archive.md`；
- 所有数字可由 `experiment/results/trees.csv` 逐行回溯到 `experiment/data/trees/*.json`；
- 参考文献条目须逐条在 DBLP / Google Scholar 核对后才可写入 `paper/refs.bib`；
- 真人评测请在 `app/frontend` 的盲评页面完成，导出 CSV 后放入 `app/data/ratings.csv`。
