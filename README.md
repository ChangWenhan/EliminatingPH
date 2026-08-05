# Eliminating Package Hallucination via Registry Verification and Substitution-Guided Repair

让代码模型写的 Python 代码**不再编造不存在的包**。核心思路:模型写完后,用"两次自报依赖 + PyPI 注册表对账"找出它编造的包名,再让模型在真实包资料的引导下重写代码。在四个 Python 数据集上,把包幻觉率从 11.9%~20.7% 压到 ~0%,同时语法有效率全面上升。

方法由两个组件构成,对应题目中的两个机制:

1. **Registry Verification(注册表验证)**:三通道提取包名(`pip install` 命令 + 两次向模型询问依赖)后,与 PyPI 注册表对账,判定真伪,并据此决定是否触发修复
2. **Substitution-Guided Repair(替代引导修复)**:对每个幻觉包检索真实替代包,生成"幻觉包 → 真实包"替换映射,引导模型重写代码并重新验证

修复触发条件:存在幻觉包 / 语法错误 / 出现 pip 安装命令 / 动态 import。基础修复变体(不提供替换映射,仅给真实包资料)用于消融对比。

## 背景

代码模型在回答编程问题时经常推荐**并不存在的包**(包幻觉,package hallucination)。例如让模型解决一个报错问题,它会建议 `pip install find-libpython`,而 PyPI 上根本没有这个包。攻击者可以利用这一点发布同名恶意包,诱导用户安装(包混淆攻击)。

- **评测口径**对齐 Spracklen et al.(USENIX Security 2025):三启发式提取包名(`pip install` 命令 + 两次向模型询问依赖)+ PyPI 全量包名对账,不在名单即幻觉。
- **方法为本仓库原创**:原文的缓解手段(RAG、Self-Refinement、Fine-tuning)都不涉及"注册表裁决 + 替代包检索 + 代码重写修复"的闭环;其中的**替代包映射机制(Substitution-Guided Repair)是本工作的核心贡献**。

## 方法

每个数据集(4 个)固定采样 500 条问题,每条在三个条件下各跑一遍。

### 条件

| 条件(代码标识) | 内容 |
|---|---|
| `no_rag` | 基线。裸问模型,无检索、无修复 |
| `vtrr` | 注册表验证 + 基础修复(检测 → 对账 → 检索真实包资料 → 重写 → 再验证) |
| `vtrr_sub` | 注册表验证 + **替代引导修复**:为每个幻觉包检索真实替代包,把替换映射喂给模型修复 |

### 单样本流程(以替代引导修复 `vtrr_sub` 为例)

```
问题 ──► [LLM 写代码] ──► 代码/回答文本
              │
              ├─► 问①"运行这段代码需要哪些包?" ──┐
              ├─► 问②"解决这个问题需要哪些包?" ──┼──► 查 PyPI 名单对账
              └─► 正则扫 pip install / import ──┘         │
                                                          ▼
                                               有幻觉?语法错?有 pip 命令?
                                                           │ 是
                                                           ▼
                                 BM25 检索真实替代包(仅 sub 变体)──► 替换映射
                                                          │
                                                          ▼
                   修复 prompt:原任务 + 代码 + 幻觉名单 + 真实包资料 (+ 替换映射)
                                                          │
                                                          ▼
                                    [LLM 重写代码] ──► 重新对账验证
```

关键设计:

1. **模型只管写,代码只管判**:所有真伪裁决来自 `pypi_package_names.csv`(PyPI 全量名单),模型没有自我判断权(区别于原文的 Self-Refinement)。
2. **检索不是生成前 RAG**:检索只发生在检测到幻觉之后的修复阶段;初始生成与基线完全一致,不依赖 RAG 增强。
3. **幻觉率只统计三通道**:问①幻觉 + 问②幻觉 + `pip install` 幻觉。代码里解析出的 import 仅用于触发修复和合法包兜底,不进入幻觉率分子分母。
4. **名称归一化**:比对前将 `_`/`.`/`-` 统一为 `-` 并小写;标准库与人工维护的 false-positive 名单直接跳过(既不算 valid 也不算幻觉)。

## 结果

Qwen2.5-Coder-7B-Instruct,每条件 n=500(数据见 `results/vtrr_full_*`)。

| 数据集 | 条件 | 幻觉率 | 幻觉包/总包 | 语法有效率 | 修复触发 |
|---|---|---|---|---|---|
| LLM_AT | no_rag | 11.86% | 133/1121 | 97.2% | — |
| LLM_AT | vtrr | **0.00%** | 0/1152 | 98.4% | 52 |
| LLM_AT | vtrr_sub | **0.00%** | 0/1157 | 97.8% | 114 |
| LLM_LY | no_rag | 15.16% | 227/1497 | 95.4% | — |
| LLM_LY | vtrr | **0.00%** | 0/1423 | 97.6% | 97 |
| LLM_LY | vtrr_sub | **0.00%** | 0/1485 | 96.0% | 172 |
| SO_AT | no_rag | 20.71% | 99/478 | 75.8% | — |
| SO_AT | vtrr | **0.00%** | 0/452 | 94.6% | 132 |
| SO_AT | vtrr_sub | **0.00%** | 0/536 | 95.0% | 174 |
| SO_LY | no_rag | 20.71% | 93/449 | 53.4% | — |
| SO_LY | vtrr | **0.00%** | 0/567 | 86.4% | 254 |
| SO_LY | vtrr_sub | **0.00%** | 0/597 | 87.0% | 261 |

观察:

- **包幻觉完全消除**:替代引导修复(`vtrr_sub`)在四个数据集上幻觉率全部为 0。
- **语法有效率同步提升**:修复机制以"语法错误"为触发条件之一,修复过程顺带重写了不合法代码;SO 类数据集提升尤其明显(53.4% → 87.0%)。
- **残余语法错误归因于基座模型**:残留失败均为模型生成问题(答错语言、截断、普通语法错误),与包名无关——包幻觉被消除后,代码质量受限于代码模型本身。

## 仓库结构

```
vtrr_alignment/          核心实现
  run_four_conditions.py 实验入口(6 个条件,含 rag_no_attack / spracks_attack)
  metrics.py             三通道提取、注册表对账、代码依赖检查、幻觉率
  retrieval.py           自实现 BM25 检索器、替代包检索
  llm.py                 OpenAI 兼容 API 客户端
Data/Python/             4 个数据集 + pypi_package_names.csv + false_positive_packages.csv
Mitigation/Data/         检索语料(RAG_data.jsonl,49920 条包-问题描述)
results/vtrr_full_*/     实验结果(summary.csv / rows.csv)
tests/                   指标单元测试
```

## 复现

1. 用 vLLM(或其他 OpenAI 兼容服务)部署模型,如 `Qwen/Qwen2.5-Coder-7B-Instruct`。
2. 安装依赖:`pip install -r requirements.txt`。
3. 跑单个数据集、单个条件:

```bash
python -m vtrr_alignment.run_four_conditions \
  --input Data/Python/SO_LY.json \
  --conditions no_rag vtrr vtrr_sub \
  --output-dir results/vtrr_full_SO_LY \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen2.5-Coder-7B-Instruct \
  --limit 500 --seed 0
```

4. 输出:`<output-dir>/<condition>/rows.csv`(逐条结果)+ `summary.csv`(聚合:幻觉率、语法有效率、修复次数等)。

常用参数:温度(代码 0.7 / 包名 0.01)、top-k 检索 5、max-prompt-chars 6000、`--workers` 并行数。`--dry-run` 可无模型冒烟测试。

## 数据集

来自 Spracklen et al. 的 Python 子集,按 2×2 划分:

| 数据集 | 条数 | 内容 |
|---|---|---|
| LLM_AT / LLM_LY | 4922 / 4892 | LLM 生成的"写 Python 代码"指令;LY 为 2023 年后生成(涉及较新包) |
| SO_AT / SO_LY | 4640 / 4630 | StackOverflow 问答帖(Title+Body);LY 为 2023 年帖子 |

注意:SO 数据集不全是"写代码"任务,还有概念解释、报错排查、环境配置等;模型可能答成 bash/其他语言,评测统一按"代码 + import + pip 命令"三通道对账,纯解释回答不构成幻觉。

## 已知边界

- 语法有效性只反映"能解析",不评估运行正确性(原文用 HumanEval pass@1,本仓库未含)。
- 注册表快照口径:幻觉率以 `pypi_package_names.csv` 为事实表,若名单被污染,结果为幻觉率下界(与原文一致)。
