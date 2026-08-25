# preexp01：LongTutor 官方基准接入检查

本记录仅用于确认官方 LongTutor 仓库的数据结构、字段和评价脚本，**不包含任何模型推理、不修改官方代码、不运行实验**。

## 1. 官方仓库版本

- 仓库：https://github.com/liano3/LongTutor
- 接入方式：git submodule，路径 `third_party/LongTutor/`
- Commit：`ba40af3b9e976f5960c42a15a645c0e9c3a7d718`
- Commit 时间：2026-07-23T11:56:27+08:00
- Commit 说明：`Revise README for dataset and script instructions`
- 分支：`main`

## 2. LongTutor-Gold 数据文件

针对 XES3G5M 数据集（论文中主用于人工评测的数据集）：

| 文件 | 路径 | 行数 | 作用 |
|---|---|---|---|
| 长期学生历史序列 | `third_party/LongTutor/data/XES3G5M/sequences_long.jsonl` | 3437 | 每个学生的原始交互序列（题号/作答/时间戳），序列长度为 200（最后一题为当前题，前 199 题为历史） |
| 题库 | `third_party/LongTutor/data/XES3G5M/questions.jsonl` | 7652 | 题目内容、答案、解析、题型、知识点（`concepts` 为 `----` 分隔的层级字符串） |
| 人工 Gold 标注（LongTutor-Gold） | `third_party/LongTutor/data/XES3G5M/human_an_updated.jsonl` | **1000** | 人工审校后的记忆问答 / 状态诊断 / 教学策略 / 教学内容标注 |
| 合成扩展评测集 | `third_party/LongTutor/data/XES3G5M/pipeline_an_scale.jsonl` | 2437 | 未经人工审校的自动生成测试用例，用于规模化评测 |

**⚠️ 重要缺失**：官方仓库中**没有**直接提供 `history_features_lastq.jsonl`（及 `history_features_lastq_scale.jsonl`）这两个"当前题+历史+统计特征"输入文件。它们需要本地运行 `scripts/compute_history_stats.py` 生成；该脚本 `__main__` 中硬编码的默认路径指向 **MOOCRadar**，若要生成 XES3G5M 版本必须手动修改脚本中的输入/输出路径（详见第 6 节）。

## 3. 两个数据文件如何对应

- `sequences_long.jsonl` 共 3437 条学生序列 = 1000（对应 Gold）+ 2437（对应 scale 合成集），行数精确吻合。
- 对应关系通过 `_key` 字段实现，生成逻辑见 `scripts/gpt_memory_diagnose.py::_sample_key`：

  ```python
  _key = f"{uid}||{sha256(question_info_string).hexdigest()}"
  ```

  即：学生 `uid` + 当前题目渲染文本的 SHA-256 哈希。`human_an_updated.jsonl` 与将来生成的 `history_features_lastq.jsonl` 通过该 `_key` 一一对应（`history_features_lastq.jsonl` 的前 1000 行对应 `human_an_updated.jsonl`，其余 2437 行对应 `pipeline_an_scale.jsonl`，这是 README 中的说明，未来生成后可用 `_key` 交叉核对确认）。
- 已验证：`human_an_updated.jsonl` 第一条样本 `_key = "8572||4a7d..."`，其 `uid=8572` 确实存在于 `sequences_long.jsonl` 中，说明对应机制成立。

## 4. 一条输入样本（history_features_lastq.jsonl，由 compute_history_stats.py 生成）的主要字段

由 `scripts/compute_history_stats.py::build_history_features` 输出，字段：

- `uid`：学生 ID
- `question_info`：当前题目渲染后的文本（题号/题型/内容/选项/知识点/时间/作答结果）
- `history_info`：历史所有题目的渲染文本列表
- `related_history`：与当前题知识点相关的历史题目渲染文本列表
- `features`（统计特征字典）：
  - `overall_acc`：历史总体正确率
  - `related_concept_acc`：相关知识点历史正确率
  - `related_concept_count`：相关知识点历史作答次数
  - `related_last_correct_interval_days`：距上次相关知识点正确作答的天数
  - `related_last_correct_question_info`：上次相关知识点正确作答的题目文本
  - `recent_acc`：最近 K 题（默认 K=10 或 20）滑动窗口正确率
  - `recent_count`：滑动窗口内作答数
  - `cur_question_error_rate`：当前题目在全体学生中的错误率（全局难度）

## 5. 一条 Gold 标注（human_an_updated.jsonl）的主要字段

```json
{
  "memory": [
    {"query": "...", "answer": "..."},   // 3 条记忆问答（对应 Information Extraction / Multi-session Reasoning / Hallucination Check）
    ...
  ],
  "diagnosis": "Conceptual Gap",          // 状态诊断标签（4 类之一，见第 7 节）
  "reason": "...",                        // 诊断依据（中文）
  "strategy": "Conceptual Explanation",   // 与诊断对应的教学策略
  "content": "...",                       // 教学干预内容（中文，人工审校后的 Gold 教学话语）
  "_key": "8572||4a7d...",                // 与输入样本对应的唯一键
  "_draft": { ... }                       // 人工审校前的自动生成草稿（辅助字段，非评测主字段）
}
```

## 6. 官方脚本路径

- 历史特征构建：`third_party/LongTutor/scripts/compute_history_stats.py`
  （`__main__` 默认路径为 MOOCRadar，XES3G5M 需手动改路径；两数据集的知识点切分函数不同，脚本顶部已给出并做了注释切换说明，与两数据集 `concepts` 字段实际格式一致：MOOCRadar 为扁平列表，XES3G5M 为 `----` 分层字符串）
- 合成测试用例生成：`third_party/LongTutor/scripts/gpt_memory_diagnose.py`
- **官方 AI Tutor 推理脚本**：`third_party/LongTutor/scripts/eval_ai_tutor.py`
- **官方 AI Tutor 评价脚本**：`third_party/LongTutor/scripts/compute_ai_tutor_eval_metrics.py`
- 辅助：`scripts/data_processing.py`（数据转换/过滤）、`scripts/openai_helper.py`（OpenAI 兼容调用封装）

## 7. Gold 状态诊断类别（4 类，`scripts/gpt_memory_diagnose.py::DIAGNOSES` / `DIAGNOSTIC_SCHEMA`）

| 诊断类别 | 含义 | 对应教学策略 |
|---|---|---|
| Recall Failure | 记忆层：曾掌握但遗忘（超过 4 天未接触相关概念） | Retrieval Practice |
| Conceptual Gap | 理解层：概念性误解/图式缺陷（新知识或基础薄弱） | Conceptual Explanation |
| Procedural Error | 应用层：理解概念但计算/步骤出错 | Stepwise Scaffolding |
| Transfer Deficit | 分析层：无法将已学知识迁移到新情境 | Analogical Transfer |

## 8. 官方教学质量评价指标（`scripts/compute_ai_tutor_eval_metrics.py`）

**记忆问答（Memory QA）**：
- 记忆问答准确率（Memory Accuracy），总体及按类型（`MEM_TYPES`）拆分：
  - 信息抽取（Information Extraction）
  - 多轮/跨会话推理（Multi-session Reasoning）
  - 幻觉检测（Hallucination Check）
  - 判定方式：字符串精确匹配，不匹配则调用 LLM 进行语义等价判断

**状态诊断（Diagnosis）**：
- 状态诊断准确率（Accuracy）
- 状态诊断宏平均 F1（Macro-F1，按 4 类诊断标签分别计算 P/R/F1 后取宏平均）

**教学内容质量（LLM 评分 1-5 分 + 文本重叠度）**：
- 长期历史利用度（History Utilization，代码字段 `history_score`）
- 教学策略一致性（Strategy Alignment，代码字段 `strategy_score`）
- 教学连贯性（Coherence，代码字段 `coherence_score`）
- 教学适切性（Appropriateness，代码字段 `comprehension_score`，评分说明为 "Comprehension & Difficulty"，对应生成 Prompt 中的第 4 维 "Appropriateness / ZPD"）
- 综合评分（Overall，代码字段 `overall_score`）
- 文本重叠度（ROUGE-L，`rouge_l`，教学内容与 Gold 内容的字面相似度参考指标）

## 9. 已确认的路径/文档不一致点

- `README.md` Quickstart 第 3 步直接给出 `--input data/XES3G5M/history_features_lastq.jsonl` 的运行示例，但该文件**并未随仓库提供**，必须先运行 `compute_history_stats.py` 并手动改路径生成（详见第 2、6 节）。这是文档隐含步骤，未在 Quickstart 中显式列出，容易被忽略。**此缺口已在第 10 节用我们自己的可复现脚本补齐。**
- 除此之外，`concepts` 字段的知识点切分函数说明与两数据集实际数据格式核对一致（MOOCRadar 扁平列表 / XES3G5M `----` 分层字符串），未发现代码与数据不符的情况。

## 10. 数据准备与一致性检查（本仓库脚本，非官方文件）

脚本：[experiments/preexp01/prepare_longtutor_gold.py](prepare_longtutor_gold.py)

**用途**：不修改 `third_party/LongTutor` 中的任何文件，直接 `import` 并复用官方函数
（`compute_history_stats.load_questions_map` / `compute_global_error_rates` /
`build_history_features`，以及 `gpt_memory_diagnose._sample_key`），为 XES3G5M
生成 `history_features_lastq.jsonl`，并对前 1000 条与 `human_an_updated.jsonl`
（LongTutor-Gold）的对应关系做自动一致性检查。

对官方代码唯一的"改动"方式：由于官方脚本当前激活的 `_concept_segments` 实现
（`return concepts`，不做切分）是面向 MOOCRadar 设计的（该数据集 `concepts`
已经是扁平字符串列表），而 XES3G5M 的 `concepts` 是 `----` 分隔的层级字符串，
必须使用官方脚本顶部注释掉的 "XES3G5M-specific" 切分逻辑。本脚本在 import 官方
模块**之后**，仅在本进程内对 `compute_history_stats._concept_segments` 做
monkeypatch（替换为与官方注释代码逐字一致的实现），磁盘上的官方文件未被写入或修改。

**运行命令**（本地与 Colab 均适用，一条命令复现；首次需按官方 README 安装依赖）：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r third_party/LongTutor/requirements.txt
python experiments/preexp01/prepare_longtutor_gold.py
```

**输入**：
- `third_party/LongTutor/data/XES3G5M/sequences_long.jsonl`
- `third_party/LongTutor/data/XES3G5M/questions.jsonl`
- `third_party/LongTutor/data/XES3G5M/human_an_updated.jsonl`（仅用于验证，不用于生成）

**输出**（均不进入官方 submodule，且大文件不进 git，见 `.gitignore`）：
- `artifacts/preexp01/history_features_lastq.jsonl`（生成的历史特征输入，约 240MB，**不提交**）
- `artifacts/preexp01/data_check.json`（一致性检查报告，**提交保留**）

**实际检查结果**（2026-08-25T04:40:21+00:00 生成，LongTutor commit `ba40af3b9e976f5960c42a15a645c0e9c3a7d718`）：

| 检查项 | 结果 |
|---|---|
| 生成总样本数（generated_count） | 3437 |
| Gold 标注样本数（gold_count） | 1000 |
| 匹配成功数（matched_gold_count） | **1000** |
| 匹配失败数（unmatched_gold_count） | **0** |
| 生成文件内重复 key 数（duplicate_key_count） | 0 |
| 前 1000 条 key 与 Gold key 集合是否完全相等 | true |
| 序列长度是否全部为 200 | true（distinct: `[200]`） |
| 历史长度是否全部为 199 | true（distinct: `[199]`） |
| 当前题是否确为序列最后一题（3437 条全查） | 全部一致，mismatched_count=0 |
| "按行位置对应" 与 "按官方 sample key 对应" 是否冲突 | **否（conflict_detected=false）**，两种方式结果完全一致 |
| recent_k 取值 | 20（沿用 `build_history_features` 函数默认值；官方仅为 MOOCRadar 显式指定 10，未对 XES3G5M 给出专门取值，故不做猜测性覆盖，已在 `data_check.json` 中记录） |

一条生成样本（`uid=8572`）的 `_sample_key` 为
`8572||4a7d36d788845c983528abe193c9fd6d7f629545b8ef6aa7e605ad12d6dde0db`，
与 `human_an_updated.jsonl` 第一条记录的 `_key` 完全一致，交叉验证通过。

完整字段级检查结果见 [artifacts/preexp01/data_check.json](../../artifacts/preexp01/data_check.json)（该文件已提交到仓库；同目录下的 `history_features_lastq.jsonl` 为可重复生成的大文件，未提交，需要时按上方命令本地重新生成）。

---

*本文档由脚本与数据的实际读取结果整理，未凭空猜测字段名。*
