#!/usr/bin/env python3
"""为 gold_state_teach_greedy 条件下的预测结果构建"教学内容人工审计"素材。

背景：diagnosis_follow_gold / strategy_follow_gold 只能说明模型"选对了标签"，
不能说明模型的最终 teaching content 是否真的执行了对应教学策略、是否利用了
学生的长期历史。本脚本不做任何自动打分/判断，只把每条样本人工判断所需的信息
（当前题、最近历史、相关历史、模型输出、Gold 参考）整理成结构化 JSON 和便于
阅读的 Markdown，供人工审阅。

重要边界：
- 不修改 predictions.jsonl（只读）、不修改 third_party/LongTutor、不运行任何
  模型/LLM/API、不做自动判分（不写"通过/失败/分数/模型没有遵循"这类结论）。
- Gold reason / Gold teaching content 只作为离线人工参考写进本脚本的输出文件
  （字段名明确为 gold_reason_reference / gold_teaching_reference），绝不会被
  这个脚本或任何推理脚本重新塞回模型 Prompt——本脚本完全不构造、不触碰任何
  Prompt，只做只读的数据整理。
- recent_history / related_history 直接取自已有数据（history_features_lastq.jsonl
  里的 history_info / related_history 字段），不重新定义任何检索/相关性算法；
  取"最近 N 条"用的 N 是预测文件里该样本自己记录的 used_history_records，
  与生成时实际喂给模型的历史条数保持一致，不做假设。

用法（--predictions / --output 均可自定义，不硬编码到某一次实验目录）：
    python experiments/preexp01/build_teaching_content_audit.py \
        --predictions /path/to/gold_state_teach_greedy_hist20_40/predictions.jsonl \
        --output artifacts/preexp01/gold_state_teach_greedy_hist20_40_audit
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LONGTUTOR_DATA = REPO_ROOT / "third_party" / "LongTutor" / "data" / "XES3G5M"
GOLD_PATH = LONGTUTOR_DATA / "human_an_updated.jsonl"
FEATURES_PATH = REPO_ROOT / "artifacts" / "preexp01" / "history_features_lastq.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "preexp01" / "teaching_content_audit"

# 官方 4 类诊断标签固定顺序（与 gpt_memory_diagnose.DIAGNOSES 一致），用于
# quick audit 按类别分组统计，不代表本脚本重新定义了分类逻辑。
DIAGNOSIS_ORDER = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")

# 只是人工审计时"重点看什么"的提示语，不是自动评价结果，不产出任何判断性结论。
AUDIT_FOCUS_BY_STRATEGY = {
    "Retrieval Practice": (
        "重点检查最终 content 是否通过回忆、提取、提示学生主动回忆已学知识，"
        "而非直接重新完整讲解答案。"
    ),
    "Conceptual Explanation": (
        "重点检查是否针对概念误解进行解释和澄清，而非只给步骤或答案。"
    ),
    "Stepwise Scaffolding": (
        "重点检查是否提供逐步支架、分解步骤或引导，而非一次性给完整解法。"
    ),
    "Analogical Transfer": (
        "重点检查是否把当前题与学生过去已经掌握的类似题/知识明确联系，帮助迁移，而非泛化讲解。"
    ),
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gold_map(gold_path: Path) -> dict:
    """按官方 sample_key（_key 字段）索引 Gold 记录，仅用于离线取
    reason/content 作为人工参考，不用于任何推理。"""
    m = {}
    for obj in _iter_jsonl(gold_path):
        key = obj.get("_key")
        if key:
            m[key] = obj
    return m


def load_features_by_index(features_path: Path, needed_indices: set) -> dict:
    """只读取 predictions 里实际出现过的那些 sample_index 对应的行，不把整个
    240MB 文件都载入内存。"""
    rows = {}
    if not needed_indices:
        return rows
    for i, obj in enumerate(_iter_jsonl(features_path)):
        if i in needed_indices:
            rows[i] = obj
        if len(rows) == len(needed_indices):
            break
    return rows


def build_case(row: dict, feat: dict, gold_obj: dict) -> dict:
    idx = row.get("sample_index")

    full_history = (feat.get("history_info") if feat else None) or []
    used_n = row.get("used_history_records")
    if isinstance(used_n, int) and used_n > 0 and used_n <= len(full_history):
        recent_history = full_history[-used_n:]
    else:
        # used_history_records 缺失或与实际历史条数对不上时，不猜测，原样给出
        # 完整历史，并在字段里如实标注 used_history_records 的真实值（可能是
        # None），人工可以自行判断。
        recent_history = full_history

    related_history = feat.get("related_history") if feat else None
    gold_strategy = row.get("gold_strategy")

    return {
        "sample_index": idx,
        "uid": row.get("uid"),
        "sample_key": row.get("sample_key"),
        "gold_diagnosis": row.get("gold_diagnosis"),
        "gold_strategy": gold_strategy,
        "model_diagnosis": row.get("diagnosis"),
        "model_strategy": row.get("strategy"),
        "model_content": row.get("content"),
        "current_question": feat.get("question_info") if feat else None,
        "recent_history": recent_history,
        "recent_history_count": len(recent_history),
        "used_history_records_recorded": used_n,
        "related_history": related_history,
        "related_history_count": len(related_history) if isinstance(related_history, list) else None,
        "gold_reason_reference": gold_obj.get("reason") if gold_obj else None,
        "gold_teaching_reference": gold_obj.get("content") if gold_obj else None,
        "audit_focus": AUDIT_FOCUS_BY_STRATEGY.get(
            gold_strategy,
            f"（未识别的策略 {gold_strategy!r}，未预置审计提示，请人工自行判断）",
        ),
    }


def render_case_markdown(case: dict, case_number: int) -> str:
    def _fmt_history_list(items, empty_note):
        if not items:
            return empty_note
        parts = []
        for i, h in enumerate(items, 1):
            parts.append(f"**#{i}**\n\n{h}")
        return "\n\n".join(parts)

    recent_block = _fmt_history_list(case["recent_history"], "（recent_history 为空）")
    if case["related_history"] is None:
        related_block = "（本条样本没有 related_history 字段，可能是数据源里没有提供）"
    else:
        related_block = _fmt_history_list(case["related_history"], "（related_history 为空列表）")

    lines = [
        f"## Case {case_number}",
        f"Sample index: {case['sample_index']}",
        f"UID: {case['uid']}",
        "",
        f"Gold Diagnosis: {case['gold_diagnosis']}",
        f"Gold Strategy: {case['gold_strategy']}",
        "",
        "### 当前题",
        case["current_question"] or "（current_question 为空）",
        "",
        f"### 最近历史（recent_history，共 {case['recent_history_count']} 条，"
        f"used_history_records={case['used_history_records_recorded']}）",
        recent_block,
        "",
        f"### 相关历史（若已有，related_history，共 "
        f"{case['related_history_count'] if case['related_history_count'] is not None else 'N/A'} 条）",
        related_block,
        "",
        "### 模型教学回复",
        f"Model Diagnosis: {case['model_diagnosis']}",
        f"Model Strategy: {case['model_strategy']}",
        "",
        case["model_content"] or "（model_content 为空）",
        "",
        "### Gold参考教学",
        f"Gold Reason: {case['gold_reason_reference'] or '（无）'}",
        "",
        f"Gold Teaching Content: {case['gold_teaching_reference'] or '（无）'}",
        "",
        "### 本案例审计重点",
        case["audit_focus"],
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="为 gold_state_teach_greedy 的预测结果生成教学内容人工审计素材（只整理数据，不打分）。"
    )
    ap.add_argument("--predictions", type=str, required=True, help="predictions.jsonl 路径")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    ap.add_argument("--gold", type=str, default=str(GOLD_PATH), help="LongTutor-Gold 标注文件路径")
    ap.add_argument("--features", type=str, default=str(FEATURES_PATH),
                     help="history_features_lastq.jsonl 路径")
    ap.add_argument("--quick-audit-per-class", type=int, default=2,
                     help="quick audit 每个 diagnosis 类别取几条，默认 2（4 类合计 8 条）")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    predictions_path = Path(args.predictions)
    gold_path = Path(args.gold)
    features_path = Path(args.features)
    output_dir = Path(args.output)

    if not predictions_path.exists():
        raise SystemExit(f"未找到 predictions 文件: {predictions_path}")
    if not gold_path.exists():
        raise SystemExit(f"未找到 Gold 文件: {gold_path}")
    if not features_path.exists():
        raise SystemExit(
            f"未找到 {features_path}。请先运行 "
            "python experiments/preexp01/prepare_longtutor_gold.py 生成该文件"
        )

    rows = list(_iter_jsonl(predictions_path))
    if not rows:
        raise SystemExit(f"{predictions_path} 是空文件")

    gold_map = load_gold_map(gold_path)
    needed_indices = {r.get("sample_index") for r in rows if r.get("sample_index") is not None}
    features_by_index = load_features_by_index(features_path, needed_indices)

    cases = []
    seen_indices = set()
    duplicate_indices = []
    empty_content_indices = []
    missing_feature_indices = []
    missing_gold_indices = []

    for row in rows:
        idx = row.get("sample_index")
        if idx in seen_indices:
            duplicate_indices.append(idx)
        seen_indices.add(idx)

        feat = features_by_index.get(idx)
        if feat is None:
            missing_feature_indices.append(idx)

        gold_obj = gold_map.get(row.get("sample_key"))
        if gold_obj is None:
            missing_gold_indices.append(idx)

        case = build_case(row, feat or {}, gold_obj or {})
        if not case["model_content"]:
            empty_content_indices.append(idx)
        cases.append(case)

    class_counts = {c: 0 for c in DIAGNOSIS_ORDER}
    for case in cases:
        d = case["gold_diagnosis"]
        if d in class_counts:
            class_counts[d] += 1

    # ---- quick audit：严格按 predictions 文件出现顺序，每类取前 quick_audit_per_class 条 ----
    per_class_taken = {c: 0 for c in DIAGNOSIS_ORDER}
    quick_cases = []
    for case in cases:
        d = case["gold_diagnosis"]
        if d in per_class_taken and per_class_taken[d] < args.quick_audit_per_class:
            quick_cases.append(case)
            per_class_taken[d] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) teaching_content_audit.json ----
    audit_json = {
        "source_predictions_path": str(predictions_path),
        "source_gold_path": str(gold_path),
        "source_features_path": str(features_path),
        "total_count": len(cases),
        "class_counts": class_counts,
        "duplicate_sample_indices": duplicate_indices,
        "empty_model_content_sample_indices": empty_content_indices,
        "missing_feature_sample_indices": missing_feature_indices,
        "missing_gold_sample_indices": missing_gold_indices,
        "cases": cases,
    }
    json_path = output_dir / "teaching_content_audit.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(audit_json, f, ensure_ascii=False, indent=2)

    # ---- 2) teaching_content_audit.md（全部 40 条）----
    md_lines = [
        "# 教学内容人工审计素材（全部样本）",
        "",
        f"来源 predictions：`{predictions_path}`",
        "",
        f"总数：{len(cases)}；四类分布：{class_counts}",
        "",
        "本文件只整理数据，不包含任何自动判分或结论性判断"
        "（不写\"通过/失败/分数/模型没有遵循\"），请人工逐条阅读后自行判断。",
        "",
        "Gold Reason / Gold Teaching Content 仅作为离线人工参考，"
        "从未进入、也不会进入任何模型推理 Prompt。",
        "",
        "---",
        "",
    ]
    for i, case in enumerate(cases, 1):
        md_lines.append(render_case_markdown(case, i))
    md_path = output_dir / "teaching_content_audit.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ---- 3) teaching_content_quick_audit_8.md ----
    quick_lines = [
        "# 教学内容 Quick Audit（8 条，按类别均衡）",
        "",
        "**选择规则（可复现）**：从 `teaching_content_audit.json` 的全部样本中，"
        f"严格按 predictions 文件里出现的原始顺序（即 manifest 固定顺序），"
        f"每个 diagnosis 类别依次取前 {args.quick_audit_per_class} 条——不做任何"
        "随机抽样，不重新排序。四类顺序固定为："
        + " / ".join(DIAGNOSIS_ORDER) + "。",
        "",
        f"来源 predictions：`{predictions_path}`",
        "",
        "本文件同样不包含任何自动判分或结论性判断，仅供优先人工审阅。",
        "",
        "---",
        "",
    ]
    for i, case in enumerate(quick_cases, 1):
        quick_lines.append(render_case_markdown(case, i))
    quick_path = output_dir / "teaching_content_quick_audit_8.md"
    with quick_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(quick_lines))

    # ---- 检查结果打印 ----
    print(f"total_count: {len(cases)}")
    print(f"class_counts: {class_counts}")
    print(f"quick_audit_total: {len(quick_cases)}")
    print(f"quick_audit_class_counts: {per_class_taken}")
    print(f"duplicate_sample_indices: {duplicate_indices}")
    print(f"empty_model_content_sample_indices: {empty_content_indices}")
    print(f"missing_feature_sample_indices: {missing_feature_indices}")
    print(f"missing_gold_sample_indices: {missing_gold_indices}")
    print(f"written: {json_path}")
    print(f"written: {md_path}")
    print(f"written: {quick_path}")

    ok = (
        not duplicate_indices
        and not empty_content_indices
        and not missing_feature_indices
        and not missing_gold_indices
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
