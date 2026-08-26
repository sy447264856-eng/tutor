#!/usr/bin/env python3
"""对已经保存的 predictions.jsonl 做离线重新解析，不重新调用模型。

用途：Qwen baseline 的一次真实推理跑完之后，如果发现有样本因为
"JSON 字符串里的 LaTeX 反斜杠转义问题"（例如 `\\times`、`\\(`、`\\)`）导致
`parse_success=False`，可以用这个脚本直接对已保存文件里的 `raw_output`
重新执行"官方 extract_json/validate -> fallback LaTeX 转义修复 -> 官方
validate"，而不需要重新跑一遍 Qwen。

复用逻辑来源：直接从 run_qwen_baseline.py 导入
`parse_model_json_output`（两步解析：官方优先，仅官方失败时才 fallback
repair）、`_align_memory`（官方 memory 对齐逻辑的复刻）；以及从官方
eval_ai_tutor.py 导入的 `_validate_output` / `_extract_mem_queries_from_test_obj`
/ `_load_tests_map`。不重复实现任何解析/校验逻辑，保证离线重跑和在线推理
用的是完全同一套代码。

用法：
    python experiments/preexp01/repair_predictions.py \
        --input artifacts/preexp01/qwen_baseline_smoke/predictions.jsonl \
        --output artifacts/preexp01/qwen_baseline_smoke/repaired_predictions.jsonl

绝不会原地覆盖 --input（会在写入前检查两个路径是否指向同一个文件，是则拒绝
执行）。同时会在 --output 同目录下写一个 repair_summary.json。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 复用同一份解析/校验逻辑，而不是在这里重新实现一遍
from run_qwen_baseline import (  # noqa: E402
    GOLD_PATH,
    official_extract_mem_queries,
    official_load_tests_map,
    official_validate_output,
    parse_model_json_output,
    _align_memory,
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def repair_one_row(row: dict, tests_map: dict) -> dict:
    """对一条 predictions.jsonl 记录重新执行"官方解析 -> fallback repair ->
    validate"。raw_output 原样透传，绝不修改；只更新解析衍生出的字段。
    """
    new_row = dict(row)
    raw_output = row.get("raw_output")

    parse_success = False
    parsed_output = None
    memory = diagnosis = reason = strategy = content = None
    parse_repair_applied = False
    parse_repair_type = None
    parse_error = row.get("parse_error")

    if raw_output:
        sample_key = row.get("sample_key")
        test_obj = tests_map.get(sample_key)

        parsed, parse_error, parse_repair_applied, parse_repair_type = parse_model_json_output(raw_output)

        if parsed is not None and test_obj is not None:
            mem_queries = official_extract_mem_queries(test_obj)
            err = official_validate_output(parsed, expected_queries=mem_queries)
            if err is None:
                parsed["memory"] = _align_memory(parsed, mem_queries)
                parse_success = True
                parsed_output = parsed
                memory = parsed.get("memory")
                diagnosis = parsed.get("diagnosis")
                reason = parsed.get("reason")
                strategy = parsed.get("strategy")
                content = parsed.get("content")
            else:
                parse_error = err
        elif parsed is not None and test_obj is None:
            # 找不到对应的 Gold 测试用例（sample_key 不匹配），无法用官方
            # expected_queries 做完整校验，明确标注原因，不假装校验通过。
            parse_error = f"missing_gold_test_obj_for_sample_key:{sample_key}"

    new_row.update({
        "parsed_output": parsed_output,
        "memory": memory,
        "diagnosis": diagnosis,
        "reason": reason,
        "strategy": strategy,
        "content": content,
        "parse_success": parse_success,
        "parse_error": parse_error,
        "parse_repair_applied": parse_repair_applied,
        "parse_repair_type": parse_repair_type,
        # raw_output 显式保持原样，不从任何修复结果覆盖
        "raw_output": raw_output,
    })
    return new_row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "对已保存的 predictions.jsonl 做离线重新解析（官方解析 -> LaTeX 转义 "
            "fallback repair -> 官方 validate），不重新调用模型。"
        )
    )
    ap.add_argument("--input", type=str, required=True, help="原始 predictions.jsonl 路径")
    ap.add_argument("--output", type=str, required=True,
                     help="修复后结果的输出路径（例如 repaired_predictions.jsonl），"
                          "绝不能与 --input 相同")
    ap.add_argument("--summary", type=str, default=None,
                     help="repair_summary.json 的输出路径；默认写到 --output 同目录下的 repair_summary.json")
    ap.add_argument("--gold", type=str, default=str(GOLD_PATH),
                     help="LongTutor-Gold 标注文件路径，用于按 sample_key 重新取回 expected_queries 做校验")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.parent / "repair_summary.json"
    gold_path = Path(args.gold)

    if not input_path.exists():
        print(f"ERROR: 输入文件不存在: {input_path}", file=sys.stderr)
        return 2

    if input_path.resolve() == output_path.resolve():
        print("ERROR: --output 不能与 --input 相同，拒绝原地覆盖原始 predictions.jsonl。", file=sys.stderr)
        return 2

    if not gold_path.exists():
        print(f"ERROR: Gold 文件不存在: {gold_path}", file=sys.stderr)
        return 2

    tests_map = official_load_tests_map(gold_path)

    total_count = 0
    original_parse_success_count = 0
    repaired_success_count = 0
    final_parse_success_count = 0
    repair_type_counts = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for row in _iter_jsonl(input_path):
            total_count += 1
            original_success = bool(row.get("parse_success"))
            if original_success:
                original_parse_success_count += 1

            new_row = repair_one_row(row, tests_map)

            if new_row.get("parse_repair_applied"):
                repair_type_counts[new_row.get("parse_repair_type")] += 1
                if new_row.get("parse_success") and not original_success:
                    repaired_success_count += 1

            if new_row.get("parse_success"):
                final_parse_success_count += 1

            f_out.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    still_failed_count = total_count - final_parse_success_count

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "gold": str(gold_path),
        "total_count": total_count,
        "original_parse_success_count": original_parse_success_count,
        "repaired_success_count": repaired_success_count,
        "final_parse_success_count": final_parse_success_count,
        "still_failed_count": still_failed_count,
        "repair_type_counts": dict(repair_type_counts),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
