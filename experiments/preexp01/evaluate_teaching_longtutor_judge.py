#!/usr/bin/env python3
"""对 gold_state_teach_greedy 等条件下的 predictions.jsonl，用 LongTutor 官方
Teaching Action LLM-as-a-Judge 逐条打分（History Utilization / Strategy
Alignment / Coherence / Appropriateness 四维 + 官方 overall_score）。

这不是重新实现一套评价器。评分逻辑 100% 复用官方：
    third_party/LongTutor/scripts/compute_ai_tutor_eval_metrics.py
        ::GradeConfig            —— 官方评分调用的配置 dataclass
        ::_grade_content_with_gpt —— 官方 Teaching Content Judge 本体：
            官方 Judge system prompt（评分 rubric + 输出 JSON schema）就定义在
            这个函数体内部，本脚本直接 import 并调用这个函数，从未复制/改写过
            官方 prompt 文本。
    third_party/LongTutor/scripts/openai_helper.py
        ::chat_completion_with_retry —— 官方 OpenAI-compatible 调用封装（重试/
            超时都由官方实现）
        ::build_messages / extract_json —— 官方消息构造 / JSON 提取

⚠️ 已确认的一个论文-代码不一致点（如实报告，不擅自"纠正"）：LongTutor 论文
Figure 6 展示的 Judge 输出 schema 第四维字段名是 `appropriateness_score`；但
`third_party/LongTutor/scripts/compute_ai_tutor_eval_metrics.py` 里实际部署的
`_grade_content_with_gpt` 用的字段名是 `comprehension_score`（rubric 内容本身
和论文 Appropriateness 维度的定义一致——"简洁、贴合学习者最近发展区、少术语"
——只是字段名不同）。本脚本按"官方源码实际实现为准"的原则，如实使用
`comprehension_score`，不会为了跟论文文字对齐而自己改名或改写官方 prompt。

论文 Eq.(3) 把 "Teaching Score" 定义为这四个维度分数的算术平均
（History Utilization / Strategy Alignment / Coherence / Appropriateness），
这是一个论文层面的派生指标，官方代码里没有直接算这个平均值——官方代码里的
`overall_score` 是 Judge 自己在同一次 JSON 输出里给出的整体打分，用于论文
Table 9 的人类-LLM一致性分析，跟 Eq.(3) 的 Teaching Score 是两个不同的数。
本脚本两个都算、都保存，不混为一谈：
    - `overall_score`     —— Judge 自己给的整体分（官方字段，原样保存）
    - `teaching_score`    —— 本样本 4 个维度算术平均（本脚本按论文 Eq.3 派生计算）

本脚本新增的部分仅仅是"胶水层"：
    - 读取本项目自己的 predictions.jsonl（sample_index/sample_key/uid/content
      等字段，是我们自己 run_qwen_baseline.py 的 schema，不是官方 pred 文件
      的 `_key` schema），按 sample_key 匹配 Gold 的 `_key`；
    - 每条样本独立调用一次官方 `_grade_content_with_gpt`（不把多条样本塞进
      同一个对话上下文）；
    - 断点续跑（按 sample_key 记录已成功的样本，允许失败样本重跑，不因为一次
      API 报错丢失其它已完成的结果）；
    - dry-run：通过 monkeypatch 官方 `chat_completion_with_retry` 为一个不
      发网络请求的桩函数，直接截获官方代码真正会发送的 messages（等价于
      "跑一遍官方函数，但把网络调用换成打印"），保证 dry-run 展示的 Prompt
      与真实运行时 100% 一致，不是脚本自己另外拼出来的等价文本。

用法：
    python experiments/preexp01/evaluate_teaching_longtutor_judge.py \
        --predictions /path/to/predictions.jsonl \
        --output-dir artifacts/preexp01/gold_state_teach_greedy_hist20_40_judge \
        --model gemini-3-flash-preview \
        --base-url https://api.gpt.ge/v1/ \
        --temperature 0 \
        --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LONGTUTOR_ROOT = REPO_ROOT / "third_party" / "LongTutor"
LONGTUTOR_SCRIPTS = LONGTUTOR_ROOT / "scripts"
GOLD_PATH = LONGTUTOR_ROOT / "data" / "XES3G5M" / "human_an_updated.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "preexp01" / "teaching_judge"

if not (LONGTUTOR_SCRIPTS / "compute_ai_tutor_eval_metrics.py").exists():
    raise RuntimeError(
        f"未找到 {LONGTUTOR_SCRIPTS / 'compute_ai_tutor_eval_metrics.py'}。"
        "third_party/LongTutor 这个 submodule 似乎没有被检出，请先在仓库根目录运行："
        "`git submodule update --init --recursive`"
    )

sys.path.insert(0, str(LONGTUTOR_SCRIPTS))

# ---- 官方 Teaching Judge 实现，原样 import，不修改 third_party 中的任何文件 ----
import compute_ai_tutor_eval_metrics as official_judge  # noqa: E402

_ORIGINAL_CHAT_COMPLETION = official_judge.chat_completion_with_retry

# 官方 _grade_content_with_gpt 实际输出/评分用到的字段名（逐字取自官方源码，
# 不是本脚本猜的）：
OFFICIAL_SCORE_KEYS = ["history_score", "strategy_score", "coherence_score", "comprehension_score", "overall_score"]
# 论文 Eq.(3) Teaching Score 用到的 4 个维度（不含官方自评的 overall_score）：
TEACHING_SCORE_DIMENSIONS = ["history_score", "strategy_score", "coherence_score", "comprehension_score"]

DIAGNOSIS_ORDER = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")

# 论文里明确不允许在这次预实验使用的模型变体。
DISALLOWED_MODELS = {
    "gemini-3-flash-preview-search",
    "gemini-3-flash-preview-thinking",
    "gemini-3-flash-preview-nothinking",
}


def _git_commit(path: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gold_map(gold_path: Path) -> dict:
    m = {}
    for obj in _iter_jsonl(gold_path):
        key = obj.get("_key")
        if key:
            m[key] = obj
    return m


def _load_resume_state(judge_predictions_path: Path):
    """跟 run_qwen_baseline.py 里的断点续跑逻辑一致：只把 judge_parse_success=true
    的样本视为完成并跳过；失败记录不保留，允许下次重跑，避免同一个 sample_key
    反复堆积失败记录。"""
    success_keys = set()
    kept_rows = []
    if not judge_predictions_path.exists():
        return success_keys, kept_rows
    for row in _iter_jsonl(judge_predictions_path):
        k = row.get("sample_key")
        if not k:
            continue
        if row.get("judge_parse_success"):
            success_keys.add(k)
            kept_rows.append(row)
    return success_keys, kept_rows


def _make_dry_run_stub(capture: list):
    """替换掉官方 chat_completion_with_retry：不发任何网络请求，只是把官方
    _grade_content_with_gpt 真正会发送的 messages/model/temperature 截获下来，
    然后返回一个能通过官方 _validate 的假 JSON，避免官方内部重试逻辑因为
    "看起来像失败" 而空转。dry-run 只关心被截获的 messages 内容本身。
    """

    def _stub(*args, **kwargs):
        messages = kwargs.get("messages")
        if messages is None and args:
            messages = args[0]
        capture.append({
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "max_completion_tokens": kwargs.get("max_completion_tokens"),
            "messages": messages,
        })
        return json.dumps({
            "history_score": 3, "strategy_score": 3, "coherence_score": 3,
            "comprehension_score": 3, "overall_score": 3,
            "reason": "[DRY-RUN STUB —— 未真正调用 API，这个分数没有意义]",
        })

    return _stub


def _make_capturing_chat_completion(raw_capture: list, api_key, base_url):
    """真实运行用：调用官方原始 chat_completion_with_retry，但 (1) 把每次真实
    API 调用返回的原始文本记录下来，供审计；(2) 用 setdefault 的方式把
    --api-key/--base-url 传给官方函数——官方 _gpt_json 调用
    chat_completion_with_retry 时并没有显式传 api_key/base_url（依赖环境变量），
    这里用 kwargs.setdefault 补上，不会覆盖官方代码本来就会传的参数。
    """

    def _wrapped(*args, **kwargs):
        kwargs.setdefault("api_key", api_key)
        kwargs.setdefault("base_url", base_url)
        text = _ORIGINAL_CHAT_COMPLETION(*args, **kwargs)
        raw_capture.append(text)
        return text

    return _wrapped


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="用 LongTutor 官方 Teaching Action LLM-as-a-Judge 给 predictions.jsonl 打分（只复用官方评分逻辑，不重新设计）。"
    )
    ap.add_argument("--predictions", type=str, required=True, help="待评价的 predictions.jsonl 路径")
    ap.add_argument("--gold", type=str, default=str(GOLD_PATH), help="LongTutor-Gold 标注文件路径")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    ap.add_argument("--model", type=str, default="gemini-3-flash-preview")
    ap.add_argument("--base-url", type=str, default="https://api.gpt.ge/v1/")
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--max-completion-tokens", type=int, default=2500,
                     help="官方 main() 里的默认值同样是 2500")
    ap.add_argument("--max-retries", type=int, default=10,
                     help="官方 GradeConfig.max_retries 的默认值同样是 10（JSON 解析/校验失败时的重试次数）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条，默认处理全部")
    ap.add_argument("--api-key", type=str, default=None,
                     help="不推荐：优先用环境变量 OPENAI_API_KEY。传入的值绝不会被打印/写入任何文件。")
    ap.add_argument("--dry-run", action="store_true",
                     help="不发任何 API 请求；只展示前 2 条最终会发送给 Judge 的 Prompt（system+user），"
                          "用于确认输入内容正确、且官方 prompt 未被改写")
    return ap


def resolve_pred_content_and_gold(row: dict, gold_map: dict):
    sample_key = row.get("sample_key")
    gold_obj = gold_map.get(sample_key)
    pred_content = row.get("content") or ""
    gold_content = (gold_obj.get("content") or "") if gold_obj else ""
    return pred_content, gold_content, gold_obj


def run_dry_check(rows: list, gold_map: dict, cfg, logger_print) -> int:
    preview_rows = rows[:2]
    if not preview_rows:
        logger_print("没有可预览的样本")
        return 1

    ok = True
    for row in preview_rows:
        pred_content, gold_content, gold_obj = resolve_pred_content_and_gold(row, gold_map)
        if gold_obj is None:
            logger_print(f"index={row.get('sample_index')} 未在 Gold 里找到 sample_key={row.get('sample_key')}，跳过")
            ok = False
            continue

        capture = []
        official_judge.chat_completion_with_retry = _make_dry_run_stub(capture)
        try:
            official_judge._grade_content_with_gpt(cfg, pred_content=pred_content, gold_content=gold_content)
        finally:
            official_judge.chat_completion_with_retry = _ORIGINAL_CHAT_COMPLETION

        if not capture:
            logger_print(
                f"index={row.get('sample_index')}: 官方 _grade_content_with_gpt 没有触发任何 API 调用"
                f"（通常是因为 pred_content 为空，官方逻辑会直接给 1 分并跳过 Judge，不需要 Prompt）"
            )
            continue

        call = capture[0]
        logger_print("=" * 80)
        logger_print(f"sample_index = {row.get('sample_index')}")
        logger_print(f"sample_key   = {row.get('sample_key')}")
        logger_print(f"model        = {call['model']}")
        logger_print(f"temperature  = {call['temperature']}")
        logger_print("---- messages（官方 build_messages 构造，原样截获）----")
        for m in call["messages"]:
            logger_print(f"[{m['role']}]")
            logger_print(m["content"])
        logger_print("=" * 80)

    return 0 if ok else 1


def main() -> int:
    args = build_argparser().parse_args()

    if args.model in DISALLOWED_MODELS:
        print(f"ERROR: 本次预实验不允许使用 {args.model}（只允许 gemini-3-flash-preview）", file=sys.stderr)
        return 2

    predictions_path = Path(args.predictions)
    gold_path = Path(args.gold)
    output_dir = Path(args.output_dir)

    if not predictions_path.exists():
        print(f"ERROR: 未找到 predictions 文件: {predictions_path}", file=sys.stderr)
        return 2
    if not gold_path.exists():
        print(f"ERROR: 未找到 Gold 文件: {gold_path}", file=sys.stderr)
        return 2

    rows = list(_iter_jsonl(predictions_path))
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print(f"ERROR: {predictions_path} 是空文件（或 --limit 太小）", file=sys.stderr)
        return 2

    gold_map = load_gold_map(gold_path)

    cfg = official_judge.GradeConfig(
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        debug=False,
    )

    if args.dry_run:
        return run_dry_check(rows, gold_map, cfg, print)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: 没有可用的 API Key。请设置环境变量 OPENAI_API_KEY，"
            "或者用 --api-key 传入（不推荐，命令行参数可能被记录进 shell 历史）。",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    judge_predictions_path = output_dir / "judge_predictions.jsonl"
    judge_summary_path = output_dir / "judge_summary.json"
    judge_run_config_path = output_dir / "judge_run_config.json"

    success_keys, kept_rows = _load_resume_state(judge_predictions_path)
    if judge_predictions_path.exists():
        n_prev = sum(1 for _ in _iter_jsonl(judge_predictions_path))
        n_dropped = n_prev - len(kept_rows)
        with judge_predictions_path.open("w", encoding="utf-8") as f_clean:
            for row in kept_rows:
                f_clean.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"重跑前整理 judge_predictions.jsonl：保留 {len(kept_rows)} 条已成功评分（将跳过），"
            f"移除 {n_dropped} 条失败记录（将重跑）"
        )

    n_total = len(rows)
    n_done = n_success = n_fail = 0

    with judge_predictions_path.open("a", encoding="utf-8") as f_out:
        for row in rows:
            sample_index = row.get("sample_index")
            sample_key = row.get("sample_key")
            uid = row.get("uid")

            if sample_key in success_keys:
                print(f"[{n_done + 1}/{n_total}] sample_index={sample_index} 已成功评分，跳过")
                n_done += 1
                n_success += 1
                continue

            pred_content, gold_content, gold_obj = resolve_pred_content_and_gold(row, gold_map)
            if gold_obj is None:
                print(f"[{n_done + 1}/{n_total}] sample_index={sample_index} 未在 Gold 里找到 sample_key={sample_key}，跳过")
                n_done += 1
                n_fail += 1
                continue

            raw_capture = []
            official_judge.chat_completion_with_retry = _make_capturing_chat_completion(raw_capture, api_key, args.base_url)
            try:
                scores, reason, err = official_judge._grade_content_with_gpt(
                    cfg, pred_content=pred_content, gold_content=gold_content
                )
            except Exception as e:
                scores, reason, err = None, None, f"exception:{type(e).__name__}:{e}"
            finally:
                official_judge.chat_completion_with_retry = _ORIGINAL_CHAT_COMPLETION

            judge_parse_success = scores is not None
            teaching_score = None
            if judge_parse_success:
                vals = [scores.get(k) for k in TEACHING_SCORE_DIMENSIONS]
                if all(isinstance(v, int) for v in vals):
                    teaching_score = round(sum(vals) / len(vals), 3)

            out_row = {
                "sample_index": sample_index,
                "sample_key": sample_key,
                "uid": uid,
                "gold_diagnosis": row.get("gold_diagnosis"),
                "gold_strategy": row.get("gold_strategy"),
                "judge_model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "history_score": scores.get("history_score") if scores else None,
                "strategy_score": scores.get("strategy_score") if scores else None,
                "coherence_score": scores.get("coherence_score") if scores else None,
                "comprehension_score": scores.get("comprehension_score") if scores else None,
                "overall_score": scores.get("overall_score") if scores else None,
                "teaching_score": teaching_score,
                "reason": reason,
                "judge_parse_success": judge_parse_success,
                "judge_error": err,
                "raw_judge_output": raw_capture,
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

            n_done += 1
            n_success += int(judge_parse_success)
            n_fail += int(not judge_parse_success)
            print(
                f"[{n_done}/{n_total}] sample_index={sample_index} judge_parse_success={judge_parse_success} "
                f"teaching_score={teaching_score} overall_score={out_row['overall_score']} err={err}"
            )

    # ---- 汇总 ----
    all_rows = list(_iter_jsonl(judge_predictions_path))
    success_rows = [r for r in all_rows if r.get("judge_parse_success")]

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "total_count": len(all_rows),
        "success_count": len(success_rows),
        "failed_count": len(all_rows) - len(success_rows),
        "history_score_mean": _mean([r.get("history_score") for r in success_rows]),
        "strategy_score_mean": _mean([r.get("strategy_score") for r in success_rows]),
        "coherence_score_mean": _mean([r.get("coherence_score") for r in success_rows]),
        "comprehension_score_mean": _mean([r.get("comprehension_score") for r in success_rows]),
        "judge_overall_score_mean": _mean([r.get("overall_score") for r in success_rows]),
        "teaching_score_mean": _mean([r.get("teaching_score") for r in success_rows]),
        "teaching_score_definition": (
            "论文 Eq.(3)：每条样本的 history_score/strategy_score/coherence_score/"
            "comprehension_score 四个维度算术平均，再对所有样本取平均。"
            "不是官方代码里 Judge 自评的 overall_score（那个是单独的 judge_overall_score_mean）。"
        ),
        "dimension_name_mapping_to_paper": {
            "history_score": "History Utilization",
            "strategy_score": "Strategy Alignment",
            "coherence_score": "Coherence",
            "comprehension_score": "Appropriateness（论文 Figure 6 文本写的字段名是 appropriateness_score，"
                                    "但官方代码实际输出字段名是 comprehension_score，本报告如实按代码字段命名）",
        },
        "by_gold_diagnosis": {},
    }
    for diag in DIAGNOSIS_ORDER:
        diag_rows = [r for r in success_rows if r.get("gold_diagnosis") == diag]
        summary["by_gold_diagnosis"][diag] = {
            "n": len(diag_rows),
            "history_score_mean": _mean([r.get("history_score") for r in diag_rows]),
            "strategy_score_mean": _mean([r.get("strategy_score") for r in diag_rows]),
            "coherence_score_mean": _mean([r.get("coherence_score") for r in diag_rows]),
            "comprehension_score_mean": _mean([r.get("comprehension_score") for r in diag_rows]),
            "judge_overall_score_mean": _mean([r.get("overall_score") for r in diag_rows]),
            "teaching_score_mean": _mean([r.get("teaching_score") for r in diag_rows]),
        }

    with judge_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_config = {
        "predictions_path": str(predictions_path),
        "gold_path": str(gold_path),
        "judge_model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "max_retries": args.max_retries,
        "judge_prompt_source": (
            "third_party/LongTutor/scripts/compute_ai_tutor_eval_metrics.py"
            "::_grade_content_with_gpt（官方 system_prompt 定义在函数体内，"
            "本脚本直接 import 并调用该函数，未复制/未改写 prompt 文本）"
        ),
        "judge_score_fields": OFFICIAL_SCORE_KEYS,
        "longtutor_commit": _git_commit(LONGTUTOR_ROOT),
        "total_requested": n_total,
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        # 不保存 API Key。
    }
    with judge_run_config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"完成：{n_success} 成功 / {n_fail} 失败 / 共 {n_total} 条。")
    print(f"written: {judge_predictions_path}")
    print(f"written: {judge_summary_path}")
    print(f"written: {judge_run_config_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
