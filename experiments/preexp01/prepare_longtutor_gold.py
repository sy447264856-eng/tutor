#!/usr/bin/env python3
"""为 XES3G5M 可复现地生成 LongTutor 官方 Tutor 推理所需的
`history_features_lastq.jsonl`，并验证其前 1000 条与 LongTutor-Gold
（human_an_updated.jsonl）按官方 sample key 的对应关系。

设计原则：
- 不修改 third_party/LongTutor 中的任何官方文件。
- 直接 import 并复用官方函数（load_questions_map / compute_global_error_rates /
  build_history_features，来自 scripts/compute_history_stats.py；_sample_key，
  来自 scripts/gpt_memory_diagnose.py），保证 history_features 的字段结构与
  sample key 生成逻辑与官方完全一致。
- 官方 compute_history_stats.py 中激活的 `_concept_segments` 实现为
  `return concepts`（不做切分），这是面向 MOOCRadar 设计的：该数据集的
  `concepts` 本身已经是扁平字符串列表（已用真实数据核实）。
  XES3G5M 的 `concepts` 是用 "----" 分隔的层级字符串（同样已用真实数据核实），
  必须使用官方脚本顶部注释掉的 "XES3G5M-specific" 切分逻辑，这与官方
  README 的说明一致。由于不能修改官方文件，本脚本在导入官方模块后，仅在本
  进程内对 `compute_history_stats._concept_segments` 做 monkeypatch，替换为
  与官方注释代码逐字一致的实现（见下方 `_xes3g5m_concept_segments`）。

用法（在 Colab / 本地均适用，一条命令即可复现）：
    python experiments/preexp01/prepare_longtutor_gold.py
"""

import contextlib
import io
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LONGTUTOR_ROOT = REPO_ROOT / "third_party" / "LongTutor"
LONGTUTOR_SCRIPTS = LONGTUTOR_ROOT / "scripts"
LONGTUTOR_DATA = LONGTUTOR_ROOT / "data" / "XES3G5M"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "preexp01"

sys.path.insert(0, str(LONGTUTOR_SCRIPTS))

import compute_history_stats as chs  # 官方模块，原样导入，未修改磁盘文件
from gpt_memory_diagnose import _sample_key as official_sample_key  # 官方 key 生成逻辑


def _xes3g5m_concept_segments(concepts):
    """与 third_party/LongTutor/scripts/compute_history_stats.py 第 8-23 行
    （官方注释掉的函数）逐字一致的实现，仅在本进程内用于 XES3G5M 的知识点切分。
    """
    if not concepts:
        return []
    seen = set()
    segs = []
    for c in concepts:
        if not c:
            continue
        for part in str(c).split("----")[1:]:
            p = part.strip()
            if p and p not in seen:
                seen.add(p)
                segs.append(p)
    return segs


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


def generate_history_features(recent_k: int) -> Path:
    sequences_path = LONGTUTOR_DATA / "sequences_long.jsonl"
    questions_path = LONGTUTOR_DATA / "questions.jsonl"
    output_path = ARTIFACTS_DIR / "history_features_lastq.jsonl"

    questions_map = chs.load_questions_map(questions_path)
    global_err = chs.compute_global_error_rates(sequences_path)

    # 仅在本进程内切换到 XES3G5M 专用知识点切分逻辑（不修改磁盘上的官方文件）
    chs._concept_segments = _xes3g5m_concept_segments

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # 官方函数内部对每条样本 print(related_total)，此处仅屏蔽输出，不改变其逻辑
        chs.build_history_features(
            sequences_jsonl=sequences_path,
            questions_map=questions_map,
            global_error_rate=global_err,
            output_jsonl=output_path,
            recent_k=recent_k,
        )
    return output_path


def validate(output_path: Path, recent_k: int) -> dict:
    sequences_path = LONGTUTOR_DATA / "sequences_long.jsonl"
    gold_path = LONGTUTOR_DATA / "human_an_updated.jsonl"

    generated = list(_iter_jsonl(output_path))
    gold = list(_iter_jsonl(gold_path))
    seqs = list(_iter_jsonl(sequences_path))

    generated_count = len(generated)
    gold_count = len(gold)

    gold_key_set = {g.get("_key") for g in gold}

    all_gen_keys = [official_sample_key(s) for s in generated]
    dup_counter = Counter(all_gen_keys)
    duplicate_key_count = sum(1 for _, c in dup_counter.items() if c > 1)

    first1000 = generated[:1000]
    first1000_keys = all_gen_keys[:1000]
    first1000_key_set = set(first1000_keys)
    first1000_internal_dupes = len(first1000_keys) - len(first1000_key_set)

    matched_gold_count = sum(1 for k in first1000_keys if k in gold_key_set)
    unmatched_gold_count = 1000 - matched_gold_count
    exact_set_equal = (first1000_key_set == gold_key_set)

    # 按行位置对应检查：生成文件第 i 条的 uid 是否与 gold 第 i 条（由 _key 拆出的 uid）一致
    gold_uids_positional = [(g.get("_key") or "").split("||")[0] for g in gold]
    gen_uids_positional = [s.get("uid") for s in first1000]
    positional_uid_match_count = sum(
        1 for a, b in zip(gen_uids_positional, gold_uids_positional) if a == b
    )
    key_based_full_match = (matched_gold_count == 1000 and unmatched_gold_count == 0)
    positional_full_match = (positional_uid_match_count == 1000)
    conflict_detected = key_based_full_match != positional_full_match

    # 序列长度检查（覆盖全部生成样本，非仅前 1000 条）
    seq_len_map = {str(o.get("uid")): len(o.get("sequence") or []) for o in seqs}
    all_uids = [s.get("uid") for s in generated]
    all_seq_lengths = [seq_len_map.get(u) for u in all_uids]
    distinct_seq_lengths = sorted(set(l for l in all_seq_lengths if l is not None))
    all_seq_len_200 = all(l == 200 for l in all_seq_lengths)

    all_history_lengths = [len(s.get("history_info") or []) for s in generated]
    distinct_history_lengths = sorted(set(all_history_lengths))
    all_history_len_199 = all(l == 199 for l in all_history_lengths)

    # 当前题是否确实为序列最后一题：question_info 首个 "[qid]" 与 sequence 最后一项 question 字段比对
    seq_last_qid_map = {
        str(o.get("uid")): str((o.get("sequence") or [{}])[-1].get("question"))
        for o in seqs
    }
    mismatched_current = []
    for s in generated:
        uid = s.get("uid")
        qinfo = s.get("question_info", "")
        rendered_qid = qinfo.split("]")[0].lstrip("[") if qinfo.startswith("[") else None
        expected_qid = seq_last_qid_map.get(uid)
        if rendered_qid != expected_qid:
            mismatched_current.append(uid)

    sample_preview = None
    if generated:
        s = generated[0]
        sample_preview = {
            "field_names": sorted(s.keys()),
            "uid": s.get("uid"),
            "question_info_preview": s.get("question_info", "")[:200],
            "history_info_count": len(s.get("history_info") or []),
            "related_history_count": len(s.get("related_history") or []),
            "features": s.get("features"),
            "_sample_key": official_sample_key(s),
        }

    return {
        "generated_count": generated_count,
        "gold_count": gold_count,
        "matched_gold_count": matched_gold_count,
        "unmatched_gold_count": unmatched_gold_count,
        "duplicate_key_count": duplicate_key_count,
        "first1000_internal_duplicate_count": first1000_internal_dupes,
        "first1000_vs_gold_exact_set_equal": exact_set_equal,
        "sequence_length_check": {
            "expected": 200,
            "all_equal_200": all_seq_len_200,
            "distinct_lengths_seen": distinct_seq_lengths,
        },
        "history_length_check": {
            "expected": 199,
            "all_equal_199": all_history_len_199,
            "distinct_lengths_seen": distinct_history_lengths,
        },
        "current_question_is_last_item_check": {
            "checked_count": len(generated),
            "mismatched_count": len(mismatched_current),
            "mismatched_uids_sample": mismatched_current[:10],
        },
        "positional_vs_key_correspondence": {
            "positional_uid_match_count": positional_uid_match_count,
            "key_based_full_match": key_based_full_match,
            "positional_full_match": positional_full_match,
            "conflict_detected": conflict_detected,
        },
        "recent_k_used": recent_k,
        "recent_k_note": (
            "build_history_features 的函数签名默认值为 20；官方 README/__main__ "
            "仅为 MOOCRadar 显式指定 recent_k=10，未对 XES3G5M 给出专门取值，"
            "因此这里沿用函数默认值 20，不做猜测性覆盖。"
        ),
        "sample_preview": sample_preview,
    }


def main() -> int:
    recent_k = 20
    longtutor_commit = _git_commit(LONGTUTOR_ROOT)
    generated_at = datetime.now(timezone.utc).isoformat()

    output_path = generate_history_features(recent_k=recent_k)
    report = validate(output_path, recent_k=recent_k)
    report["longtutor_commit"] = longtutor_commit
    report["generated_at"] = generated_at
    report["output_path"] = str(output_path.relative_to(REPO_ROOT))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    check_path = ARTIFACTS_DIR / "data_check.json"
    with check_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["positional_vs_key_correspondence"]["conflict_detected"]:
        print(
            "\n[STOP] 检测到‘前 1000 条按行对应’与‘按官方 sample key 对应’结果不一致，"
            "请人工核实，不做自动假设。",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
