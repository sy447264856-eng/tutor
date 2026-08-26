#!/usr/bin/env python3
"""为下游预实验建立固定的分层抽样测试清单。

从 LongTutor-Gold（third_party/LongTutor/data/XES3G5M/human_an_updated.jsonl）
按官方 diagnosis 四类分层抽样，每类固定抽取 --per-class 条（默认 10 条，
共 40 条），生成一份供后续所有下游预实验共用的固定样本清单（manifest）。

不加载模型、不调用任何 API、不修改 third_party/LongTutor 或任何已有实验结果，
只读 Gold 数据、写一个新的 manifest JSON。

用法：
    python experiments/preexp01/create_preexperiment_manifest.py
    python experiments/preexp01/create_preexperiment_manifest.py --seed 2026 --per-class 10 \
        --output artifacts/preexp01/preexperiment_manifest_40.json
"""

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LONGTUTOR_ROOT = REPO_ROOT / "third_party" / "LongTutor"
LONGTUTOR_SCRIPTS = LONGTUTOR_ROOT / "scripts"
GOLD_PATH = LONGTUTOR_ROOT / "data" / "XES3G5M" / "human_an_updated.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "preexp01" / "preexperiment_manifest_40.json"

sys.path.insert(0, str(LONGTUTOR_SCRIPTS))
# 官方 4 类诊断标签及其顺序，直接复用，不自行猜测/重新排列
from gpt_memory_diagnose import DIAGNOSES  # noqa: E402


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


def _uid_from_sample_key(sample_key):
    """sample_key 格式为 uid||sha256(question_info)（官方
    gpt_memory_diagnose._sample_key 的生成逻辑），从中拆出 uid。"""
    if not isinstance(sample_key, str) or "||" not in sample_key:
        return None
    return sample_key.split("||", 1)[0]


def load_gold_by_class(gold_path: Path) -> dict:
    """按 diagnosis 分桶。sample_index 就是 human_an_updated.jsonl 里的真实行号
    （从 0 开始，逐行读取的顺序），这也是 history_features_lastq.jsonl 前 1000
    条按位置严格对应 Gold 的那个"位置"，已在 preexp01 数据一致性检查中验证过。
    每个桶内部按 sample_index 升序排列，保证抽样前的输入顺序完全由文件内容
    决定，不依赖字典遍历顺序等未定义行为，这是"相同 seed 得到相同结果"的
    前提条件。
    """
    buckets = defaultdict(list)
    for sample_index, obj in enumerate(_iter_jsonl(gold_path)):
        diagnosis = obj.get("diagnosis")
        buckets[diagnosis].append({
            "sample_index": sample_index,
            "sample_key": obj.get("_key"),
            "uid": _uid_from_sample_key(obj.get("_key")),
            "gold_diagnosis": diagnosis,
            "gold_strategy": obj.get("strategy"),
        })
    for bucket in buckets.values():
        bucket.sort(key=lambda r: r["sample_index"])
    return buckets


def stratified_sample(buckets: dict, classes: list, per_class: int, seed: int):
    """用同一个 random.Random(seed) 按 classes 给定的顺序依次对每一类做
    rng.sample()。只要 buckets 里每一类的输入顺序固定（已在 load_gold_by_class
    里排序过），相同 seed + 相同 classes 顺序 + 相同 per_class，就一定得到
    完全相同的抽样结果。
    """
    rng = random.Random(seed)
    selected = []
    class_counts = {}
    for cls in classes:
        pool = buckets.get(cls, [])
        if len(pool) < per_class:
            raise ValueError(
                f"类别 {cls!r} 只有 {len(pool)} 条样本，不足以抽取 {per_class} 条。"
            )
        picked = rng.sample(pool, per_class)
        picked.sort(key=lambda r: r["sample_index"])
        class_counts[cls] = len(picked)
        selected.extend(picked)
    return selected, class_counts


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="从 LongTutor-Gold 按 diagnosis 四类分层抽样，生成固定的下游预实验样本清单。"
    )
    ap.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    ap.add_argument("--per-class", type=int, default=10, help="每个 diagnosis 类别抽取的样本数，默认 10")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出 manifest JSON 路径")
    ap.add_argument("--gold", type=str, default=str(GOLD_PATH), help="LongTutor-Gold 标注文件路径")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    gold_path = Path(args.gold)
    output_path = Path(args.output)

    if not gold_path.exists():
        print(f"ERROR: Gold 文件不存在: {gold_path}", file=sys.stderr)
        return 2

    buckets = load_gold_by_class(gold_path)

    missing_classes = [c for c in DIAGNOSES if c not in buckets]
    if missing_classes:
        print(f"ERROR: Gold 数据中没有找到这些 diagnosis 类别: {missing_classes}", file=sys.stderr)
        return 2

    extra_classes = sorted(set(buckets.keys()) - set(DIAGNOSES))
    if extra_classes:
        print(
            f"WARNING: Gold 数据里出现了官方 DIAGNOSES 之外的 diagnosis 取值: {extra_classes}"
            "（本次抽样忽略这些，只处理官方 4 类）",
            file=sys.stderr,
        )

    selected, class_counts = stratified_sample(buckets, DIAGNOSES, args.per_class, args.seed)

    sample_keys = [s["sample_key"] for s in selected]
    duplicate_keys = len(sample_keys) != len(set(sample_keys))
    sample_indices_sorted = sorted(s["sample_index"] for s in selected)

    manifest = {
        "seed": args.seed,
        "per_class": args.per_class,
        "total_count": len(selected),
        "class_counts": class_counts,
        "source_gold_path": str(gold_path),
        "longtutor_commit": _git_commit(LONGTUTOR_ROOT),
        # samples 按 DIAGNOSES 的官方顺序分组、组内按 sample_index 升序排列，
        # 方便下游按类别成批处理；下面终端打印的是全局按 sample_index 排序的
        # 40 个编号，只是为了方便肉眼检查分布，不代表 manifest 里的存储顺序。
        "samples": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- 进程内复现性自检：同一个 buckets，用同样的 seed 再采一次，比较结果 ----
    selected_again, _ = stratified_sample(buckets, DIAGNOSES, args.per_class, args.seed)
    reproducible = [s["sample_key"] for s in selected] == [s["sample_key"] for s in selected_again]

    print(f"四类各自数量: {class_counts}")
    print(f"总计: {len(selected)}")
    print(f"40 个 sample_index（按数值升序）: {sample_indices_sorted}")
    print(f"是否存在重复 sample_key: {duplicate_keys}")
    print(f"reproducibility check（同进程内用相同 seed 重新采样一次，结果是否完全一致）: {reproducible}")
    print(f"manifest 已写入: {output_path}")

    if duplicate_keys or not reproducible:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
