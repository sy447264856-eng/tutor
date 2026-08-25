#!/usr/bin/env python3
"""LongTutor-Gold 上的 Qwen2.5-7B-Instruct baseline smoke test（本地 HuggingFace 推理）。

目标：验证"长期历史 + 当前问题 -> 官方 AI Tutor baseline"这条链路在本地
Transformers + bitsandbytes 4-bit 量化下可以跑通，供 Colab Tesla T4（约 15GB
显存）上先跑 5 条 smoke test。不涉及 Student State / CAD / ECD / Decoder。

Prompt 复用方式（忠实官方，不自行优化）：
- 直接 import third_party/LongTutor/scripts/eval_ai_tutor.py 中的
  `_build_prompts` / `_validate_output` / `_extract_mem_queries_from_test_obj` /
  `_load_tests_map`，与官方 `process_one_sample` 在默认参数下调用
  `_build_prompts(sample, test_obj)`（history_mode="long", use_feature=False,
  use_teach=False）完全一致 —— 即官方 baseline 本身就不启用统计特征说明块和
  额外教学指导块，本脚本对此不做任何"优化"或补充。
- 直接 import third_party/LongTutor/scripts/gpt_memory_diagnose.py 中的
  `DIAGNOSES` / `strategy_MAP`（供 `_validate_output` 内部使用）以及
  `_sample_key`（官方样本 key 生成逻辑，与上一阶段 prepare_longtutor_gold.py
  使用的完全一致）。
- 直接 import third_party/LongTutor/scripts/openai_helper.py 中的
  `build_messages`（构造 system/user 消息列表，与官方喂给 OpenAI 的消息结构
  完全一致，这里改为喂给本地 Qwen 的 chat template）和 `extract_json`
  （官方的 JSON 提取逻辑）。

与官方 eval_ai_tutor.py 的唯一差异：官方通过 OpenAI 兼容 API
（`chat_completion_with_retry`）调用远程模型；本脚本改为本地
AutoModelForCausalLM.generate() 调用 Qwen，且严格贪心解码
（do_sample=False，不设置 temperature/top_p），因此不做官方那种"多次重试
换取不同采样结果"的循环 —— 贪心解码下重试会产生完全相同的输出，重试没有
意义，本脚本对每条样本只生成一次。

官方输出 schema（原样保留，不新增/不臆造字段）：
    {
      "memory": [{"id": "Q1", "answer": "..."}, ...],
      "diagnosis": "<DIAGNOSES 中的一个>",
      "reason": "...",
      "strategy": "<与 diagnosis 对应的策略>",
      "content": "..."
    }
官方 schema 中没有独立的 "evidence" 字段；memory 列表本身就承载了
"证据式问答"。本脚本不会臆造 evidence 字段。
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LONGTUTOR_ROOT = REPO_ROOT / "third_party" / "LongTutor"
LONGTUTOR_SCRIPTS = LONGTUTOR_ROOT / "scripts"
LONGTUTOR_DATA = LONGTUTOR_ROOT / "data" / "XES3G5M"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "preexp01"
FEATURES_PATH = ARTIFACTS_DIR / "history_features_lastq.jsonl"
GOLD_PATH = LONGTUTOR_DATA / "human_an_updated.jsonl"

sys.path.insert(0, str(LONGTUTOR_SCRIPTS))

# ---- 官方函数 / schema，直接复用，不修改 third_party 中的任何文件 ----
from eval_ai_tutor import (  # noqa: E402
    _build_prompts as official_build_prompts,
    _validate_output as official_validate_output,
    _extract_mem_queries_from_test_obj as official_extract_mem_queries,
    _load_tests_map as official_load_tests_map,
)
from gpt_memory_diagnose import _sample_key as official_sample_key  # noqa: E402
from openai_helper import build_messages, extract_json  # noqa: E402


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


def _load_slice(path: Path, start_index: int, sample_size: int):
    """只读取需要的那一段（不把 240MB 文件整体读进内存也可以正常工作）。"""
    end_index = start_index + sample_size
    out = []
    for i, obj in enumerate(_iter_jsonl(path)):
        if i < start_index:
            continue
        if i >= end_index:
            break
        out.append((i, obj))
    return out


def _align_memory(parsed: dict, mem_queries: list) -> list:
    """与官方 eval_ai_tutor.py::process_one_sample 中的对齐逻辑逐字一致
    （该逻辑是 process_one_sample 内联代码，非独立可导入函数，故在此复刻，
    行为与官方保持一致：按 id 优先匹配，缺失则按顺序回退）。
    """
    raw_mem = parsed.get("memory", [])
    mem_map = {}
    for idx, item in enumerate(raw_mem):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        key_id = str(item_id).strip().upper() if item_id else f"Q{idx + 1}"
        mem_map[key_id] = str(item.get("answer", ""))

    aligned = []
    for i in range(len(mem_queries)):
        target_id = f"Q{i + 1}"
        aligned.append({"id": target_id, "answer": mem_map.get(target_id, "")})
    return aligned


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "LongTutor-Gold (XES3G5M) 上的 Qwen2.5-7B-Instruct baseline smoke test。"
            "复用官方 eval_ai_tutor.py 的 prompt 与输出 schema，本地 4-bit 推理。"
        )
    )
    ap.add_argument("--sample-size", type=int, default=5, help="本次读取的样本数量，默认 5")
    ap.add_argument("--start-index", type=int, default=0, help="从 history_features_lastq.jsonl 第几行开始读取，默认 0")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                     help="生成的最大新 token 数（默认 1024；官方 API baseline 用的是 3000，"
                          "但那是面向远程 API 的宽松上限，本地 T4 上 smoke test 用更保守的默认值，"
                          "可用本参数按需调大）")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR / "qwen_baseline_smoke"))
    ap.add_argument("--dry-run", action="store_true",
                     help="不加载模型、不做任何生成；只做数据读取/prompt构造/key对齐的静态检查"
                          "（用于没有 GPU 的环境，例如先在本机确认逻辑正确，再上 Colab 跑真实推理）")
    return ap


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qwen_baseline_smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_resume_state(predictions_path: Path):
    """区分成功与失败样本，用于断点重跑：

    - `parse_success=true` 的样本视为已完成，跳过；
    - 之前失败（没有 raw_output / parse_success=false）的样本允许重跑，
      且不会保留在文件里累积——重跑前会用只含成功样本的干净版本重写
      predictions.jsonl，避免同一个 sample_key 反复堆积失败记录。

    返回 (success_keys, kept_rows)。
    """
    success_keys = set()
    kept_rows = []
    if not predictions_path.exists():
        return success_keys, kept_rows
    for row in _iter_jsonl(predictions_path):
        k = row.get("sample_key")
        if not k:
            continue
        if row.get("parse_success"):
            success_keys.add(k)
            kept_rows.append(row)
        # 失败记录（parse_success 非 true）不保留，允许下次重跑重新生成
    return success_keys, kept_rows


def load_model_and_tokenizer(model_name: str, logger: logging.Logger):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到可用的 CUDA GPU。本脚本按 T4（约 15GB 显存）4-bit 量化设计，"
            "不会自动退回 CPU 加载 7B 模型。请在 Colab（运行时类型选择 GPU / T4）"
            "或其他带 NVIDIA GPU 的环境中运行；如需在无 GPU 环境下检查脚本逻辑，"
            "请加 --dry-run。"
        )

    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"bitsandbytes 不可用（{e}）。4-bit 量化依赖 bitsandbytes，"
            "本脚本不会自动退回全精度加载 7B 模型。请先 `pip install bitsandbytes` 后重试。"
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU: {gpu_name} | 显存总量: {total_mem_gb:.1f} GB | 量化方式: 4-bit NF4 (bitsandbytes)")

    return model, tokenizer, gpu_name, total_mem_gb


def generate_local(model, tokenizer, messages, max_new_tokens: int, stage_tracker: dict):
    """执行 apply_chat_template -> tokenize -> move_inputs -> generate -> decode。

    每一步之前更新 stage_tracker["stage"]，这样如果中途抛异常，调用方可以从
    stage_tracker 里读到"最后成功进入的阶段"，即 failed_stage。

    与最初版本相比，这里把"渲染 chat template"和"tokenize 成张量"拆成了两步
    （先 apply_chat_template(tokenize=False) 拿到渲染后的字符串，再用
    tokenizer(...) 单独分词），而不是让 apply_chat_template 同时做
    tokenize=True + return_tensors="pt"。这样 tokenizer(...) 的返回值就是
    标准的 BatchEncoding（明确支持 `.to(device)` 和 `["input_ids"]`），
    不必依赖对 apply_chat_template 返回类型的隐含假设。
    """
    import torch

    stage_tracker["stage"] = "apply_chat_template"
    rendered_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    stage_tracker["stage"] = "tokenize"
    encoded = tokenizer(rendered_text, return_tensors="pt")

    stage_tracker["stage"] = "move_inputs"
    encoded = encoded.to(model.device)
    input_token_count = int(encoded["input_ids"].shape[-1])

    stage_tracker["stage"] = "generate"
    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    stage_tracker["stage"] = "decode"
    new_tokens = output_ids[0][input_token_count:]
    output_token_count = int(new_tokens.shape[-1])
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_text, input_token_count, output_token_count, elapsed


def process_one_sample_local(
    model, tokenizer, sample: dict, test_obj: dict, max_new_tokens: int, stage_tracker: dict
) -> dict:
    stage_tracker["stage"] = "build_prompt"
    mem_queries = official_extract_mem_queries(test_obj)
    # 与官方 process_one_sample 默认调用完全一致：history_mode="long", use_feature=False, use_teach=False
    system_prompt, user_prompt = official_build_prompts(sample, test_obj)
    messages = build_messages(system_prompt=system_prompt, user_prompt=user_prompt)

    raw_text, in_tok, out_tok, elapsed = generate_local(model, tokenizer, messages, max_new_tokens, stage_tracker)

    stage_tracker["stage"] = "parse_output"
    parse_success = False
    parse_error = None
    parsed_output = None
    memory = diagnosis = reason = strategy = content = None

    try:
        parsed = extract_json(raw_text)
    except Exception as e:
        parsed = None
        parse_error = f"json_parse_error:{e}"

    if parsed is None and parse_error is None:
        parse_error = "json_parse_error:empty_or_unparseable"

    if parsed is not None:
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

    return {
        "raw_output": raw_text,
        "parsed_output": parsed_output,
        "memory": memory,
        "diagnosis": diagnosis,
        "reason": reason,
        "strategy": strategy,
        "content": content,
        "parse_success": parse_success,
        "parse_error": parse_error,
        "failed_stage": None,
        "input_token_count": in_tok,
        "output_token_count": out_tok,
        "elapsed_seconds": round(elapsed, 3),
    }


def run_dry_check(samples, tests_map, logger: logging.Logger) -> int:
    """不加载模型的静态检查：数据读取 / sample key 对齐 / prompt 构造是否正常。"""
    n_ok = 0
    for idx, sample in samples:
        key = official_sample_key(sample)
        test_obj = tests_map.get(key)
        if test_obj is None:
            logger.error(f"[dry-run] index={idx} uid={sample.get('uid')} 未在 human_an_updated.jsonl 中找到对应 key，跳过")
            continue
        mem_queries = official_extract_mem_queries(test_obj)
        system_prompt, user_prompt = official_build_prompts(sample, test_obj)
        logger.info(
            f"[dry-run] index={idx} uid={sample.get('uid')} sample_key={key} "
            f"mem_queries={len(mem_queries)} "
            f"system_prompt_chars={len(system_prompt)} user_prompt_chars={len(user_prompt)}"
        )
        n_ok += 1
    logger.info(f"[dry-run] 完成，{n_ok}/{len(samples)} 条样本通过 key 对齐与 prompt 构造检查（未调用任何模型）")
    return 0 if n_ok == len(samples) else 1


def main() -> int:
    args = build_argparser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    run_config_path = output_dir / "run_config.json"
    log_path = output_dir / "run.log"

    logger = _setup_logging(log_path)

    if not FEATURES_PATH.exists():
        logger.error(
            f"未找到 {FEATURES_PATH.relative_to(REPO_ROOT)}。"
            "请先运行：python experiments/preexp01/prepare_longtutor_gold.py "
            "（本脚本不会静默重新生成该文件）"
        )
        return 2

    samples = _load_slice(FEATURES_PATH, args.start_index, args.sample_size)
    tests_map = official_load_tests_map(GOLD_PATH)
    logger.info(f"读取到 {len(samples)} 条待处理样本（start_index={args.start_index}, sample_size={args.sample_size}）；"
                f"Gold 测试用例共 {len(tests_map)} 条")

    if args.dry_run:
        return run_dry_check(samples, tests_map, logger)

    run_started_at = datetime.now(timezone.utc).isoformat()

    model, tokenizer, gpu_name, total_mem_gb = load_model_and_tokenizer(args.model_name, logger)

    import torch
    import transformers as _tf
    try:
        import bitsandbytes as _bnb
        bnb_version = getattr(_bnb, "__version__", "unknown")
    except Exception:
        bnb_version = None

    success_keys, kept_rows = _load_resume_state(predictions_path)
    if predictions_path.exists():
        n_prev = sum(1 for _ in _iter_jsonl(predictions_path))
        n_dropped = n_prev - len(kept_rows)
        with predictions_path.open("w", encoding="utf-8") as f_clean:
            for row in kept_rows:
                f_clean.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            f"重跑前整理 predictions.jsonl：保留 {len(kept_rows)} 条已成功样本（将跳过），"
            f"移除 {n_dropped} 条失败记录（将重跑，不再累积旧的失败结果）"
        )

    n_total = len(samples)
    n_done = n_success = n_fail = 0

    with predictions_path.open("a", encoding="utf-8") as f_out:
        for idx, sample in samples:
            key = official_sample_key(sample)
            uid = sample.get("uid")

            if key in success_keys:
                logger.info(f"[{n_done + 1}/{n_total}] index={idx} uid={uid} 已成功完成，跳过")
                n_done += 1
                n_success += 1
                continue

            test_obj = tests_map.get(key)
            if test_obj is None:
                logger.error(f"[{n_done + 1}/{n_total}] index={idx} uid={uid} 未找到对应 Gold 样本（key={key}），跳过")
                n_done += 1
                n_fail += 1
                continue

            stage_tracker = {"stage": "init"}
            try:
                result = process_one_sample_local(
                    model, tokenizer, sample, test_obj, args.max_new_tokens, stage_tracker
                )
                row = {
                    "sample_index": idx,
                    "sample_key": key,
                    "uid": uid,
                    **result,
                }
                success = bool(result["parse_success"])
            except Exception as e:
                tb = traceback.format_exc()
                failed_stage = stage_tracker.get("stage", "unknown")
                row = {
                    "sample_index": idx,
                    "sample_key": key,
                    "uid": uid,
                    "raw_output": None,
                    "parsed_output": None,
                    "memory": None,
                    "diagnosis": None,
                    "reason": None,
                    "strategy": None,
                    "content": None,
                    "parse_success": False,
                    "parse_error": f"{type(e).__name__}: {e}",
                    "failed_stage": failed_stage,
                    "input_token_count": None,
                    "output_token_count": None,
                    "elapsed_seconds": None,
                }
                success = False
                logger.error(
                    f"[{n_done + 1}/{n_total}] index={idx} uid={uid} sample_key={key} "
                    f"failed_stage={failed_stage} {type(e).__name__}: {e}"
                )
                logger.error(f"完整 traceback (index={idx} uid={uid}):\n{tb}")

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            n_done += 1
            n_success += int(success)
            n_fail += int(not success)

            logger.info(
                f"[{n_done}/{n_total}] index={idx} uid={uid} success={success} "
                f"input_tokens={row.get('input_token_count')} output_tokens={row.get('output_token_count')} "
                f"elapsed={row.get('elapsed_seconds')} parse_success={row.get('parse_success')}"
            )

    run_finished_at = datetime.now(timezone.utc).isoformat()

    run_config = {
        "model_name": args.model_name,
        "quantization": "4-bit NF4 (bitsandbytes)",
        "compute_dtype": "float16",
        "decoding": "greedy",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "sample_size": args.sample_size,
        "start_index": args.start_index,
        "dataset": "XES3G5M / LongTutor-Gold (human_an_updated.jsonl) via history_features_lastq.jsonl",
        "longtutor_commit": _git_commit(LONGTUTOR_ROOT),
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "bitsandbytes_version": bnb_version,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": round(total_mem_gb, 1),
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "n_total": n_total,
        "n_success": n_success,
        "n_fail": n_fail,
    }
    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    logger.info(f"完成：{n_success} 成功 / {n_fail} 失败 / 共 {n_total} 条。结果见 {predictions_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
