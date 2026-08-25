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
    ap.add_argument("--sdpa-backend", type=str, choices=["auto", "efficient"], default="auto",
                     help="auto：不干预，使用 PyTorch SDPA 默认调度；"
                          "efficient：在 model.generate() 外层用 torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) "
                          "强制只允许 memory-efficient attention backend，其他 backend 不可用时会明确报错，不会静默退回 math backend")
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


def estimate_naive_attention_gib(batch_size: int, num_attention_heads: int, seq_len: int, element_size_bytes: int) -> float:
    """理论诊断值：若某个 attention 实现显式构造完整的
    [batch, num_attention_heads, seq_len, seq_len] score matrix，
    单层大约需要多少 GiB。只是用来判断显存需求是否与 O(seq_len^2) attention
    展开的量级吻合，不代表 flash / memory-efficient attention 实际会分配这么多
    （它们不显式物化完整 score matrix）。element_size_bytes 由调用方按实际
    可能采用的精度传入（例如 fp16=2、fp32=4），不在此处假设某个固定精度。
    """
    total_bytes = batch_size * num_attention_heads * (seq_len ** 2) * element_size_bytes
    return total_bytes / (1024 ** 3)


def get_model_arch_info(model) -> dict:
    """从模型真实 config 读取结构信息，不使用任何猜测的固定数字。"""
    config = model.config
    num_attention_heads = getattr(config, "num_attention_heads")
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    hidden_size = getattr(config, "hidden_size")
    head_dim = getattr(config, "head_dim", None) or (hidden_size // num_attention_heads)
    num_hidden_layers = getattr(config, "num_hidden_layers")
    attn_implementation = getattr(config, "_attn_implementation", None)
    return {
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_hidden_layers": num_hidden_layers,
        "attn_implementation": attn_implementation,
    }


def get_sdpa_backend_status() -> dict:
    """查询 PyTorch SDPA 各 backend 是否被全局启用/可用；这是"是否 enabled"，
    不是"这次调用实际选中了哪个 kernel"——PyTorch 没有稳定公开的 API 能可靠
    回答后一个问题，因此这里不做这种声称。任何在当前 torch 版本里查不到的
    getter，都记录为字符串说明，而不是猜一个布尔值。
    """
    import torch

    status = {}
    for key, fn_name in [
        ("flash_sdp_enabled", "flash_sdp_enabled"),
        ("mem_efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
        ("math_sdp_enabled", "math_sdp_enabled"),
    ]:
        fn = getattr(torch.backends.cuda, fn_name, None)
        if fn is None:
            status[key] = "unknown (this torch version has no torch.backends.cuda.%s)" % fn_name
        else:
            try:
                status[key] = bool(fn())
            except Exception as e:
                status[key] = f"unknown (query failed: {e})"
    return status


def require_efficient_sdpa_context():
    """返回一个只允许 memory-efficient attention backend 的上下文管理器。

    使用当前 PyTorch 推荐的新 API（torch.nn.attention.sdpa_kernel +
    SDPBackend.EFFICIENT_ATTENTION），不使用已废弃的
    torch.backends.cuda.sdp_kernel(...) 写法。如果当前 PyTorch 版本没有这个
    API，直接报错（不静默退回默认调度），错误信息以 "efficient SDPA
    unsupported" 开头，方便在 run.log 里检索。
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError as e:
        raise RuntimeError(
            f"efficient SDPA unsupported: 当前 PyTorch 版本没有 "
            f"torch.nn.attention.sdpa_kernel / SDPBackend（{e}），无法强制 "
            f"memory-efficient attention backend。"
        )
    try:
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    except Exception as e:
        raise RuntimeError(
            f"efficient SDPA unsupported: 构造 sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) "
            f"上下文失败（{e}）。"
        )


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
        # 显式要求走 torch 的 SDPA 路径（而不是 eager attention）：
        # --sdpa-backend efficient 是通过 torch.nn.attention.sdpa_kernel 上下文
        # 强制 attention backend 的，这只对调用了 F.scaled_dot_product_attention
        # 的实现有效；如果模型走的是 eager attention，强制 SDPA backend 不会
        # 生效（也不会报错，测试会失去意义），所以这里显式固定为 "sdpa"。
        attn_implementation="sdpa",
    )
    model.eval()

    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU: {gpu_name} | 显存总量: {total_mem_gb:.1f} GB | 量化方式: 4-bit NF4 (bitsandbytes)")

    return model, tokenizer, gpu_name, total_mem_gb


def generate_local(
    model, tokenizer, messages, max_new_tokens: int, stage_tracker: dict,
    arch_info: dict, sdpa_backend: str, logger: logging.Logger,
):
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
    batch_size = int(encoded["input_ids"].shape[0])

    # ---- 真实输入规模 + 理论 attention 显存诊断（在调用 generate 之前记录，
    # 这样即使随后 OOM，这些信息也已经写进 run.log） ----
    gpu_name = torch.cuda.get_device_name(0)
    gpu_cc = torch.cuda.get_device_capability(0)
    diagnostics = {
        "exact_input_token_count": input_token_count,
        "batch_size": batch_size,
        "num_attention_heads": arch_info["num_attention_heads"],
        "num_key_value_heads": arch_info["num_key_value_heads"],
        "hidden_size": arch_info["hidden_size"],
        "head_dim": arch_info["head_dim"],
        "num_hidden_layers": arch_info["num_hidden_layers"],
        "model_compute_dtype": str(model.dtype),
        "attn_implementation": arch_info["attn_implementation"],
        "gpu_name": gpu_name,
        "gpu_compute_capability": f"{gpu_cc[0]}.{gpu_cc[1]}",
        "sdpa_backend_requested": sdpa_backend,
        "naive_attention_score_matrix_estimate_gib_fp16": round(
            estimate_naive_attention_gib(batch_size, arch_info["num_attention_heads"], input_token_count, 2), 3
        ),
        "naive_attention_score_matrix_estimate_gib_fp32": round(
            estimate_naive_attention_gib(batch_size, arch_info["num_attention_heads"], input_token_count, 4), 3
        ),
    }
    stage_tracker["diagnostics"] = diagnostics

    logger.info(
        f"[diag] exact_input_token_count={diagnostics['exact_input_token_count']} "
        f"batch_size={diagnostics['batch_size']} num_attention_heads={diagnostics['num_attention_heads']} "
        f"num_key_value_heads={diagnostics['num_key_value_heads']} hidden_size={diagnostics['hidden_size']} "
        f"head_dim={diagnostics['head_dim']} num_hidden_layers={diagnostics['num_hidden_layers']} "
        f"model_compute_dtype={diagnostics['model_compute_dtype']} attn_implementation={diagnostics['attn_implementation']} "
        f"gpu={diagnostics['gpu_name']} compute_capability={diagnostics['gpu_compute_capability']} "
        f"sdpa_backend_requested={sdpa_backend}"
    )
    logger.info(
        "[diag] naive_attention_score_matrix_estimate_gib_fp16="
        f"{diagnostics['naive_attention_score_matrix_estimate_gib_fp16']} "
        "naive_attention_score_matrix_estimate_gib_fp32="
        f"{diagnostics['naive_attention_score_matrix_estimate_gib_fp32']} "
        "（理论上限：batch×heads×seq_len²×element_size 单层显式展开的量级，"
        "仅用于判断是否与 O(seq_len²) attention 展开吻合，不代表 flash/efficient "
        "attention 实际会分配这么多显存）"
    )

    stage_tracker["stage"] = "generate"
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    start = time.time()
    if sdpa_backend == "efficient":
        cm = require_efficient_sdpa_context()
        try:
            with torch.inference_mode():
                with cm:
                    output_ids = model.generate(**encoded, **generate_kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                # 显存不足是"efficient backend 被选中但仍然不够用"，不是"不支持"，原样抛出
                raise
            raise RuntimeError(f"efficient SDPA unsupported: {msg}") from e
    else:
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generate_kwargs)
    elapsed = time.time() - start

    stage_tracker["stage"] = "decode"
    new_tokens = output_ids[0][input_token_count:]
    output_token_count = int(new_tokens.shape[-1])
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_text, input_token_count, output_token_count, elapsed


def process_one_sample_local(
    model, tokenizer, sample: dict, test_obj: dict, max_new_tokens: int, stage_tracker: dict,
    arch_info: dict, sdpa_backend: str, logger: logging.Logger,
) -> dict:
    stage_tracker["stage"] = "build_prompt"
    mem_queries = official_extract_mem_queries(test_obj)
    # 与官方 process_one_sample 默认调用完全一致：history_mode="long", use_feature=False, use_teach=False
    system_prompt, user_prompt = official_build_prompts(sample, test_obj)
    messages = build_messages(system_prompt=system_prompt, user_prompt=user_prompt)

    raw_text, in_tok, out_tok, elapsed = generate_local(
        model, tokenizer, messages, max_new_tokens, stage_tracker, arch_info, sdpa_backend, logger
    )

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
        "diagnostics": stage_tracker.get("diagnostics"),
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

    arch_info = get_model_arch_info(model)
    sdpa_status = get_sdpa_backend_status()
    gpu_cc = torch.cuda.get_device_capability(0)
    gpu_cc_str = f"{gpu_cc[0]}.{gpu_cc[1]}"
    is_t4 = "T4" in gpu_name
    logger.info(
        f"[env] torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu_name} "
        f"compute_capability={gpu_cc_str} is_t4={is_t4} attn_implementation={arch_info['attn_implementation']} "
        f"sdpa_backend_requested={args.sdpa_backend}"
    )
    logger.info(
        f"[env] sdpa_flash_sdp_enabled={sdpa_status['flash_sdp_enabled']} "
        f"sdpa_mem_efficient_sdp_enabled={sdpa_status['mem_efficient_sdp_enabled']} "
        f"sdpa_math_sdp_enabled={sdpa_status['math_sdp_enabled']} "
        "（这些是 backend 是否被全局启用/可用，不代表某次调用实际选中了哪个 kernel，"
        "PyTorch 没有稳定 API 能可靠回答后者）"
    )

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

            stage_tracker = {"stage": "init", "diagnostics": None}
            try:
                result = process_one_sample_local(
                    model, tokenizer, sample, test_obj, args.max_new_tokens, stage_tracker,
                    arch_info, args.sdpa_backend, logger,
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
                    # 即使在 generate 阶段 OOM，move_inputs 阶段已经算好的诊断信息
                    # 也会保留在这里，不会因为异常而丢失。
                    "diagnostics": stage_tracker.get("diagnostics"),
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
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": round(total_mem_gb, 1),
        "gpu_compute_capability": gpu_cc_str,
        "is_t4": is_t4,
        "sdpa_backend_requested": args.sdpa_backend,
        "sdpa_flash_sdp_enabled": sdpa_status["flash_sdp_enabled"],
        "sdpa_mem_efficient_sdp_enabled": sdpa_status["mem_efficient_sdp_enabled"],
        "sdpa_math_sdp_enabled": sdpa_status["math_sdp_enabled"],
        "model_arch": arch_info,
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
