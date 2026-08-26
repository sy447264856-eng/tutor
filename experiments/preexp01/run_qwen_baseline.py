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
import copy
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

if not (LONGTUTOR_SCRIPTS / "eval_ai_tutor.py").exists():
    raise RuntimeError(
        f"未找到 {LONGTUTOR_SCRIPTS / 'eval_ai_tutor.py'}"
        f"（本文件推算出的项目根目录是 {REPO_ROOT}，推算方式：Path(__file__).resolve().parents[2]）。\n"
        "最常见原因：third_party/LongTutor 这个 git submodule 还没有被检出（目录存在但是空的）。"
        "请在仓库根目录运行：\n"
        "    git submodule update --init --recursive\n"
        "如果上面打印的 REPO_ROOT 本身就不是你的 tutor 仓库根目录，说明这个脚本文件当前所在的"
        "实际磁盘路径不是 <仓库根目录>/experiments/preexp01/run_qwen_baseline.py"
        "（例如 clone 出现了嵌套/重复的目录），请检查 clone 结构，而不是继续往下跑。"
    )

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


# LaTeX 命令名称（不含反斜杠），仅用于消歧"这个反斜杠到底是不是 JSON 合法转义"。
# 重点覆盖首字母恰好是 b/f/n/r/t 的命令——这几个字母本身也是合法的单字符 JSON
# 转义（\b \f \n \r \t），只有这种情况才存在歧义（例如 \times 以 \t 开头，
# json.loads 会把 \t 当成合法的 tab 转义，然后把 "imes" 当普通文本，不会报错，
# 而是静默产生错误内容）。其余命令（如 \(、\div、\sqrt、\le、\ge 等）本来就
# 不属于任何合法 JSON 转义，通用规则已经能正确处理，列在这里只是为了文档完整、
# 方便以后扩展。
_LATEX_COMMANDS = (
    # 首字母与合法 JSON 单字符转义冲突，必须显式识别
    "times", "tan", "to", "text", "tfrac", "theta", "triangle",
    "frac", "forall",
    "neq", "nabla", "notin", "ne",
    "binom", "bigcup", "bigcap", "boxed", "bar", "because",
    "rightarrow", "rfloor", "rceil", "rho",
    # 首字母本身就不是合法 JSON 转义，通用规则已覆盖，这里仅作说明用途
    "cdot", "div", "sqrt", "leq", "geq", "le", "ge", "pm", "approx",
    "infty", "alpha", "beta", "pi", "sum", "int", "ast", "left", "right",
    "mid", "vec", "overline", "underline", "begin", "end", "mathrm",
    "dfrac", "circ", "angle", "perp", "parallel", "cong", "sim",
)


def repair_latex_json_escapes(json_str: str) -> str:
    r"""在 JSON 字符串字面量内部，把"看起来像 LaTeX 命令、但对 json.loads 而言
    是非法或有歧义转义"的反斜杠改成两个反斜杠（合法的字面反斜杠转义），使其
    能被 json.loads 解析，且解析结果里的文本内容与模型原始输出的可见字符
    完全一致（例如 `\\times` 修复前后解析出来的都还是文本 "\times"，只是从
    "非法/有歧义转义" 变成了 "合法的字面反斜杠 + 普通字母"，不改变任何数学
    语义）。

    只处理**字符串字面量内部**的反斜杠；字符串外的 JSON 结构字符（大括号、
    冒号、逗号等）原样保留，不做任何改动。

    对每个反斜杠，按以下顺序判断：
    1. 下一个字符是 `"` `\\` `/` —— 这三者不会是任何 LaTeX 命令的首字母，没有
       歧义，是合法转义，原样保留。
    2. 下一个字符是 `u` —— 检查后面是否紧跟 4 位十六进制数字：是则是合法的
       `\\uXXXX` 转义，原样保留；不是（比如其实是 `\underline`）则不可能是
       合法转义，转义这个反斜杠。
    3. 下一个字符是 `b`/`f`/`n`/`r`/`t` —— 用 `_LATEX_COMMANDS` 检查从这个
       反斜杠往后是否精确匹配一个已知 LaTeX 命令的前缀：匹配则判定为 LaTeX，
       转义这个反斜杠（后面的字母作为普通字符，在循环的下一轮被原样保留，
       不需要一次性跳过整个命令名）；不匹配则判定为模型确实想输出一个真实的
       控制字符（退格/换页/换行/回车/制表符），原样保留，绝不重复处理。
    4. 其他任何字符——一定不在合法 JSON 转义字符集 `" \\ / b f n r t u` 里，
       转义这个反斜杠。这一条顺带处理了 `\\(` `\\)` `\\[` `\\]` `\\cdot`
       `\\sqrt` `\\le` `\\ge` 等，不依赖上面的命令表也能正确处理。
    """
    out = []
    i = 0
    n = len(json_str)
    in_string = False

    while i < n:
        c = json_str[i]

        if not in_string:
            out.append(c)
            if c == '"':
                in_string = True
            i += 1
            continue

        if c == '"':
            out.append(c)
            in_string = False
            i += 1
            continue

        if c != '\\':
            out.append(c)
            i += 1
            continue

        # c == '\\'，且当前处于字符串字面量内部
        nxt = json_str[i + 1] if i + 1 < n else ''

        if nxt in ('"', '\\', '/'):
            out.append(json_str[i:i + 2])
            i += 2
            continue

        if nxt == 'u':
            hex_part = json_str[i + 2:i + 6]
            if len(hex_part) == 4 and all(ch in '0123456789abcdefABCDEF' for ch in hex_part):
                out.append(json_str[i:i + 6])
                i += 6
                continue
            out.append('\\\\')
            i += 1
            continue

        if nxt in ('b', 'f', 'n', 'r', 't'):
            rest = json_str[i + 1:]
            if any(rest.startswith(cmd) for cmd in _LATEX_COMMANDS):
                out.append('\\\\')
                i += 1
                continue
            out.append(json_str[i:i + 2])
            i += 2
            continue

        # 其余任何字符都不在合法 JSON 转义字符集里
        out.append('\\\\')
        i += 1

    return ''.join(out)


def parse_model_json_output(raw_text):
    """两步解析：第一步用官方 `extract_json` 原样解析 `raw_text`（不做任何
    修改）；只有官方解析失败（抛异常 / 返回 None）时，才在原始文本上截取出
    JSON 子串、执行 LaTeX 反斜杠修复，再解析一次。绝不会先修改 raw_text 再
    调用官方解析。

    返回 (parsed_or_None, parse_error_or_None, repair_applied: bool,
          repair_type: Optional[str])。
    - 官方解析成功：(parsed, None, False, None)。
    - 官方解析失败、修复后解析成功：(parsed, None, True, "latex_json_escape")。
    - 两步都失败：(None, 官方解析的原始错误信息, False, None) —— 按要求保留
      的是"原始" parse_error，不会被 fallback 修复过程中的错误覆盖。
    """
    try:
        parsed = extract_json(raw_text)
    except Exception as e:
        parsed = None
        official_error = f"json_parse_error:{e}"
    else:
        official_error = None if parsed is not None else "json_parse_error:empty_or_unparseable"

    if parsed is not None:
        return parsed, None, False, None

    if not raw_text:
        return None, official_error, False, None

    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None, official_error, False, None

    json_str = raw_text[start:end + 1]
    repaired_str = repair_latex_json_escapes(json_str)

    try:
        repaired_parsed = json.loads(repaired_str)
    except Exception:
        return None, official_error, False, None

    return repaired_parsed, None, True, "latex_json_escape"


def limit_history(sample: dict, history_length: int):
    """在调用官方 `_build_prompts` 之前，把喂给 prompt 的 history_info 限制为
    最近 history_length 条。

    - 只裁剪 `history_info`（官方 `_build_prompts` 默认 history_mode="long" 时，
      正是用这个字段渲染历史文本块；`related_history` 是按知识点筛选出来的另一
      个字段，语义不是"最近 N 条"，本函数不动它）。
    - 保留原始时间顺序：history_info 本身已经按时间升序排列，直接取列表末尾的
      N 条即为"最近 N 条，原始顺序不变"。
    - 不修改每条记录的文本内容，不做任何 token 级截断——只是减少喂进去的
      记录条数。
    - history_length <= 0 或 >= 实际条数时视为不截断，使用全部历史（避免用一个
      看似合理的默认值悄悄丢数据）。

    返回 (处理后的 sample 副本, original_history_records, used_history_records)。
    """
    history_info = sample.get("history_info") or []
    original_count = len(history_info)

    if history_length is None or history_length <= 0 or history_length >= original_count:
        used_history = history_info
    else:
        used_history = history_info[-history_length:]

    # 用 deepcopy 而不是浅拷贝，确保 limited_sample 不与原始 sample 共享任何
    # 可变对象的引用（即便 history_info 目前只是字符串列表、浅拷贝本身不会
    # 造成实际问题，这里仍按要求做成完全独立的拷贝，逻辑不变）。
    limited_sample = copy.deepcopy(sample)
    limited_sample["history_info"] = copy.deepcopy(used_history)
    return limited_sample, original_count, len(used_history)


# 支持的实验 condition。"baseline" 是原有行为（不启用官方 Teaching
# Guidelines、不注入 Gold 状态），保证不传 --condition 时完全向后兼容。
# "gold_state_teach_greedy" 是本次新增的固定配方：最近 history_length 条历史
# + 当前错题 + Gold Diagnosis + Gold Teaching Strategy + 官方 Teaching
# Guidelines（use_teach=True）+ 贪心解码；不提供 Gold reason/content/memory
# answer，模型自己生成最终 teaching content。这两个变量（use_teach /
# gold_state_injected）只由 condition 决定，不额外开放成独立的命令行开关，
# 避免下游 Decoder 对比实验里出现"记不清这次到底组合了哪些设置"的问题。
CONDITIONS = ("baseline", "gold_state_teach_greedy")


def condition_flags(condition: str):
    """返回 (use_teach, gold_state_injected)。"""
    if condition == "gold_state_teach_greedy":
        return True, True
    return False, False


def build_gold_state_block(gold_diagnosis, gold_strategy) -> str:
    """最小的 Gold State 控制块，附加在官方 user_prompt 之后（不改动官方
    Prompt 主体本身）。只给出 diagnosis 和 strategy 两个变量，不解释"为什么"
    （不含 Gold reason），不给结论内容（不含 Gold content/memory answer），
    并明确告诉模型这两个状态已经给定、生成教学时必须遵循、不要重新改判。
    """
    return (
        "### PROVIDED STUDENT STATE\n"
        f"Gold Diagnosis: {gold_diagnosis}\n"
        f"Gold Teaching Strategy: {gold_strategy}\n"
        "The diagnosis and strategy above are already determined and given to you as ground truth "
        "for this student and this question. Do not re-diagnose the student or change the strategy. "
        "In your output JSON, set \"diagnosis\" and \"strategy\" to exactly these given values, and "
        "generate \"content\" (and \"memory\"/\"reason\") according to the official output format, "
        "executing the given strategy's teaching action yourself."
    )


def audit_prompt_leakage(system_prompt: str, user_prompt: str, test_obj: dict) -> dict:
    """检查 Gold reason / Gold content / Gold memory answers 的完整文本是否
    原样出现在了最终 Prompt（system_prompt + user_prompt）里。只用于 dry-run
    阶段的人工审计，不参与真实推理逻辑，不修改 Prompt。

    "Unknown" 这种占位答案被排除在外：它不包含任何真实信息，即使巧合出现在
    Prompt 别处也不构成信息泄漏，纳入检查只会制造无意义的误报。
    """
    full_prompt = f"{system_prompt}\n{user_prompt}"

    gold_reason = test_obj.get("reason")
    reason_leaked = bool(gold_reason) and gold_reason in full_prompt

    gold_content = test_obj.get("content")
    content_leaked = bool(gold_content) and gold_content in full_prompt

    leaked_answers = []
    for item in (test_obj.get("memory") or []):
        if not isinstance(item, dict):
            continue
        answer = item.get("answer")
        if answer and answer != "Unknown" and answer in full_prompt:
            leaked_answers.append(answer)

    return {
        "gold_reason_leaked": reason_leaked,
        "gold_content_leaked": content_leaked,
        "gold_memory_answers_leaked": leaked_answers,
        "any_leak": reason_leaked or content_leaked or bool(leaked_answers),
    }


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_samples(manifest_path: Path, features_path: Path, limit):
    """按 manifest 里记录的 sample_index 精确取出对应的
    history_features_lastq.jsonl 行，严格保持 manifest 的固定顺序（不是文件里
    的行序，不做任何重新排序），不使用 start_index/sample_size 的连续切片逻辑。

    manifest 的 sample_index 与 history_features_lastq.jsonl 的行号是同一个
    坐标系——这一点已经在 preexp01 的数据一致性检查里验证过：两者的前 1000
    行按位置严格一一对应。

    返回 (ordered_samples, missing_indices, manifest_dict)，其中
    ordered_samples 是 [(sample_index, feature_row, manifest_entry), ...]。
    """
    manifest = load_manifest(manifest_path)
    entries = manifest.get("samples", [])
    if limit is not None and limit > 0:
        entries = entries[:limit]

    needed_indices = {e["sample_index"] for e in entries}
    rows_by_index = {}
    for i, obj in enumerate(_iter_jsonl(features_path)):
        if i in needed_indices:
            rows_by_index[i] = obj
        if len(rows_by_index) == len(needed_indices):
            break

    ordered = []
    missing = []
    for e in entries:
        idx = e["sample_index"]
        row = rows_by_index.get(idx)
        if row is None:
            missing.append(idx)
            continue
        ordered.append((idx, row, e))
    return ordered, missing, manifest


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
    ap.add_argument("--history-length", type=int, default=100,
                     help="喂给 prompt 的 history_info 只保留最近 N 条记录（默认 100，"
                          "对应论文 Appendix H 的标准协议 L=100；官方发布的 "
                          "history_features_lastq.jsonl 里 history_info 实际是全量 199 条，"
                          "不做任何截断）。保持原始时间顺序，不改动每条记录的内容，不做 token "
                          "级截断，只是从列表末尾（最近）取 N 条。设为 <=0 或 >= 实际条数则不截断，"
                          "使用全部历史。")
    ap.add_argument("--sdpa-backend", type=str, choices=["auto", "efficient"], default="auto",
                     help="auto：不干预，使用 PyTorch SDPA 默认调度；"
                          "efficient：在 model.generate() 外层用 torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) "
                          "强制只允许 memory-efficient attention backend，其他 backend 不可用时会明确报错，不会静默退回 math backend")
    ap.add_argument("--manifest", type=str, default=None,
                     help="固定样本清单 JSON 路径（如 preexperiment_manifest_40.json）。提供后按"
                          "manifest['samples'] 里记录的 sample_index 精确选取样本、严格保持 manifest 的"
                          "固定顺序，不再使用 --start-index/--sample-size 的连续切片逻辑。不提供本参数时，"
                          "现有行为（--start-index/--sample-size）完全不变。")
    ap.add_argument("--manifest-limit", type=int, default=None,
                     help="只取 manifest 里前 N 条（用于先做 smoke test），默认不限制、取全部")
    ap.add_argument("--condition", type=str, choices=list(CONDITIONS), default="baseline",
                     help="baseline（默认）：与原有行为完全一致，不启用官方 Teaching Guidelines、不注入"
                          "Gold 状态。gold_state_teach_greedy：固定配方 = 最近 history_length 条历史 + "
                          "当前错题 + Gold Diagnosis + Gold Teaching Strategy + 官方 Teaching Guidelines"
                          "（use_teach=True）+ 贪心解码；不提供 Gold reason/content/memory answer，模型自己"
                          "生成最终 teaching content。")
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
    arch_info: dict, sdpa_backend: str, logger: logging.Logger, history_length: int, condition: str,
) -> dict:
    stage_tracker["stage"] = "build_prompt"
    mem_queries = official_extract_mem_queries(test_obj)
    use_teach, gold_state_injected = condition_flags(condition)
    gold_diagnosis = test_obj.get("diagnosis")
    gold_strategy = test_obj.get("strategy")

    limited_sample, original_history_records, used_history_records = limit_history(sample, history_length)
    # 也存进 stage_tracker：如果后面 generate 阶段异常（例如 OOM），main() 的
    # except 分支拿不到这个函数的返回值，但可以从 stage_tracker 里补上这两个数。
    stage_tracker["history_counts"] = (original_history_records, used_history_records)
    logger.info(
        f"[history] uid={sample.get('uid')} original_history_records={original_history_records} "
        f"used_history_records={used_history_records} history_length_arg={history_length} "
        f"condition={condition} use_teach={use_teach} gold_state_injected={gold_state_injected}"
    )

    # 官方 process_one_sample 默认调用是 history_mode="long", use_feature=False,
    # use_teach=False；这里只把 use_teach 换成按 condition 决定的值，官方
    # Prompt 构造逻辑本身（_build_prompts 函数体）完全不改。
    system_prompt, user_prompt = official_build_prompts(limited_sample, test_obj, use_teach=use_teach)

    if gold_state_injected:
        # 只在官方 user_prompt 之后追加一个最小的 Gold State 控制块，不改动
        # 官方 Prompt 主体；不注入 Gold reason/content/memory answer。
        user_prompt = user_prompt + "\n\n" + build_gold_state_block(gold_diagnosis, gold_strategy)

    messages = build_messages(system_prompt=system_prompt, user_prompt=user_prompt)

    raw_text, in_tok, out_tok, elapsed = generate_local(
        model, tokenizer, messages, max_new_tokens, stage_tracker, arch_info, sdpa_backend, logger
    )

    stage_tracker["stage"] = "parse_output"
    parse_success = False
    parsed_output = None
    memory = diagnosis = reason = strategy = content = None

    # 两步解析：先官方 extract_json（不改动 raw_text），只有官方解析失败时
    # 才在原始文本上做 LaTeX 反斜杠 fallback repair 再解析一次。
    parsed, parse_error, parse_repair_applied, parse_repair_type = parse_model_json_output(raw_text)

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

    diagnosis_follow_gold = (diagnosis == gold_diagnosis) if parse_success else None
    strategy_follow_gold = (strategy == gold_strategy) if parse_success else None

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
        "parse_repair_applied": parse_repair_applied,
        "parse_repair_type": parse_repair_type,
        "failed_stage": None,
        "input_token_count": in_tok,
        "output_token_count": out_tok,
        "elapsed_seconds": round(elapsed, 3),
        "diagnostics": stage_tracker.get("diagnostics"),
        "original_history_records": original_history_records,
        "used_history_records": used_history_records,
        "condition": condition,
        "gold_diagnosis": gold_diagnosis,
        "gold_strategy": gold_strategy,
        "diagnosis_follow_gold": diagnosis_follow_gold,
        "strategy_follow_gold": strategy_follow_gold,
        "use_teach": use_teach,
        "gold_state_injected": gold_state_injected,
        "decoding_method": "greedy",
    }


def run_dry_check(samples, tests_map, history_length: int, condition: str, output_dir: Path, logger: logging.Logger) -> int:
    """不加载模型的静态检查：数据读取 / sample key 对齐 / prompt 构造是否正常。

    与正式推理路径（process_one_sample_local）共用同一个 limit_history /
    official_build_prompts / build_gold_state_block 逻辑，所以这里打印出的
    original_history_records / used_history_records / user_prompt_chars 以及
    Gold State 注入结果，反映的是真实会喂给模型的那份 prompt。当 condition 是
    gold_state_teach_greedy 时，额外做一次 Prompt 审计：逐项确认最近
    history_length 条历史存在、当前题存在、Gold Diagnosis/Strategy 存在、官方
    Teaching Guidelines 存在、以及自动 leakage audit（Gold reason/content/
    memory answers 的完整文本不应出现在最终 Prompt 里）。全程不 import
    torch/transformers，不下载 tokenizer，不调用任何模型。
    """
    use_teach, gold_state_injected = condition_flags(condition)
    n_ok = 0
    n_leak = 0
    for idx, sample in samples:
        key = official_sample_key(sample)
        test_obj = tests_map.get(key)
        if test_obj is None:
            logger.error(f"[dry-run] index={idx} uid={sample.get('uid')} 未在 human_an_updated.jsonl 中找到对应 key，跳过")
            continue
        mem_queries = official_extract_mem_queries(test_obj)
        gold_diagnosis = test_obj.get("diagnosis")
        gold_strategy = test_obj.get("strategy")

        limited_sample, original_history_records, used_history_records = limit_history(sample, history_length)
        system_prompt, user_prompt = official_build_prompts(limited_sample, test_obj, use_teach=use_teach)

        if gold_state_injected:
            user_prompt = user_prompt + "\n\n" + build_gold_state_block(gold_diagnosis, gold_strategy)

        logger.info(
            f"[dry-run] index={idx} uid={sample.get('uid')} sample_key={key} condition={condition} "
            f"original_history_records={original_history_records} used_history_records={used_history_records} "
            f"history_length_arg={history_length} mem_queries={len(mem_queries)} "
            f"system_prompt_chars={len(system_prompt)} user_prompt_chars={len(user_prompt)}"
        )

        # 把完整最终 Prompt（system + user，含 Gold State 控制块）落盘，供人工逐项
        # 审阅"最近 N 条历史 / 当前题 / Gold Diagnosis / Gold Strategy / 官方
        # Teaching Guidelines 是否存在，Gold reason/content/memory answer 是否
        # 不存在"，而不是只看字符数这种间接信号。
        prompts_dir = output_dir / "dry_run_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompts_dir / f"index_{idx}.txt"
        with prompt_file.open("w", encoding="utf-8") as pf:
            pf.write("===== SYSTEM PROMPT =====\n")
            pf.write(system_prompt)
            pf.write("\n\n===== USER PROMPT =====\n")
            pf.write(user_prompt)
        logger.info(f"[dry-run] index={idx} 完整 Prompt 已写入: {prompt_file}")

        # ---- Prompt 审计：逐项确认 + 自动 leakage audit ----
        current_question_present = bool(sample.get("question_info")) and sample["question_info"] in user_prompt
        teaching_guidelines_present = "[Teaching Guidelines]" in system_prompt
        gold_diagnosis_block_present = gold_state_injected and (f"Gold Diagnosis: {gold_diagnosis}" in user_prompt)
        gold_strategy_block_present = gold_state_injected and (f"Gold Teaching Strategy: {gold_strategy}" in user_prompt)
        leakage = audit_prompt_leakage(system_prompt, user_prompt, test_obj)

        logger.info(
            f"[audit] index={idx} history_present={used_history_records > 0} "
            f"current_question_present={current_question_present} "
            f"teaching_guidelines_present={teaching_guidelines_present} "
            f"gold_diagnosis_block_present={gold_diagnosis_block_present} "
            f"gold_strategy_block_present={gold_strategy_block_present} "
            f"gold_reason_leaked={leakage['gold_reason_leaked']} "
            f"gold_content_leaked={leakage['gold_content_leaked']} "
            f"gold_memory_answers_leaked={leakage['gold_memory_answers_leaked']}"
        )
        if leakage["any_leak"]:
            n_leak += 1
            logger.error(f"[audit] index={idx} 检测到 Gold 信息泄漏到 Prompt 里！{leakage}")

        n_ok += 1

    logger.info(f"[dry-run] 完成，{n_ok}/{len(samples)} 条样本通过 key 对齐与 prompt 构造检查（未调用任何模型）")
    logger.info(f"[audit] leakage audit：{n_leak}/{n_ok} 条样本检测到 Gold 信息泄漏")
    if n_ok != len(samples):
        return 1
    return 1 if n_leak > 0 else 0


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

    manifest_meta = None
    manifest_indices_by_index = {}
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            logger.error(f"未找到 manifest 文件: {manifest_path}")
            return 2
        manifest_samples, missing_indices, manifest_meta = load_manifest_samples(
            manifest_path, FEATURES_PATH, args.manifest_limit
        )
        if missing_indices:
            logger.error(
                f"manifest 中有 {len(missing_indices)} 个 sample_index 在 "
                f"{FEATURES_PATH.relative_to(REPO_ROOT)} 里找不到对应行: {missing_indices}"
            )
            return 2
        samples = [(idx, row) for idx, row, _entry in manifest_samples]
        manifest_indices_by_index = {idx: entry for idx, _row, entry in manifest_samples}
        logger.info(
            f"从 manifest={manifest_path} 精确选取 {len(samples)} 条样本"
            f"（manifest 总条数={len(manifest_meta.get('samples', []))}，"
            f"manifest-limit={args.manifest_limit}），保持 manifest 固定顺序，不使用连续切片逻辑"
        )
    else:
        samples = _load_slice(FEATURES_PATH, args.start_index, args.sample_size)
        logger.info(f"读取到 {len(samples)} 条待处理样本（start_index={args.start_index}, sample_size={args.sample_size}）")

    tests_map = official_load_tests_map(GOLD_PATH)
    logger.info(f"Gold 测试用例共 {len(tests_map)} 条；本次 condition={args.condition}")

    if args.dry_run:
        return run_dry_check(samples, tests_map, args.history_length, args.condition, output_dir, logger)

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

            manifest_entry = manifest_indices_by_index.get(idx)
            if manifest_entry is not None and (
                manifest_entry.get("gold_diagnosis") != test_obj.get("diagnosis")
                or manifest_entry.get("gold_strategy") != test_obj.get("strategy")
            ):
                logger.error(
                    f"index={idx} uid={uid}: manifest 里记录的 gold_diagnosis/gold_strategy 与当前 Gold "
                    f"文件里的不一致（manifest={manifest_entry.get('gold_diagnosis')}/"
                    f"{manifest_entry.get('gold_strategy')}，当前 Gold="
                    f"{test_obj.get('diagnosis')}/{test_obj.get('strategy')}），Gold 数据似乎变了，请核实"
                )

            stage_tracker = {"stage": "init", "diagnostics": None, "history_counts": (None, None)}
            manifest_sample_index = idx if args.manifest else None
            try:
                result = process_one_sample_local(
                    model, tokenizer, sample, test_obj, args.max_new_tokens, stage_tracker,
                    arch_info, args.sdpa_backend, logger, args.history_length, args.condition,
                )
                row = {
                    "sample_index": idx,
                    "sample_key": key,
                    "uid": uid,
                    "manifest_sample_index": manifest_sample_index,
                    **result,
                }
                success = bool(result["parse_success"])
            except Exception as e:
                tb = traceback.format_exc()
                failed_stage = stage_tracker.get("stage", "unknown")
                orig_hist, used_hist = stage_tracker.get("history_counts", (None, None))
                use_teach, gold_state_injected = condition_flags(args.condition)
                row = {
                    "sample_index": idx,
                    "sample_key": key,
                    "uid": uid,
                    "manifest_sample_index": manifest_sample_index,
                    "raw_output": None,
                    "parsed_output": None,
                    "memory": None,
                    "diagnosis": None,
                    "reason": None,
                    "strategy": None,
                    "content": None,
                    "parse_success": False,
                    "parse_error": f"{type(e).__name__}: {e}",
                    "parse_repair_applied": False,
                    "parse_repair_type": None,
                    "failed_stage": failed_stage,
                    "input_token_count": None,
                    "output_token_count": None,
                    "elapsed_seconds": None,
                    # 即使在 generate 阶段 OOM，move_inputs 阶段已经算好的诊断信息
                    # 也会保留在这里，不会因为异常而丢失。
                    "diagnostics": stage_tracker.get("diagnostics"),
                    # history 截取发生在 build_prompt 阶段，比 generate 更早，所以
                    # 即使 generate 阶段 OOM，这两个数也几乎总是已经算出来了。
                    "original_history_records": orig_hist,
                    "used_history_records": used_hist,
                    # test_obj 在异常发生前就已经取到，Gold 相关字段依然可以照常记录，
                    # 不会因为 generate/parse 阶段的异常而丢失。
                    "condition": args.condition,
                    "gold_diagnosis": test_obj.get("diagnosis"),
                    "gold_strategy": test_obj.get("strategy"),
                    "diagnosis_follow_gold": None,
                    "strategy_follow_gold": None,
                    "use_teach": use_teach,
                    "gold_state_injected": gold_state_injected,
                    "decoding_method": "greedy",
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
        # 本次实验实际请求"最近 N 条历史"的 N。history_length_requested 是原始 CLI
        # 输入值；history_length 是本次实际生效的值——目前两者总是相同（脚本没有
        # 任何会二次覆盖这个数的逻辑），分别记录是为了以后如果加了自动降级之类的
        # 逻辑，也不会丢掉"用户原始请求的是多少"这个信息。
        "history_length_requested": args.history_length,
        "history_length": args.history_length,
        "condition": args.condition,
        "use_teach": condition_flags(args.condition)[0],
        "gold_state_injected": condition_flags(args.condition)[1],
        "decoding_method": "greedy",
        "manifest_path": str(Path(args.manifest).resolve()) if args.manifest else None,
        "manifest_total_count": len(manifest_meta.get("samples", [])) if manifest_meta else None,
        "manifest_limit": args.manifest_limit,
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
