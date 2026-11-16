# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audited Flow-Factory/TextPecker sentence-profile helpers (Apache-2.0).

Only helpers used by the sentence/substring recipe are retained. Algorithms and
OCR prompt are unchanged; runtime transport/model lifecycle live in the scorer.
Reference: Flow-Factory 6d5b4710f7f1d504fe8f09c6e506ba9f21644e30.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


def parse_json_from_answer(answer_raw: str):
    """Extract & parse the JSON object from a raw model answer, tolerating
    ```json fences, surrounding quotes, and trailing notes.

    Returns: (parsed_dict, list_of_keys). Raises AssertionError on failure.
    """
    json_str = None

    # Prefer ```json ... ``` fenced blocks.
    json_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_block_pattern.search(answer_raw)
    if match:
        json_str = match.group(1).strip()

    # Otherwise strip a single layer of wrapping single/double quotes.
    if not json_str:
        processed = answer_raw.strip()
        if (processed.startswith("'") and processed.endswith("'")) or (
            processed.startswith('"') and processed.endswith('"')
        ):
            processed = processed[1:-1].strip()
        json_str = processed

    if not json_str:
        assert False, "未提取到任何可能的JSON内容"

    # Truncate any trailing text after the JSON object/array.
    if "}" in json_str:
        last_brace_idx = json_str.rfind("}")
        json_str = json_str[: last_brace_idx + 1]
    elif "]" in json_str:
        last_bracket_idx = json_str.rfind("]")
        json_str = json_str[: last_bracket_idx + 1]

    first_e = None
    try:
        answer_dict = json.loads(json_str)
        return answer_dict, list(answer_dict.keys())
    except json.JSONDecodeError as e:
        first_e = e

    # Escape literal newlines inside quoted strings, then retry.
    def replace_newlines(match):
        inner = match.group(1)
        return inner.replace("\n", "\\n").replace("\r", "\\r")

    cleaned_str = re.sub(r'"([^"]*)"', lambda m: f'"{replace_newlines(m)}"', json_str)

    try:
        answer_dict = json.loads(cleaned_str)
        return answer_dict, list(answer_dict.keys())
    except json.JSONDecodeError as second_e:
        error_msg = [
            f"JSON解析失败（原始内容前50字符: {json_str[:50]}...）",
            f"第一次错误: {str(first_e)}" if first_e else "",
            f"清理后内容前50字符: {cleaned_str[:50]}...",
            f"第二次错误: {str(second_e)}",
        ]
        assert False, "\n".join(filter(None, error_msg))


def extract_rec_texts(s: str):
    """Best-effort regex extraction of the ``recognized_text`` field when full
    JSON parsing fails."""
    rec_text = None
    try:
        rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)",\s*"duplicate_text"', s, re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)

        if not rec_text:
            rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)"(?:,\s*"[^\"]*"|\s*})', s, re.DOTALL)
            if rec_match:
                rec_text = rec_match.group(1)

        if not rec_text:
            rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)"', s, re.DOTALL)
            if rec_match:
                rec_text = rec_match.group(1)

        def clean_text(text):
            if text is None:
                return None
            try:
                text = text.replace("\\n", "\n")
                text = text.replace("\\t", "\t")
                text = text.replace("\\r", "\r")
                text = text.replace("\\\\", "\\")
                text = text.replace("\\'", "'")
                text = text.replace('\\"', '"')
                text = re.sub(r"<[^>]*>", "", text)
                return text
            except Exception:
                return text

        rec_text = clean_text(rec_text)
    except Exception:
        pass

    return rec_text


def process_raw_text(raw_text) -> str:
    """Normalize the recognized_text: restore JSON escapes, treat quotes as
    content boundaries, and collapse punctuation/whitespace into single
    spaces."""
    if not raw_text:
        return ""

    if isinstance(raw_text, List):
        pro_text = " ".join(raw_text)
    elif isinstance(raw_text, str):
        pro_text = raw_text
    else:
        return ""

    pro_text = pro_text.replace("\\\\", "\\")
    pro_text = pro_text.replace("\\n", "\n")
    pro_text = pro_text.replace('\\"', '"')
    pro_text = pro_text.replace("\\'", "'")

    quote_chars = ['"', "“", "‘", "'", "〝"]
    quote_count = sum(pro_text.count(char) for char in quote_chars)

    if quote_count >= 2:
        result = []
        in_quote = False
        current_quote = None
        close_quote_mapping = {'"': '"', "“": "”", "‘": "’", "'": "'", "〝": "〞"}

        for char in pro_text:
            if char in close_quote_mapping.keys():
                if not in_quote:
                    in_quote = True
                    current_quote = char
                elif char == close_quote_mapping.get(current_quote):
                    in_quote = False
                    current_quote = None
                else:
                    result.append(char)
            elif char in ["，", "。", "！", "？", ",", " "] and not in_quote:
                result.append(" ")
            else:
                result.append(char)

        pro_text = " ".join("".join(result).split())
    else:
        pro_text = re.sub(r"[，。！？,]+", " ", pro_text)
        pro_text = " ".join(pro_text.split())

    return pro_text


def get_rec_template() -> str:
    """Whole-image recognition prompt (image-level)."""
    return """
This is a text-generated image. Please recognize all visible text in the entire image.
Marking rules:
1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
Output in the following JSON format:
{
"recognized_text": "All text in the image (including structural error markers)"
}
"""


def tokenize_text(text: str) -> List[str]:
    """
    分词函数 - 为图像模型输出设计

    策略：保留所有非空白字符，不丢弃任何信息（包括可能的乱码）
    这样可以：
    1. 保留完整的模型输出信息
    2. 精确捕捉识别偏差和乱码
    3. 进行更准确的编辑距离计算

    分词规则：
    - 英文单词（包含撇号如 don't）
    - 中文字符（逐字）
    - 数字序列（包含小数点）
    - 其他单个符号和字符（包括潜在乱码）

    Args:
        text: 输入文本

    Returns:
        token 列表
    """
    if not text:
        return []

    # 只保留文本类 token：英文单词 / 数字 / 中文逐字
    # 标点、符号、换行、空白等全部丢弃
    pattern = re.compile(
        r"[A-Za-z*]+(?:'[A-Za-z*]+)*"  # 英文单词
        r"|\d+(?:\.\d+)?"  # 数字
        r"|[一-鿿]"  # 中文字符（逐个）
    )

    tokens = pattern.findall(text)
    return tokens


def normalize_sentence(
    text: Optional[str],
    case_sensitive: bool = True,
    collapse_ws: bool = True,
    strip_punct: bool = True,
) -> str:
    """整句归一化：用于整句比较前的轻量清洗。

    - collapse_ws : 多个空白压成单个空格并 strip（吸收 OCR 的空格抖动）。
    - strip_punct : 去掉标点/符号(保留字母数字、空格、中日韩字符)。**默认 True**，
                    即忽略标点 —— 标点差异/错误不计入惩罚（用户要求）。词间空格保留，
                    故 "Buy One, Get" 与 "Buy One Get"、"Offer." 与 "Offer"、
                    "don't" 与 "dont" 都视为一致。
    - case_sensitive: False 时转小写。字形任务默认 True。
    """
    if text is None:
        return ""
    s = str(text)
    if strip_punct:
        # 仅保留 \w(字母数字下划线)、空白、CJK 统一表意文字
        s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)
    if collapse_ws:
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = s.strip()
    if not case_sensitive:
        s = s.lower()
    return s


def _best_substring_match(pattern: str, text: str):
    """在 text 中找与 pattern 最相似的子串（近似子串匹配 / fuzzy substring）。

    返回 (min_edit_distance, matched_substring)。与普通编辑距离唯一的区别： 首行 dp[0][j]=0（空
    pattern 可在 text 任意位置起匹配，跳过前缀免费）、答案取 min_j dp[m][j]（可在任意位置结束）。即只问"pattern
    这句话作为一段出现在 text 里 有多准"，**完全忽略 OCR 的断行/合并**。同时跟踪起始列以便回取命中子串(调试用)。
    """
    m, n = len(pattern), len(text)
    if m == 0:
        return 0, ""
    if n == 0:
        return m, ""
    prev = [0] * (n + 1)  # dp[0][j] = 0
    prev_st = list(range(n + 1))  # 起始列：空 pattern 起点即当前 j
    for i in range(1, m + 1):
        cur = [i] + [0] * n  # dp[i][0] = i
        cur_st = [0] * (n + 1)  # dp[i][0]：子串空、起点 0
        pc = pattern[i - 1]
        for j in range(1, n + 1):
            cost = 0 if pc == text[j - 1] else 1
            diag = prev[j - 1] + cost
            up = prev[j] + 1
            left = cur[j - 1] + 1
            best = min(diag, up, left)
            cur[j] = best
            if best == diag:
                cur_st[j] = prev_st[j - 1]
            elif best == up:
                cur_st[j] = prev_st[j]
            else:
                cur_st[j] = cur_st[j - 1]
        prev, prev_st = cur, cur_st
    best_j, best_d = 0, prev[0]
    for j in range(n + 1):
        if prev[j] < best_d:
            best_d, best_j = prev[j], j
    return best_d, text[prev_st[best_j] : best_j]


def match_substring(
    targets: List[str],
    pred_lines: List[str],
    *,
    case_sensitive: bool = True,
    collapse_ws: bool = True,
    strip_punct: bool = True,
) -> List[Dict]:
    """每个 target 句子在「整段 OCR 文本(所有行拼接)」里找最相似子串（推荐，默认）。

    彻底忽略 OCR 断行：无论 OCR 把一句拆成多行、还是把多句并成一行，都不影响
    ——只看每句话作为一段是否准确出现。各 target 相互独立(允许命中重叠子串)，
    重复渲染由 f_dup 另行处理。sim = 1 - 子串编辑距离 / len(target)。

    返回 per-target 列表，每项::

        {"target": 原句, "pred": 命中的子串, "sim": 子串NED相似度, "exact": 1.0 / 0.0}
    """
    pred_full = normalize_sentence(" ".join(pred_lines), case_sensitive, collapse_ws, strip_punct)
    results: List[Dict] = []
    for t in targets:
        nt = normalize_sentence(t, case_sensitive, collapse_ws, strip_punct)
        if not nt:
            results.append({"target": t, "pred": "", "sim": 1.0 if not pred_full else 0.0, "exact": 1.0})
            continue
        if not pred_full:
            results.append({"target": t, "pred": "", "sim": 0.0, "exact": 0.0})
            continue
        dist, matched = _best_substring_match(nt, pred_full)
        sim = max(1.0 - dist / len(nt), 0.0)
        exact = 1.0 if nt in pred_full else 0.0
        results.append({"target": t, "pred": matched, "sim": float(sim), "exact": float(exact)})
    return results
