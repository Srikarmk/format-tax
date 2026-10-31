"""Deterministic answer checking.

Free-text replies use prompt-brittleness's parsers ("Answer:" cue, then fallbacks). JSON replies are parsed
two ways: strictly (the whole reply is one JSON object) and leniently (the first JSON object anywhere in
the reply, so code fences and a sentence around it are tolerated). Accuracy uses the lenient parse, as a
production system usually would; the strict rate is reported as schema validity. A reply with no usable
JSON answer counts as wrong; `rescued` records what a free-text parser would have found in it anyway, and
`written_correct` whether the value written after "answer": was right even if the JSON never closed.
"""

from __future__ import annotations

import json
import re

from .data import FIELDS, norm

NUM = r"-?\d[\d,]*(?:\.\d+)?"
CUE = r"(?i:final answer|answer|a)"
NULLS = {"", "null", "none", "n a", "na", "unknown", "not given", "not specified", "not provided"}


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("**", "").replace("\\(", "").replace("\\)", "")


def canon_number(s) -> str | None:
    s = str(s).replace(",", "").replace("$", "").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    if x == int(x):
        return str(int(x))
    return f"{x:.6g}"


def parse_gsm8k(text: str) -> tuple[str | None, str]:
    t = _clean(text)
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", t)
    cues = list(re.finditer(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*(.*)$", t))
    if cues:
        tail = cues[-1].group(1)
        m = re.search(NUM, tail.replace("$", ""))
        if m:
            return canon_number(m.group(0)), "cue"
    if boxed:
        m = re.search(NUM, boxed[-1])
        if m:
            return canon_number(m.group(0)), "boxed"
    nums = re.findall(NUM, t.replace("$", ""))
    if nums:
        return canon_number(nums[-1]), "last number"
    return None, "none"


def _letter_from_text(rest: str, options: list[str] | None) -> str | None:
    if not options:
        return None
    rest = norm(rest)
    hits = [(len(norm(o)), i) for i, o in enumerate(options) if norm(o) and rest.startswith(norm(o))]
    return "ABCD"[max(hits)[1]] if hits else None


def parse_mmlu(text: str, options: list[str] | None = None) -> tuple[str | None, str]:
    t = _clean(text).strip()
    cues = re.findall(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*\(?([A-D])\b", t)
    if cues:
        return cues[-1], "cue"
    m = re.search(r"\b(?i:answer|option|choice) is\s*:?\s*\(?([A-D])\b", t)
    if m:
        return m.group(1), "answer is"
    m = re.fullmatch(r"\(?([A-D])\)?[.:)]?", t)
    if m:
        return m.group(1), "bare letter"
    m = re.match(r"\(?([A-D])[.):]\s", t + " ")
    if m:
        return m.group(1), "leading letter"
    m = re.search(r"\b(?i:answer)\s*[:：]?\s*\(?([A-D])\b", t)
    if m:
        return m.group(1), "inline cue"
    return None, "none"


def _json_objects(t: str) -> list[str]:
    """Top-level {...} spans, skipping braces inside strings."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(t[start:i + 1])
    return out


def parse_json(text: str) -> tuple[dict | None, bool]:
    """(object, strict). `strict` is True when the whole reply is exactly one JSON object."""
    t = _clean(text).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    for blob in _json_objects(t):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, False
    return None, False


def json_answer(task: str, obj: dict | None, options: list[str] | None) -> str | None:
    if not obj or "answer" not in obj or obj["answer"] is None:
        return None
    a = obj["answer"]
    if task == "gsm8k":
        if isinstance(a, bool):
            return None
        m = re.search(NUM, str(a).replace("$", ""))
        return canon_number(m.group(0)) if m else None
    s = str(a).strip()
    m = re.fullmatch(r"\(?([A-Da-d])\)?[.:)]?", s)
    if m:
        return m.group(1).upper()
    m = re.match(r"\(?([A-D])[.):]\s", s + " ")
    if m:
        return m.group(1)
    return _letter_from_text(s, options)


def check_a(task: str, cond: str, text: str, gold: str, options: list[str] | None = None) -> dict:
    """Part A. `cond` is the condition with any A/A prefix removed."""
    free = parse_gsm8k if task == "gsm8k" else (lambda t: parse_mmlu(t, options))
    g = canon_number(gold) if task == "gsm8k" else gold
    if cond.startswith("free_"):
        a, how = free(text)
        return {"answer": a, "correct": a is not None and a == g, "method": how, "json_strict": None,
                "json_found": None, "rescued": None, "written_correct": None}
    obj, strict = parse_json(text)
    a = json_answer(task, obj, options)
    # the value written after "answer": in the raw reply, even when the JSON around it is broken
    m = re.search(r'"answer"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\s]+)', _clean(text))
    written = None
    if m:
        try:
            written = json_answer(task, {"answer": json.loads(m.group(1))}, options)
        except json.JSONDecodeError:
            written = json_answer(task, {"answer": m.group(1)}, options)
    rescued = None
    if a is None:
        r, _ = free(text)
        rescued = r is not None and r == g
    return {"answer": a, "correct": a is not None and a == g,
            "method": ("strict json" if strict else "json in text") if a is not None else "none",
            "json_strict": strict and a is not None, "json_found": obj is not None, "rescued": rescued,
            "written_correct": written is not None and written == g}


# ---------------------------------------------------------------- Part B

def _is_null(x) -> bool:
    return x is None or (isinstance(x, str) and norm(x) in NULLS)


def _as_bool(x):
    if isinstance(x, bool):
        return x
    s = norm(x)
    if s in ("yes", "true", "y"):
        return True
    if s in ("no", "false", "n"):
        return False
    return "other"


def field_ok(field: str, pred, gold) -> bool:
    if gold is None:
        return _is_null(pred)
    if _is_null(pred):
        return False
    if field == "family_friendly":
        return _as_bool(pred) == gold
    return norm(pred) == norm(gold)


def check_drift(text: str, gold: dict) -> dict:
    obj, strict = parse_json(text)
    if obj is None:
        return {"parsed": False, "json_strict": False, "keys_exact": False, "fields": {f: False for f in FIELDS},
                "filled": {f: False for f in FIELDS}, "answer": None}
    fields = {f: field_ok(f, obj.get(f), gold[f]) for f in FIELDS}
    # a value given for a field the record doesn't have
    filled = {f: gold[f] is None and not _is_null(obj.get(f)) for f in FIELDS}
    answer = json.dumps({f: (None if _is_null(obj.get(f)) else
                             (_as_bool(obj.get(f)) if f == "family_friendly" else norm(obj.get(f))))
                         for f in FIELDS}, sort_keys=True)
    return {"parsed": True, "json_strict": strict, "keys_exact": set(obj) == set(FIELDS), "fields": fields,
            "filled": filled, "answer": answer}
