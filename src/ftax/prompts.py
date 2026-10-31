"""Prompt templates and output schemas.

Part A (format tax), GSM8K and MMLU. Eight conditions cross three things:
  - how the answer is written: free text ("Answer: ...") or a JSON object;
  - whether there is room to reason, and where: reasoning first, answer first, or no reasoning;
  - for JSON, whether decoding is constrained to a schema (Ollama `format`) or only asked for in the prompt.
The schema conditions use the same prompt text as their prompt-only twins, so the two differ only in the
constraint. Two A/A repeats (free_cot and schema_cot sent again) give the noise floor.

Part B (schema drift), E2E records. The same record is shown in nine source formats, each changing one
thing about the base JSON export, and converted to one fixed target schema, either prompt-only or with
schema-constrained decoding.

Every template is written out in full so it can be read and checked by eye.
"""

from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass

from . import config as C
from .data import AREAS, EAT_TYPES, FIELDS, PRICES, RATINGS, TARGET

# ---------------------------------------------------------------- Part A: format tax

CONDITIONS = ["free_cot", "free_direct", "json_cot", "json_answer_first", "json_direct",
              "schema_cot", "schema_answer_first", "schema_direct"]
AA = {"aa_free_cot": "free_cot", "aa_schema_cot": "schema_cot"}
A_VARIANTS = CONDITIONS + list(AA)
LABEL = {
    "free_cot": "free text, reasoning first",
    "free_direct": "free text, answer only",
    "json_cot": "JSON (prompt), reasoning first",
    "json_answer_first": "JSON (prompt), answer first",
    "json_direct": "JSON (prompt), answer only",
    "schema_cot": "JSON (schema), reasoning first",
    "schema_answer_first": "JSON (schema), answer first",
    "schema_direct": "JSON (schema), answer only",
    "aa_free_cot": "free text, reasoning first (A/A)",
    "aa_schema_cot": "JSON (schema), reasoning first (A/A)",
}
# which JSON fields each condition asks for, in order
JSON_KEYS = {"cot": ["reasoning", "answer"], "answer_first": ["answer", "reasoning"], "direct": ["answer"]}

G_T = "Solve the following math problem."
GSM8K = {
    "free_cot": f'{G_T} Reason step by step, then write the final answer on the last line as "Answer: <number>".',
    "free_direct": f'{G_T} Reply with only the final answer, in the form "Answer: <number>". Do not show your working.',
    "cot": f'{G_T} Reply with only a JSON object with two keys, in this order: "reasoning", your step-by-step '
           'working as a string, then "answer", the final answer as a number.',
    "answer_first": f'{G_T} Reply with only a JSON object with two keys, in this order: "answer", the final '
                    'answer as a number, then "reasoning", your step-by-step working as a string.',
    "direct": f'{G_T} Reply with only a JSON object with one key, "answer", the final answer as a number. '
              "Do not show your working.",
}
M_T = "Answer the following multiple-choice question about {subject}."
MMLU = {
    "free_cot": f'{M_T} Reason step by step, then write the letter of the correct option on the last line as '
                '"Answer: <letter>".',
    "free_direct": f'{M_T} Reply with only the letter of the correct option, in the form "Answer: <letter>". '
                   "Do not explain.",
    "cot": f'{M_T} Reply with only a JSON object with two keys, in this order: "reasoning", your step-by-step '
           'reasoning as a string, then "answer", the letter of the correct option.',
    "answer_first": f'{M_T} Reply with only a JSON object with two keys, in this order: "answer", the letter '
                    'of the correct option, then "reasoning", your step-by-step reasoning as a string.',
    "direct": f'{M_T} Reply with only a JSON object with one key, "answer", the letter of the correct option. '
              "Do not explain.",
}


def answer_schema(task: str, keys: list[str]) -> dict:
    """JSON schema for a Part A reply. Property order follows `keys`; Ollama generates in that order."""
    ans = {"type": "number"} if task == "gsm8k" else {"type": "string", "enum": list("ABCD")}
    props = {k: (ans if k == "answer" else {"type": "string"}) for k in keys}
    return {"type": "object", "properties": props, "required": keys}


def mmlu_body(it: dict) -> str:
    opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", it["options"]))
    return f"Question: {it['question']}\n\n{opts}"


# ---------------------------------------------------------------- Part B: schema drift

DRIFT_VARIANTS = ["base", "aa", "renamed", "reordered", "reformatted", "distractors", "dropped", "csv", "nested"]
DRIFT_LABEL = {
    "base": "base JSON export",
    "aa": "base again (A/A)",
    "renamed": "keys renamed",
    "reordered": "keys reordered",
    "reformatted": "values reformatted",
    "distractors": "extra look-alike fields",
    "dropped": "one field missing",
    "csv": "CSV instead of JSON",
    "nested": "nested JSON",
}
MODES = ["json", "schema"]

RENAMED = {"name": "venue_name", "eatType": "venue_type", "food": "cuisine", "priceRange": "price_band",
           "customer rating": "rating", "area": "location", "familyFriendly": "kids_welcome", "near": "landmark"}
REFORMAT = {
    "priceRange": {"less than £20": "under 20 GBP", "£20-25": "20-25 GBP", "more than £30": "over 30 GBP"},
    "customer rating": {"1 out of 5": "1/5", "3 out of 5": "3/5", "5 out of 5": "5/5"},
    "familyFriendly": {"yes": "Y", "no": "N"},
    "eatType": {"coffee shop": "coffee_shop"},
}
NESTED = {"venue": ["name", "eatType"], "menu": ["food", "priceRange"], "reviews": ["customer rating"],
          "location": ["area", "near"], "policies": ["familyFriendly"]}
# look-alike extra fields: another venue's name, another landmark, and a date
OTHER_NAMES = ["The Golden Curry", "The Wrestlers", "Aromi", "Zizzi", "The Cricketers", "Loch Fyne"]
OTHER_LANDMARKS = ["The Bakers", "Burger King", "The Sorrento", "Café Brazil", "Ranch", "The Portland Arms"]

DRIFT_PROMPT = (
    "Convert the restaurant record below into a JSON object with exactly these keys:\n"
    "- name: the venue's name\n"
    '- eat_type: one of "restaurant", "pub", "coffee shop"\n'
    "- food: the type of food served\n"
    '- price_range: one of "cheap", "moderate", "high", "less than £20", "£20-25", "more than £30"\n'
    '- customer_rating: one of "low", "average", "high", "1 out of 5", "3 out of 5", "5 out of 5"\n'
    '- area: one of "riverside", "city centre"\n'
    "- family_friendly: true or false\n"
    "- near: a nearby landmark\n"
    "Use null for any field the record does not give. Reply with only the JSON object.\n\n"
    "Record:\n{record}"
)


def _enum_or_null(values: list[str]) -> dict:
    return {"anyOf": [{"type": "string", "enum": values}, {"type": "null"}]}


DRIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": ["string", "null"]},
        "eat_type": _enum_or_null(EAT_TYPES),
        "food": {"type": ["string", "null"]},
        "price_range": _enum_or_null(PRICES),
        "customer_rating": _enum_or_null(RATINGS),
        "area": _enum_or_null(AREAS),
        "family_friendly": {"type": ["boolean", "null"]},
        "near": {"type": ["string", "null"]},
    },
    "required": FIELDS,
}


def _rng(it: dict, salt: str) -> random.Random:
    return random.Random(f"{C.SEED}-{it['id']}-{salt}")


def dropped_key(it: dict) -> str:
    """The field removed in the `dropped` variant: any present field except the name, chosen per record."""
    keys = [k for k in it["record"] if k != "name"]
    return _rng(it, "drop").choice(keys)


def source_record(it: dict, v: str) -> tuple[str, dict]:
    """The record as shown to the model under variant `v`, and the gold target for it."""
    rec, gold = dict(it["record"]), dict(it["gold"])
    if v in ("base", "aa"):
        return json.dumps(rec, indent=2, ensure_ascii=False), gold
    if v == "renamed":
        return json.dumps({RENAMED[k]: x for k, x in rec.items()}, indent=2, ensure_ascii=False), gold
    if v == "reordered":
        keys = list(rec)
        r = _rng(it, "order")
        while True:
            r.shuffle(keys)
            if keys != list(rec):
                break
        return json.dumps({k: rec[k] for k in keys}, indent=2, ensure_ascii=False), gold
    if v == "reformatted":
        return json.dumps({k: REFORMAT.get(k, {}).get(x, x) for k, x in rec.items()}, indent=2,
                          ensure_ascii=False), gold
    if v == "distractors":
        r = _rng(it, "distract")
        out = dict(rec)
        out["previous_name"] = r.choice([n for n in OTHER_NAMES if n != rec["name"]])
        out["opposite"] = r.choice([n for n in OTHER_LANDMARKS if n != rec.get("near")])
        out["last_inspected"] = f"2025-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
        return json.dumps(out, indent=2, ensure_ascii=False), gold
    if v == "dropped":
        k = dropped_key(it)
        del rec[k]
        gold[TARGET[k]] = None
        return json.dumps(rec, indent=2, ensure_ascii=False), gold
    if v == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(list(rec))
        w.writerow(list(rec.values()))
        return buf.getvalue().strip(), gold
    if v == "nested":
        out = {}
        for group, keys in NESTED.items():
            sub = {k: rec[k] for k in keys if k in rec}
            if sub:
                out[group] = sub
        return json.dumps(out, indent=2, ensure_ascii=False), gold
    raise KeyError(v)


# ---------------------------------------------------------------- rendering

@dataclass
class Rendered:
    messages: list[dict]
    fmt: dict | None
    gold: object
    tag: str = ""


def render(it: dict, v: str, mode: str = "json") -> Rendered:
    """`v` is a Part A condition or a drift variant; `mode` applies to drift only."""
    if it["task"] == "drift":
        text, gold = source_record(it, v)
        return Rendered([{"role": "user", "content": DRIFT_PROMPT.format(record=text)}],
                        DRIFT_SCHEMA if mode == "schema" else None, gold, tag="aa" if v == "aa" else "")
    tag = "aa" if v in AA else ""
    cond = AA.get(v, v)
    table = GSM8K if it["task"] == "gsm8k" else MMLU
    shape = cond.split("_", 1)[1] if cond.startswith(("json_", "schema_")) else None
    instr = table[cond] if shape is None else table[shape]
    if it["task"] == "gsm8k":
        user = f"{instr}\n\nProblem: {it['question']}"
    else:
        user = instr.format(subject=it["subject"].replace("high_school_", "high school ").replace("_", " ")) \
            + "\n\n" + mmlu_body(it)
    fmt = answer_schema(it["task"], JSON_KEYS[shape]) if cond.startswith("schema_") else None
    return Rendered([{"role": "user", "content": user}], fmt, it["gold"], tag)


def variants(task: str) -> list[str]:
    return DRIFT_VARIANTS if task == "drift" else A_VARIANTS
