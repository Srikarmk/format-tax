"""Tables for the report, computed from results/answers/*.jsonl. Every interval resamples items.

Part A contrasts are paired by item within a model and task: (accuracy under one condition) minus
(accuracy under another). Part B contrasts are paired by record: field accuracy under a source format
minus field accuracy under the base export, in the same output mode.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from . import config as C
from .data import FIELDS, TARGET, load_items
from .parse import NUM, canon_number
from .prompts import AA, CONDITIONS, DRIFT_VARIANTS, REFORMAT, dropped_key
from .stats import mean_ci, paired_ci

# Part A contrasts: (name, condition, reference)
CONTRASTS = [
    ("reasoning helps (free text)", "free_cot", "free_direct"),
    ("JSON tax, reasoning first (prompt)", "json_cot", "free_cot"),
    ("JSON tax, reasoning first (schema)", "schema_cot", "free_cot"),
    ("answer first vs reasoning first (prompt)", "json_answer_first", "json_cot"),
    ("answer first vs reasoning first (schema)", "schema_answer_first", "schema_cot"),
    ("answer first vs answer only (prompt)", "json_answer_first", "json_direct"),
    ("JSON tax, answer only (prompt)", "json_direct", "free_direct"),
    ("schema vs prompt, reasoning first", "schema_cot", "json_cot"),
    ("schema vs prompt, answer first", "schema_answer_first", "json_answer_first"),
    ("schema vs prompt, answer only", "schema_direct", "json_direct"),
]


def load(split: str = "main") -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(C.RESULTS, "answers", "*.jsonl"))):
        is_main = not path.endswith(("_pilot.jsonl",))
        if (split == "main") != is_main:
            continue
        with open(path) as f:
            rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    if len(df):
        df["correct"] = df["correct"].astype(bool).astype(float)
    return df


def part_a(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.task.isin(["gsm8k", "mmlu"])].copy()


def part_b(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.task == "drift"].copy()
    d["field_acc"] = d["fields_correct"] / len(FIELDS)
    return d


def models_in(df: pd.DataFrame) -> list[str]:
    return [m for m in C.MODELS if m in set(df.model)]


# ---------------------------------------------------------------- Part A

def accuracy_table(a: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, t, v), g in a.groupby(["model", "task", "variant"]):
        ci = mean_ci(g["correct"].values)
        out.append({"model": m, "task": t, "condition": v, "n": len(g), "acc": ci["mean"], "lo": ci["lo"],
                    "hi": ci["hi"], "json_strict": g["json_strict"].mean() if g["json_strict"].notna().any() else np.nan,
                    "json_found": g["json_found"].mean() if g["json_found"].notna().any() else np.nan,
                    "rescued": g["rescued"].fillna(False).astype(bool).sum(),
                    "truncated": int(g["truncated"].sum()), "tokens": g["output_tokens"].mean()})
    return pd.DataFrame(out)


def contrast_table(a: pd.DataFrame, pooled_tasks: bool = False) -> pd.DataFrame:
    out = []
    groups = [("all", a)] if pooled_tasks else list(a.groupby("task"))
    for task, at in groups:
        for m in models_in(at):
            am = at[at.model == m]
            for name, x, ref in CONTRASTS:
                dx, dr = am[am.variant == x], am[am.variant == ref]
                if not len(dx) or not len(dr):
                    continue
                ci = paired_ci(dx, "correct", dr)
                out.append({"task": task, "model": m, "contrast": name, "condition": x, "reference": ref,
                            "diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"]})
    return pd.DataFrame(out)


def aa_table(a: pd.DataFrame) -> pd.DataFrame:
    """Answer changes between a condition and its A/A repeat."""
    out = []
    for (m, t), g in a.groupby(["model", "task"]):
        for rep, base in AA.items():
            x = g[g.variant == base].set_index("id")["answer"]
            y = g[g.variant == rep].set_index("id")["answer"]
            ids = x.index.intersection(y.index)
            if not len(ids):
                continue
            flips = (x[ids].astype(str) != y[ids].astype(str)).astype(float).values
            ci = mean_ci(flips)
            out.append({"model": m, "task": t, "condition": base, "flips": int(flips.sum()), "n": len(ids),
                        "rate": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]})
    return pd.DataFrame(out)


def _reasoning_number(text: str) -> str | None:
    """The last number in the reasoning field of an answer-first GSM8K reply."""
    from .parse import parse_json
    obj, _ = parse_json(text)
    if not obj or not isinstance(obj.get("reasoning"), str):
        return None
    nums = re.findall(NUM, obj["reasoning"].replace("$", ""))
    return canon_number(nums[-1]) if nums else None


def self_contradiction(a: pd.DataFrame) -> pd.DataFrame:
    """GSM8K answer-first replies: how often the reasoning's last number differs from the answer given
    before it, and how often that later number is the right one."""
    items = {it["id"]: it for it in load_items("main", ["gsm8k"])}
    out = []
    g = a[(a.task == "gsm8k") & a.variant.isin(["json_answer_first", "schema_answer_first"])]
    for (m, v), gg in g.groupby(["model", "variant"]):
        later = gg["output"].map(_reasoning_number)
        gold = gg["id"].map(lambda i: canon_number(items[i]["gold"]))
        has = later.notna() & gg["answer"].notna()
        differ = has & (later != gg["answer"])
        out.append({"model": m, "condition": v, "n": len(gg), "with_both": int(has.sum()),
                    "differ": int(differ.sum()), "differ_rate": differ.sum() / max(has.sum(), 1),
                    "later_right": int((differ & (later == gold)).sum()),
                    "answer_right_when_differ": int((differ & (gg["answer"] == gold)).sum())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Part B

def drift_table(b: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, v, mode), g in b.groupby(["model", "variant", "mode"]):
        fa, ex = mean_ci(g["field_acc"].values), mean_ci(g["correct"].values)
        out.append({"model": m, "variant": v, "mode": mode, "n": len(g), "field_acc": fa["mean"],
                    "lo": fa["lo"], "hi": fa["hi"], "exact": ex["mean"], "exact_lo": ex["lo"], "exact_hi": ex["hi"],
                    "parsed": g["parsed"].mean(), "json_strict": g["json_strict"].mean(),
                    "keys_exact": g["keys_exact"].mean()})
    return pd.DataFrame(out)


def drift_contrasts(b: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in models_in(b):
        for mode in ["json", "schema"]:
            g = b[(b.model == m) & (b["mode"] == mode)]
            base = g[g.variant == "base"]
            for v in DRIFT_VARIANTS:
                if v == "base":
                    continue
                ci = paired_ci(g[g.variant == v], "field_acc", base)
                ce = paired_ci(g[g.variant == v], "correct", base)
                out.append({"model": m, "mode": mode, "variant": v, "diff": ci["mean"], "lo": ci["lo"],
                            "hi": ci["hi"], "exact_diff": ce["mean"], "exact_lo": ce["lo"], "exact_hi": ce["hi"]})
        g = b[b.model == m]
        for v in DRIFT_VARIANTS:
            gv = g[g.variant == v]
            ci = paired_ci(gv[gv["mode"] == "schema"], "field_acc", gv[gv["mode"] == "json"])
            out.append({"model": m, "mode": "schema minus json", "variant": v, "diff": ci["mean"],
                        "lo": ci["lo"], "hi": ci["hi"], "exact_diff": np.nan, "exact_lo": np.nan, "exact_hi": np.nan})
    return pd.DataFrame(out)


def _field_rows(b: pd.DataFrame, variant: str, pick) -> pd.DataFrame:
    """One row per (model, mode, record, field) for the fields `pick(item)` selects."""
    items = {it["id"]: it for it in load_items("main", ["drift"])}
    rows = []
    for r in b[b.variant == variant].itertuples():
        for f in pick(items[r.id]):
            rows.append({"model": r.model, "mode": r.mode, "id": r.id, "task": "drift", "field": f,
                         "ok": float(r.fields[f]), "filled": float(r.filled[f])})
    return pd.DataFrame(rows)


def reformat_table(b: pd.DataFrame) -> pd.DataFrame:
    """Accuracy on the fields whose value the reformatted export changed, against the same fields in base."""
    def changed(it):
        return [TARGET[k] for k, x in it["record"].items() if x in REFORMAT.get(k, {})]
    ref, base = _field_rows(b, "reformatted", changed), _field_rows(b, "base", changed)
    out = []
    for (m, mode), g in ref.groupby(["model", "mode"]):
        gb = base[(base.model == m) & (base["mode"] == mode)]
        for f in ["all"] + sorted(g.field.unique()):
            x = g if f == "all" else g[g.field == f]
            y = gb if f == "all" else gb[gb.field == f]
            ci = paired_ci(x, "ok", y)
            out.append({"model": m, "mode": mode, "field": f, "n": len(x), "reformatted": x["ok"].mean(),
                        "base": y["ok"].mean(), "diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]})
    return pd.DataFrame(out)


def dropped_table(b: pd.DataFrame) -> pd.DataFrame:
    """How often a model fills in the field that was removed, against how often it fills a field the
    record never had (in base)."""
    drop = _field_rows(b, "dropped", lambda it: [TARGET[dropped_key(it)]])
    natural = _field_rows(b, "base", lambda it: [f for f in FIELDS if it["gold"][f] is None])
    out = []
    for (m, mode), g in drop.groupby(["model", "mode"]):
        gn = natural[(natural.model == m) & (natural["mode"] == mode)]
        a, n = mean_ci(g["filled"].values), mean_ci(gn["filled"].values) if len(gn) else {"mean": np.nan, "lo": np.nan, "hi": np.nan}
        out.append({"model": m, "mode": mode, "n": len(g), "dropped_filled": a["mean"], "lo": a["lo"], "hi": a["hi"],
                    "n_natural": len(gn), "natural_filled": n["mean"], "natural_lo": n["lo"], "natural_hi": n["hi"]})
    return pd.DataFrame(out)


def distractor_table(b: pd.DataFrame) -> pd.DataFrame:
    """Records where the name or landmark came from the look-alike fields."""
    from .parse import parse_json
    from .prompts import source_record
    from .data import norm
    items = {it["id"]: it for it in load_items("main", ["drift"])}
    out = []
    g = b[b.variant == "distractors"]
    for (m, mode), gg in g.groupby(["model", "mode"]):
        took = []
        for r in gg.itertuples():
            shown = json.loads(source_record(items[r.id], "distractors")[0])
            obj, _ = parse_json(r.output)
            obj = obj or {}
            took.append(float(norm(obj.get("name") or "") == norm(shown["previous_name"])
                              or norm(obj.get("near") or "") == norm(shown["opposite"])))
        ci = mean_ci(np.array(took))
        out.append({"model": m, "mode": mode, "n": len(took), "took": int(sum(took)), "rate": ci["mean"],
                    "lo": ci["lo"], "hi": ci["hi"]})
    return pd.DataFrame(out)


def drift_aa(b: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, mode), g in b.groupby(["model", "mode"]):
        x = g[g.variant == "base"].set_index("id")["answer"]
        y = g[g.variant == "aa"].set_index("id")["answer"]
        ids = x.index.intersection(y.index)
        flips = int((x[ids].astype(str) != y[ids].astype(str)).sum())
        out.append({"model": m, "mode": mode, "flips": flips, "n": len(ids)})
    return pd.DataFrame(out)
