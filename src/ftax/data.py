"""Download the pinned benchmark files and build the item set deterministically.

GSM8K and MMLU items are the same seeded draws as prompt-brittleness. Drift items are distinct E2E
meaning representations (structured restaurant records), each rendered later in several source formats.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

# E2E keys in the order the dataset writes them, and the target field each maps to.
E2E_KEYS = ["name", "eatType", "food", "priceRange", "customer rating", "area", "familyFriendly", "near"]
TARGET = {"name": "name", "eatType": "eat_type", "food": "food", "priceRange": "price_range",
          "customer rating": "customer_rating", "area": "area", "familyFriendly": "family_friendly",
          "near": "near"}
FIELDS = list(TARGET.values())
EAT_TYPES = ["restaurant", "pub", "coffee shop"]
PRICES = ["cheap", "moderate", "high", "less than £20", "£20-25", "more than £30"]
RATINGS = ["low", "average", "high", "1 out of 5", "3 out of 5", "5 out of 5"]
AREAS = ["riverside", "city centre"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_path(name: str) -> str:
    return os.path.join(C.RAW, f"{name}.parquet")


def fetch() -> dict:
    os.makedirs(C.RAW, exist_ok=True)
    out = {}
    for name, (repo, rev, path, pinned) in C.SOURCES.items():
        dest = raw_path(name)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
        got = sha256(dest)
        if pinned and got != pinned:
            raise RuntimeError(f"{dest}: sha256 {got} does not match pinned {pinned}")
        out[name] = {"repo": repo, "revision": rev, "path": path, "bytes": os.path.getsize(dest), "sha256": got}
    with open(os.path.join(C.DATA, "sources.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def parse_mr(mr: str) -> dict:
    return {k.strip(): v.strip() for k, v in re.findall(r"([^\[,]+)\[([^\]]*)\]", mr)}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9£ ]", " ", s)).strip()


def drift_gold(mr: dict) -> dict:
    """Target record: every field, None where the meaning representation doesn't give it."""
    g = {f: None for f in FIELDS}
    for k, v in mr.items():
        g[TARGET[k]] = v
    if g["family_friendly"] is not None:
        g["family_friendly"] = g["family_friendly"] == "yes"
    return g


def drift_usable(mr: dict) -> bool:
    if set(mr) - set(E2E_KEYS) or "name" not in mr or len(mr) < 5:
        return False
    if mr.get("eatType", EAT_TYPES[0]) not in EAT_TYPES or mr.get("area", AREAS[0]) not in AREAS:
        return False
    if mr.get("priceRange", PRICES[0]) not in PRICES or mr.get("customer rating", RATINGS[0]) not in RATINGS:
        return False
    return norm(mr["name"]) != norm(mr.get("near", ""))


def _sample(n_total: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_total)


def build_items() -> list[dict]:
    """Pilot items first, then main items, per task. All draws are seeded."""
    items = []
    n = C.N_PER_TASK + C.N_PILOT

    g = pd.read_parquet(raw_path("gsm8k_test")).reset_index(drop=True)
    for rank, i in enumerate(_sample(len(g), C.SEED)[:n]):
        r = g.iloc[i]
        items.append({"id": f"gsm8k-{i:04d}", "task": "gsm8k", "split": "pilot" if rank < C.N_PILOT else "main",
                      "question": r.question.strip(), "gold": gsm8k_gold(r.answer), "source_index": int(i)})

    per_subject = C.N_PER_TASK // len(C.MMLU_SUBJECTS)
    pilot_per = -(-C.N_PILOT // len(C.MMLU_SUBJECTS))
    n_pilot_left = C.N_PILOT
    mm = []
    for si, subj in enumerate(C.MMLU_SUBJECTS):
        m = pd.read_parquet(raw_path(f"mmlu_{subj}")).reset_index(drop=True)
        k_pilot = min(pilot_per, n_pilot_left)
        n_pilot_left -= k_pilot
        order = _sample(len(m), C.SEED + 1 + si)
        for rank, i in enumerate(order[:per_subject + k_pilot]):
            r = m.iloc[i]
            mm.append({"id": f"mmlu-{subj}-{i:03d}", "task": "mmlu", "subject": subj,
                       "split": "pilot" if rank < k_pilot else "main",
                       "question": r.question.strip(), "options": [str(c).strip() for c in r.choices],
                       "gold": "ABCD"[int(r.answer)], "source_index": int(i)})
    # interleave subjects so that any prefix of the MMLU items is balanced across them
    by_subj = [[it for it in mm if it["subject"] == s] for s in C.MMLU_SUBJECTS]
    mixed = [it for row in zip(*by_subj) for it in row]
    mixed += [it for it in mm if it not in mixed]
    items += sorted(mixed, key=lambda it: it["split"] != "pilot")

    e = pd.read_parquet(raw_path("e2e_test")).reset_index(drop=True)
    seen, usable = set(), []
    for i, r in e.iterrows():
        if r.meaning_representation in seen:
            continue
        seen.add(r.meaning_representation)
        mr = parse_mr(r.meaning_representation)
        if drift_usable(mr):
            usable.append((i, mr))
    n = C.N_DRIFT + C.N_DRIFT_PILOT
    for rank, j in enumerate(_sample(len(usable), C.SEED + 20)[:n]):
        i, mr = usable[j]
        items.append({"id": f"drift-{i:04d}", "task": "drift", "split": "pilot" if rank < C.N_DRIFT_PILOT else "main",
                      "record": {k: mr[k] for k in E2E_KEYS if k in mr}, "gold": drift_gold(mr),
                      "source_index": int(i)})
    return items


def write_items() -> list[dict]:
    items = build_items()
    with open(C.ITEMS, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def load_items(split: str | None = "main", tasks: list[str] | None = None) -> list[dict]:
    with open(C.ITEMS) as f:
        items = [json.loads(line) for line in f]
    return [it for it in items if (split is None or it["split"] == split) and (tasks is None or it["task"] in tasks)]
