"""Send every (item, condition) prompt to a model, parse the answer, store one row per call."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C
from .llm import CacheMiss, Client
from .parse import check_a, check_drift
from .prompts import AA, MODES, render, variants


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def answers_path(model: str, split: str = "main") -> str:
    name = slug(model) + ("" if split == "main" else f"_{split}")
    return os.path.join(C.RESULTS, "answers", f"{name}.jsonl")


class Progress:
    def __init__(self, total: int, label: str, every: int = 50):
        self.total, self.label, self.every = total, label, every
        self.done, self.t0, self.lock = 0, time.time(), threading.Lock()

    def tick(self, client: Client):
        with self.lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                el = time.time() - self.t0
                print(f"[{self.label}] {self.done}/{self.total} new_calls={client.stats['misses']} "
                      f"hits={client.stats['hits']} elapsed={el:.0f}s", flush=True)


def jobs_for(items: list[dict], only: list[str] | None = None) -> list[tuple[dict, str, str]]:
    """Item by item, so an interrupted run leaves whole items behind. Order within an item is shuffled
    so a condition and its A/A repeat are not sent back to back."""
    jobs = []
    rng = random.Random(C.SEED)
    for it in items:
        if it["task"] == "drift":
            js = [(it, v, m) for v in variants("drift") for m in MODES]
        else:
            js = [(it, v, "") for v in variants(it["task"])]
        js = [j for j in js if only is None or j[1] in only]
        rng.shuffle(js)
        jobs += js
    return jobs


def run_one(client: Client, model: str, it: dict, v: str, mode: str) -> dict:
    r = render(it, v, mode or "json")
    out = client.chat(model, r.messages, C.options(it["task"], model), tag=r.tag, fmt=r.fmt)
    row = {
        "model": model, "id": it["id"], "task": it["task"], "split": it["split"], "variant": v, "mode": mode,
        "output": out["text"], "truncated": out.get("done_reason") == "length",
        "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
        "thinking_chars": len(out.get("thinking") or ""), "seconds": out["seconds"], "cached": out["cached"],
    }
    if it["task"] == "drift":
        chk = check_drift(out["text"], r.gold)
        row.update(parsed=chk["parsed"], json_strict=chk["json_strict"], keys_exact=chk["keys_exact"],
                   answer=chk["answer"], fields=chk["fields"], filled=chk["filled"],
                   fields_correct=sum(chk["fields"].values()), correct=all(chk["fields"].values()))
    else:
        chk = check_a(it["task"], AA.get(v, v), out["text"], r.gold, it.get("options"))
        row.update(chk)
    return row


def run(items: list[dict], client: Client, model: str, workers: int = 2, only: list[str] | None = None,
        split: str = "main", skip_missing: bool = False) -> list[dict]:
    jobs = jobs_for(items, only)
    prog = Progress(len(jobs), model)

    def work(job):
        try:
            row = run_one(client, model, *job)
        except CacheMiss:
            if not skip_missing:
                raise
            row = None
        prog.tick(client)
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(work, jobs) if r is not None]
    path = answers_path(model, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = []
    if os.path.exists(path):
        keys = {(r["id"], r["variant"], r["mode"]) for r in rows}
        with open(path) as f:
            old = [r for r in map(json.loads, f) if (r["id"], r["variant"], r["mode"]) not in keys]
    rows = sorted(old + rows, key=lambda r: (r["task"], r["id"], r["variant"], r["mode"]))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
