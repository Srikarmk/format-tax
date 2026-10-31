"""ftax command line."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as C


def _client(args):
    from .llm import Cache, Client
    os.makedirs(os.path.dirname(C.CACHE_DB), exist_ok=True)
    return Client(Cache(C.CACHE_DB), cache_only=getattr(args, "cache_only", False))


def _items(args):
    from .data import load_items
    tasks = args.tasks.split(",") if args.tasks else None
    items = load_items(args.split, tasks)
    if args.per_task:
        keep = []
        for t in C.TASKS:
            keep += [it for it in items if it["task"] == t][:args.per_task]
        items = keep
    return items


def cmd_fetch(args):
    from .data import fetch
    for name, meta in fetch().items():
        print(f"{name}: {meta['repo']}@{meta['revision'][:10]} {meta['path']} {meta['bytes']} bytes")


def cmd_items(args):
    from collections import Counter

    from .data import write_items
    items = write_items()
    print(f"wrote {len(items)} items to data/items.jsonl:",
          dict(Counter(f"{i['task']}/{i['split']}" for i in items)))


def cmd_run(args):
    from .runner import run
    items = _items(args)
    client = _client(args)
    only = args.variants.split(",") if args.variants else None
    for m in args.models.split(","):
        if not args.cache_only:
            _record_model(m)
        rows = run(items, client, m, workers=args.workers, only=only, split=args.split,
                   skip_missing=args.skip_missing)
        acc = sum(bool(r["correct"]) for r in rows) / max(len(rows), 1)
        print(f"{m}: {len(rows)} rows, accuracy {acc:.3f}")
    print("cache", client.stats)


def _record_model(model: str):
    from .llm import model_info, think_flag
    path = os.path.join(C.RESULTS, "models.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta[model] = {**model_info(model), "think": think_flag(model),
                   "options": {t: C.options(t, model) for t in C.TASKS}}
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)


def cmd_show(args):
    """Print the rendered prompts for one item, and the stored outputs if there are any."""
    from .data import load_items
    from .prompts import render, variants
    from .runner import answers_path
    it = next(i for i in load_items(None) if i["id"] == args.id)
    vs = args.variants.split(",") if args.variants else variants(it["task"])
    rows = {}
    path = answers_path(args.model, it["split"])
    if os.path.exists(path):
        with open(path) as f:
            rows = {(r["variant"], r["mode"]): r for r in map(json.loads, f) if r["id"] == it["id"]}
    modes = ["json", "schema"] if it["task"] == "drift" else [""]
    for v in vs:
        r = render(it, v)
        print(f"===== {v}  gold={r.gold}")
        for m in r.messages:
            print(f"[{m['role']}]\n{m['content']}")
        for mode in modes:
            x = rows.get((v, mode))
            if x:
                print(f"--- {args.model} {mode}: answer={x['answer']} correct={x['correct']}\n{x['output']}")
        print()


def cmd_report(args):
    from .report import write_report
    write_report(args.split)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ftax")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the pinned benchmark files")
    sub.add_parser("items", help="build data/items.jsonl from the downloaded files")
    r = sub.add_parser("run", help="send every condition of every item to the models")
    r.add_argument("--models", default=C.MODELS[0])
    r.add_argument("--split", default="main", choices=["main", "pilot"])
    r.add_argument("--tasks", default="")
    r.add_argument("--variants", default="")
    r.add_argument("--per-task", type=int, default=0, help="only the first N items of each task")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--cache-only", action="store_true")
    r.add_argument("--skip-missing", action="store_true", help="with --cache-only, skip prompts not in the cache")
    s = sub.add_parser("show", help="print an item's rendered prompts and stored outputs")
    s.add_argument("id")
    s.add_argument("--model", default=C.MODELS[0])
    s.add_argument("--variants", default="")
    rp = sub.add_parser("report", help="write reports/results.md and reports/figures/")
    rp.add_argument("--split", default="main", choices=["main", "pilot"])
    args = p.parse_args(argv)
    fn = {"fetch": cmd_fetch, "items": cmd_items, "run": cmd_run, "show": cmd_show, "report": cmd_report}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
