"""Ollama chat client with an on-disk cache (from prompt-brittleness, plus the `format` field).

The cache key is a hash of (model, messages, options, think, format, tag). `fmt` is sent as Ollama's
`format`: None for free text, or a JSON schema for constrained decoding. `tag` is not
sent to the server; it gives the A/A repeat of the base prompt its own cache entry,
so the repeat is a real second call.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import urllib.request

from . import config as C

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def think_flag(model: str) -> bool | None:
    """The `think` field sent to Ollama: off for qwen3, on for the thinking-only tag, absent otherwise."""
    if model in C.THINKING_ONLY:
        return True
    return False if model.startswith("qwen3") else None


def request_body(model: str, messages: list[dict], options: dict, fmt: dict | None = None) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if think_flag(model) is not None:
        body["think"] = think_flag(model)
    if fmt is not None:
        body["format"] = fmt
    return body


def cache_key(model: str, messages: list[dict], options: dict, tag: str = "", fmt: dict | None = None) -> str:
    # sort_keys would hide a change in schema property order, which is one of the things being tested,
    # so the schema is serialised in its own order
    blob = json.dumps({"model": model, "messages": messages, "options": options,
                       "think": think_flag(model), "tag": tag}, sort_keys=True)
    if fmt is not None:
        blob += "|format=" + json.dumps(fmt)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("create table if not exists calls (key text primary key, model text, response text)")
        self.db.commit()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("select response from calls where key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, resp: dict) -> None:
        with self.lock:
            self.db.execute("insert or replace into calls values (?, ?, ?)", (key, model, json.dumps(resp)))
            self.db.commit()

    def __len__(self) -> int:
        with self.lock:
            return self.db.execute("select count(*) from calls").fetchone()[0]


class CacheMiss(RuntimeError):
    pass


class Client:
    def __init__(self, cache: Cache, url: str = C.OLLAMA_URL, cache_only: bool = False):
        self.cache, self.url, self.cache_only = cache, url, cache_only
        self.stats = {"hits": 0, "misses": 0}
        self.lock = threading.Lock()

    def chat(self, model: str, messages: list[dict], options: dict, tag: str = "",
             fmt: dict | None = None) -> dict:
        """Returns {"text", "prompt_tokens", "output_tokens", "seconds", "cached"}."""
        key = cache_key(model, messages, options, tag, fmt)
        hit = self.cache.get(key)
        if hit is not None:
            with self.lock:
                self.stats["hits"] += 1
            return {**hit, "cached": True}
        if self.cache_only:
            raise CacheMiss(key)
        body = json.dumps(request_body(model, messages, options, fmt)).encode()
        for attempt in range(5):
            try:
                t0 = time.time()
                req = urllib.request.Request(self.url + "/api/chat", body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.load(r)
                wall = time.time() - t0
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        raw = data["message"]["content"]
        resp = {
            "text": THINK_RE.sub("", raw).strip(),
            "raw": raw,
            "thinking": data["message"].get("thinking", ""),
            "done_reason": data.get("done_reason", ""),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            # server-side compute time, which excludes time spent queued behind other clients
            "seconds": (data.get("prompt_eval_duration", 0) + data.get("eval_duration", 0)) / 1e9,
            "wall_seconds": wall,
        }
        self.cache.put(key, model, resp)
        with self.lock:
            self.stats["misses"] += 1
        return {**resp, "cached": False}


def model_info(model: str, url: str = C.OLLAMA_URL) -> dict:
    req = urllib.request.Request(url + "/api/show", json.dumps({"model": model}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    tags = json.load(urllib.request.urlopen(url + "/api/tags", timeout=60))
    digest = next((m["digest"] for m in tags["models"] if m["name"] == model), "")
    return {"model": model, "digest": digest, "capabilities": d.get("capabilities", []),
            "details": d.get("details", {})}
