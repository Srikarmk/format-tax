import csv
import io
import json

from ftax import data
from ftax.llm import cache_key
from ftax.prompts import (A_VARIANTS, DRIFT_VARIANTS, REFORMAT, RENAMED, answer_schema, dropped_key,
                          render, source_record)

ITEMS = data.load_items(None)
DRIFT = [it for it in ITEMS if it["task"] == "drift"]


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    else:
        yield x


def test_every_drift_format_carries_exactly_the_record():
    for it in DRIFT:
        rec = it["record"]
        for v in DRIFT_VARIANTS:
            text, gold = source_record(it, v)
            if v == "csv":
                header, row = list(csv.reader(io.StringIO(text)))
                shown = dict(zip(header, row))
                assert shown == rec
                continue
            obj = json.loads(text)
            vals = list(_leaves(obj))
            if v == "renamed":
                assert obj == {RENAMED[k]: x for k, x in rec.items()}
            elif v == "reformatted":
                assert obj == {k: REFORMAT.get(k, {}).get(x, x) for k, x in rec.items()}
            elif v == "distractors":
                assert {k: obj[k] for k in rec} == rec and len(obj) == len(rec) + 3
                assert obj["previous_name"] != rec["name"]
            elif v == "dropped":
                k = dropped_key(it)
                assert k != "name" and obj == {x: y for x, y in rec.items() if x != k}
                assert gold[data.TARGET[k]] is None
                assert sum(gold[f] is None for f in data.FIELDS) == sum(it["gold"][f] is None for f in data.FIELDS) + 1
            elif v == "reordered":
                assert obj == rec and list(obj) != list(rec)
            else:
                assert sorted(map(str, vals)) == sorted(map(str, rec.values()))
            if v != "dropped":
                assert gold == it["gold"]


def test_gold_matches_record():
    for it in DRIFT:
        for k, x in it["record"].items():
            g = it["gold"][data.TARGET[k]]
            assert g == (x == "yes" if k == "familyFriendly" else x)


def test_schema_property_order_follows_condition():
    assert list(answer_schema("gsm8k", ["answer", "reasoning"])["properties"]) == ["answer", "reasoning"]
    assert list(answer_schema("mmlu", ["reasoning", "answer"])["properties"]) == ["reasoning", "answer"]


def test_schema_twins_share_prompt_text_and_aa_gets_its_own_cache_entry():
    for it in [i for i in ITEMS if i["task"] != "drift"][:20]:
        for shape in ["cot", "answer_first", "direct"]:
            a, b = render(it, f"json_{shape}"), render(it, f"schema_{shape}")
            assert a.messages == b.messages and a.fmt is None and b.fmt is not None
        base, aa = render(it, "free_cot"), render(it, "aa_free_cot")
        assert base.messages == aa.messages
        opts = {"temperature": 0}
        assert cache_key("m", base.messages, opts, base.tag, base.fmt) != cache_key("m", aa.messages, opts, aa.tag, aa.fmt)
        s1, s2 = render(it, "schema_cot"), render(it, "schema_answer_first")
        assert cache_key("m", s1.messages, opts, "", s1.fmt) != cache_key("m", s1.messages, opts, "", s2.fmt)
    assert len(A_VARIANTS) == 10


def test_items_are_seeded_and_disjoint():
    ids = [it["id"] for it in ITEMS]
    assert len(ids) == len(set(ids)) == 380
    names = {(it["task"], it["split"]) for it in ITEMS}
    assert ("drift", "main") in names and ("drift", "pilot") in names
