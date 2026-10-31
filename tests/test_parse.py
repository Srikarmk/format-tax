from ftax.parse import check_a, check_drift, json_answer, parse_json

GOLD = {"name": "The Mill", "eat_type": "pub", "food": None, "price_range": "£20-25",
        "customer_rating": "5 out of 5", "area": "riverside", "family_friendly": True, "near": "Café Sicilia"}


def test_parse_json_strict_and_lenient():
    assert parse_json('{"answer": 3}') == ({"answer": 3}, True)
    obj, strict = parse_json('Sure:\n```json\n{"reasoning": "a {b} c", "answer": 3}\n```')
    assert obj == {"reasoning": "a {b} c", "answer": 3} and not strict
    assert parse_json("no json here") == (None, False)
    assert parse_json('{"answer": 3') == (None, False)


def test_json_answer_gsm8k_and_mmlu():
    assert json_answer("gsm8k", {"answer": "$1,250.00"}, None) == "1250"
    assert json_answer("gsm8k", {"answer": 18}, None) == "18"
    assert json_answer("gsm8k", {"answer": True}, None) is None
    assert json_answer("gsm8k", {"reasoning": "x"}, None) is None
    assert json_answer("mmlu", {"answer": "b"}, None) == "B"
    assert json_answer("mmlu", {"answer": "C. Mitosis"}, None) == "C"
    assert json_answer("mmlu", {"answer": "Mitosis"}, ["Meiosis", "Osmosis", "Mitosis", "Fission"]) == "C"


def test_check_a_counts_missing_json_answer_wrong_but_records_rescue():
    r = check_a("gsm8k", "json_cot", "The answer is 18.\nAnswer: 18", "18")
    assert r["correct"] is False and r["rescued"] is True and r["json_found"] is False
    r = check_a("gsm8k", "free_cot", "6 + 12 = 18\nAnswer: 18", "18")
    assert r["correct"] and r["method"] == "cue"
    r = check_a("mmlu", "schema_answer_first", '{"answer": "D", "reasoning": "..."}', "D")
    assert r["correct"] and r["json_strict"]


def test_check_drift_nulls_bools_and_fills():
    out = ('{"name": "the mill", "eat_type": "pub", "food": null, "price_range": "£20-25", '
           '"customer_rating": "5 out of 5", "area": "Riverside", "family_friendly": "yes", "near": "Cafe Sicilia"}')
    r = check_drift(out, GOLD)
    assert all(r["fields"].values()) and not any(r["filled"].values()) and r["keys_exact"]
    r = check_drift(out.replace('"food": null', '"food": "English"'), GOLD)
    assert r["fields"]["food"] is False and r["filled"]["food"] is True
    r = check_drift(out.replace('"food": null', '"food": "N/A"'), GOLD)
    assert r["fields"]["food"] and not r["filled"]["food"]
    r = check_drift(out.replace('"£20-25"', '"20-25 GBP"'), GOLD)
    assert r["fields"]["price_range"] is False
    r = check_drift("not json", GOLD)
    assert not r["parsed"] and not any(r["fields"].values())
