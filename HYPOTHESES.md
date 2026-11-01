# Hypotheses and analysis plan

Written 2026-10-31 at 13:27 EDT, before the full run. Nothing below is edited after the full run starts; corrections go
in a dated addendum at the end.

What had been seen when this was written: a pilot on 4 pilot items per task for `qwen3:8b`, `llama3.1:8b`
and `gemma3:12b`, every condition (456 calls), used to check the templates, schemas and parsers on real
outputs, plus one hand-written question per model with an answer-first schema, on which `qwen3:8b` answered
9 and `llama3.1:8b` 19 before reasoning their way to the right 18. The pilot items are excluded from every
reported number. Pilot observations that informed a prediction are marked (P). The pilot showed, on far too
few items to estimate anything: `llama3.1:8b` running on after an answer-first reply until the 1,024-token
cap on 3 of 4 GSM8K items, so the JSON never closed; `gemma3:12b` wrapping every prompt-only JSON reply in a
code fence; and Part B near ceiling for all three models, with one invented price in a CSV record. The
GSM8K and MMLU items are the same seeded draws as prompt-brittleness, whose free-text results on them I
have read.

## Questions

**Part A, the format tax.** When a model has to answer in JSON instead of free text, does accuracy drop?
Is it the JSON itself, or where the reasoning goes (before the answer, after it, or nowhere)? And does
schema-constrained decoding (Ollama's `format`) help or hurt compared with asking for JSON in the prompt?

**Part B, schema drift.** A model converts a restaurant record into a fixed target schema. When the source
export changes in one way that keeps the content (keys renamed, keys reordered, values reformatted, extra
look-alike fields, a field missing, CSV instead of JSON, nested JSON), how much does extraction break, and
which changes matter? Does schema-constrained decoding change that?

## Design

- Models: `qwen3:8b` (thinking off), `llama3.1:8b`, `gemma3:12b`, all conditions on all items. `qwen3:30b-a3b`
  (the thinking-only release, thinking on) runs Part A on the first 50 items of each task, all conditions.
  4-bit, temperature 0, seed 7, through Ollama 0.40.
- Part A: 100 GSM8K and 100 MMLU items (four high-school subjects), 8 conditions each, plus two A/A
  repeats (free text with reasoning, and schema JSON with reasoning first, each sent a second time).
  Conditions: free text with reasoning; free text, answer only; JSON asked for in the prompt with reasoning
  first, answer first, or answer only; and the same three JSON shapes with schema-constrained decoding. A
  schema condition uses the identical prompt text as its prompt-only twin.
- Part B: 150 distinct E2E meaning representations with 5 to 8 of the 8 fields, each shown in 9 source
  formats (base JSON export, its A/A repeat, and 7 drifts, one change each), converted to one fixed target
  schema in two output modes (prompt-only JSON, schema-constrained). Fields the record lacks must be null.
- Caps: 1,024 output tokens for GSM8K, 768 for MMLU, 300 for Part B (4,096 for the thinking model), so a
  reasoning-first JSON reply is not cut off before its answer. Truncations are counted.

## Predictions

pp = percentage points. "For every model" means the three 8–12B models unless stated.

Part A

1. Reasoning helps: free text with reasoning beats free text answer-only on GSM8K by at least 20 pp for every
   model.
2. No tax when the reasoning comes first: prompt-only JSON with reasoning first is within 5 pp of free text
   with reasoning on GSM8K for every model, and the interval of the difference includes zero for at least two.
3. Answer first throws the reasoning away: JSON with the answer before the reasoning scores at least 10 pp
   below JSON with the reasoning first on GSM8K for every model, in both modes, and within 5 pp of JSON
   answer-only.
4. (P) In answer-first GSM8K replies, the reasoning written after the answer ends on a different number in
   at least 10% of items for at least two models.
5. Constraint is neutral for accuracy: schema-constrained minus prompt-only is within 3 pp for each JSON
   shape, averaged over the three models and both tasks.
6. (P) Prompt-only JSON contains a usable answer object (lenient parse) in at least 95% of replies that
   were not cut off, for every model and task, but the whole reply is a bare JSON object (strict) in under
   50% of `gemma3:12b`'s replies, because of code fences. Schema-constrained replies are strictly valid in
   100% of replies that were not cut off.
7. MMLU is less sensitive: on MMLU, JSON with reasoning first is within 3 pp of free text with reasoning,
   and answer-first costs less than on GSM8K, for every model.
8. Thinking removes the answer-first cost: for `qwen3:30b-a3b`, answer-first JSON is within 3 pp of
   reasoning-first JSON on GSM8K.
9. (P) Answer-first replies run away: at least one model hits the output cap on at least 20% of its
   answer-first GSM8K replies, against under 2% of its reasoning-first JSON replies.

Part B

10. The base export is easy: field accuracy at least 97% for every model in both modes.
11. Renamed and reordered keys cost nothing: within 2 pp of base field accuracy, every model, both modes.
12. Reformatted values are the expensive drift: on the fields whose values were reformatted, accuracy is
    at least 10 pp below the same fields in base, prompt-only, for at least two models; schema-constrained
    decoding recovers at least half of that drop.
13. (P) A missing field gets filled in: with one field removed, at least one model fills that field in at
    least 10% of records, more often than it fills fields the base record never had.
14. Look-alike extra fields rarely leak: the name or landmark is taken from an extra field in under 3% of
    records, every model.
15. CSV and nested JSON are within 3 pp of base field accuracy, every model.
16. Schema-constrained replies always parse with exactly the target keys; prompt-only replies do in at
    least 95%.
17. The A/A repeat changes no Part B answers for any model.

What would contradict "the format tax is about where the reasoning goes, not JSON": a large drop for
reasoning-first JSON, or no drop for answer-first JSON.

## Analysis plan (fixed in advance)

- Unit of resampling: item (Part A) or record (Part B). Every condition of an item is paired with the
  same item under the reference condition. 95% percentile bootstrap, 2,000 draws.
- Part A primary metric: accuracy. JSON replies are parsed leniently (the first JSON object in the reply,
  so code fences are tolerated); a reply with no usable JSON answer counts as wrong. Secondary: strict
  validity (the whole reply is one JSON object), accuracy of the value written after `"answer":` even when
  the JSON never closed ("as written", which separates a late or missing answer from a broken reply), how
  many invalid replies a free-text parser would have scored right ("rescued"), output tokens, truncations,
  and the answer-first self-contradiction rate (the last number in the reasoning differs from the answer).
- Part A contrasts, per model and task: reasoning vs answer-only (free text); JSON vs free text (reasoning
  first, and answer-only); answer-first vs reasoning-first and vs answer-only (each mode); schema vs
  prompt-only (each shape). An effect is said to exist when its interval excludes zero; "within x pp" is
  judged on the point estimate.
- Part B primary metric: field accuracy, the share of the 8 target fields right (a missing field is right
  only if null). Secondary: whole-record exact match, parse rate, strict validity, exact key set.
- Part B contrasts, per model and mode: each drift minus base (paired by record), and schema minus
  prompt-only for each source format. Reformatted values are also scored on the changed fields only,
  against the same fields in base. The missing-field fill rate is compared with the fill rate on fields the
  base record never had. Distractor leakage counts records whose name or landmark equals an extra field.
- Noise floor: answer changes between each A/A pair. Field comparison normalises case, accents and
  punctuation; `family_friendly` accepts true/false or yes/no.
- Not done: no prompt or schema was tuned on the main items, and no item is dropped after the run.

## Addendum, 2026-11-01 13:49 EST: scoring after the full run

The full run finished at 13:47 EST on 2026-11-01 (15,100 calls; `qwen3:30b-a3b` on 50 items per task as planned).
The Mac slept for about 9.7 hours during the `qwen3:30b-a3b` leg when the charger came out; calls are
cached one at a time, so this cost time and nothing else. Before scoring I read the failures: every
base-record Part B error, every unparsed free-text reply, the GSM8K items `llama3.1:8b` lost under JSON,
the truncated answer-first replies, and the Part B reformat and missing-field errors. No parser, prompt
or scoring change was made after the run; every number below is from `reports/results.md` as first
generated. Checks that came out clean: the schema setting took effect (across both parts, 1,363 of
`llama3.1:8b`'s 1,950 prompt/schema pairs differ in text, all 1,950 of `gemma3:12b`'s, mostly code fences
and whitespace), so the near-zero schema contrasts are not a request that ignored `format`; every unparsed free-text reply
was cut off at the cap (2 `qwen3:8b` MMLU, 4 `qwen3:30b-a3b` that spent all 4,096 tokens thinking); every
base Part B error is a real invented value (`eat_type` "restaurant" or `family_friendly` false for a record
without that field), not a gold-label problem; and the Part B prompt lists the allowed values, so
`llama3.1:8b` writing "20-25 GBP" for "£20-25" is a format violation, not strict scoring.

**Score: 15 of 17 held, 2 failed.** Several held predictions were cheap (Part B sat near ceiling, as the
pilot said it would); the two failures are the informative ones.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | Reasoning helps GSM8K by ≥20 pp | +56, +73, +75 | held |
| 2 | Reasoning-first JSON within 5 pp of free text on GSM8K, CI includes 0 for ≥2 | qwen −2 [−8, +3], gemma −5 [−11, 0], **llama −9 [−17, −1]** | **failed** (Llama) |
| 3 | Answer-first ≥10 pp below reasoning-first, within 5 pp of answer-only | −60/−61/−62 both modes; vs answer-only 0, −2, +2 | held |
| 4 | Answer-first reasoning ends on a different number in ≥10% for ≥2 models | 30%, 41%, 29% (prompt) | held |
| 5 | Schema minus prompt within 3 pp per shape, averaged | +0.3, −0.2, +0.2 pp | held |
| 6 | Lenient parse ≥95% of uncut prompt-only replies; gemma strict <50%; schema strict 100% of uncut | min 98.8% (gemma GSM8K); gemma strict 0/600; schema 2,689/2,689 | held |
| 7 | MMLU reasoning-first JSON within 3 pp of free text; answer-first costs less than GSM8K | −2, −2, −3; answer-first −3, +3, −4 | held |
| 8 | 30B thinking: answer-first within 3 pp of reasoning-first, GSM8K | −2 [−8, +4] | held |
| 9 | A model hits the cap on ≥20% of answer-first GSM8K vs <2% reasoning-first | gemma 47%/1%, llama 44%/0% (prompt) | held |
| 10 | Base export field accuracy ≥97% | 99.6–99.8% | held |
| 11 | Renamed/reordered within 2 pp of base | max 0.2 pp | held |
| 12 | Reformatted fields ≥10 pp below base for ≥2 models; schema recovers ≥half | **only llama**, −4.1 pp on changed fields (rating −18, price −16); qwen and gemma 0 | **failed**; schema recovered all of llama's drop |
| 13 | A removed field is filled in ≥10% of records for ≥1 model, above the never-present rate | gemma 16% vs 2.5%, qwen 11% vs 2.0% | held |
| 14 | Look-alike fields leak in <3% | 0/150 every model and mode | held |
| 15 | CSV and nested within 3 pp of base | max −1.2 pp | held |
| 16 | Schema always parses with exact keys; prompt-only ≥95% | schema 4,050/4,050; prompt-only ≥98.7% | held |
| 17 | Part B A/A changes no answers | 0/150 everywhere | held |

The pre-stated contradiction ("a large drop for reasoning-first JSON, or no drop for answer-first JSON")
did not happen for answer-first; for reasoning-first, Llama's −9 pp is a real but moderate drop, so the
conclusion holds for qwen3 and Gemma and only mostly for Llama.

**Exploratory, not pre-registered** (found while reading failures; treat as leads, not results):

- Why Llama pays a tax with the reasoning first: its JSON reasoning is shorter (125 vs 211 tokens) and in
  8 of 100 replies contains no worked arithmetic at all, only a plan ("First, calculate… Then…"), against 0
  of 100 in free text; 4 of those 8 were wrong. The rest of the gap is arithmetic slips in the shorter text.
- Answer-first replies know they are wrong: when the later reasoning ends on a different number, that
  number is the right one in 26/29 (qwen3), 16/23 (llama) and 14/15 (gemma) cases, and the committed answer
  is right in 0–1. The runaway replies are the model finding the right answer, failing to reconcile it with
  the one it already wrote ("I am still getting 80. I must be misinterpreting something."), and looping
  to the cap.
- With greedy decoding, the schema constraint changes the answer only where the model would have broken
  the format: the same Part A answer in 598/600 (qwen3) and 596/600 (llama) prompt/schema pairs, and Llama's reformat errors
  vanish under the enum. It is a format guarantee, not an accuracy lever, in either direction.
- Missing-field invention is almost entirely one field: with `eatType` removed, gemma writes "restaurant"
  in 24/24 records and qwen3 in 17/24; every other removed field is filled in at most 3 of ~25.
