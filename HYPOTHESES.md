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
