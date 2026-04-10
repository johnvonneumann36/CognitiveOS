# Benchmark Caveats

These benchmarks are useful, but none of them alone defines "real" long-term memory quality.

## LongMemEval

- Strong for cross-session chat-assistant memory.
- Still narrower than open-ended lifelong memory because the history is pre-constructed.
- Official scoring depends on the upstream evaluation flow and external model-based judging.

## LoCoMo

- Strong for long conversational recall and QA.
- The public repo covers more than QA, but this harness only uses QA in v1.
- Scores here should not be interpreted as event-summarization or multimodal memory quality.

## MemBench

- Stronger coverage of memory types and interaction settings.
- Data preparation is less turnkey than the other two benchmarks.
- The first harness version emphasizes data validation and runnable subsets, not the full paper reproduction.

## Reporting Guidance

- Always distinguish `smoke` scores from real provider scores.
- Prefer reporting both exact-match style local metrics and any official or upstream metrics separately.
- Do not claim system-wide long-term memory quality from a single benchmark number.
