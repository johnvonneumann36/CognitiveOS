# CognitiveOS External Benchmark Harness

This folder prepares and runs three long-term memory benchmarks against CognitiveOS:

- `LongMemEval`
- `LoCoMo`
- `MemBench`

The harness is intentionally light-weight:

- external benchmark repos are cloned into `benchmark/vendor/`
- benchmark datasets live under `benchmark/data/`
- run outputs live under `benchmark/results/`
- only the wrapper scripts, manifests, prompts, and notes are tracked in Git

## Modes

- `smoke`
  - fast local sanity path
  - uses isolated CognitiveOS runtimes plus deterministic fake providers
  - validates ingestion, retrieval, file layout, and output contracts
- `provider`
  - uses the current CognitiveOS provider environment
  - generates real predictions with your configured embedding and chat providers
  - uses official or half-official scoring where practical

## Commands

```bash
python benchmark/scripts/prepare.py --suite all
python benchmark/scripts/validate.py --suite all
python benchmark/scripts/run.py --suite longmemeval --mode smoke --limit 3
python benchmark/scripts/run.py --suite longmemeval --mode provider --dataset-split oracle --limit 10
python benchmark/scripts/run.py --suite locomo --mode smoke --limit 3
python benchmark/scripts/run.py --suite membench --mode smoke --dataset-split factual --limit 3
```

## Data Preparation Policy

- `LongMemEval`
  - benchmark repo is cloned automatically
  - `oracle` and `s` data files are downloaded automatically
  - `m` is optional and only needed when explicitly requested
- `LoCoMo`
  - benchmark repo is cloned automatically
  - `data/locomo10.json` is copied into `benchmark/data/locomo/`
- `MemBench`
  - benchmark repo is cloned automatically
  - benchmark data still requires manual download from the links documented in the upstream README
  - extract the bundle so `benchmark/data/membench/data/` contains:
    - `FirstAgentDataLowLevel.json`
    - `ThirdAgentDataLowLevel.json`
    - `FirstAgentDataHighLevel.json`
    - `ThirdAgentDataHighLevel.json`
  - an empty `data2test/` directory is acceptable; the harness no longer depends on it
  - `prepare.py` leaves the suite in a partial state until the manual files are placed
  - `validate.py` prints the exact target directory and next command

## Output Layout

Each run writes to:

```text
benchmark/results/<suite>/<run_id>/
  run_config.json
  predictions.jsonl
  metrics.json
  runtime_metadata.json
```

## Current Scope

- LongMemEval: `oracle` and `s`
- LoCoMo: QA task only
- MemBench: factual / reflective evaluation on the manually downloaded official `data/` bundle

Event summarization, multimodal dialog generation, and extended noise experiments are intentionally out of scope for v1.
