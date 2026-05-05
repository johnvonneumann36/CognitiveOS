# CognitiveOS 0.2 Readiness

Use this as the local cut checklist for `v0.2.0`.

## Checks

- Version alignment: `pyproject.toml`, `src/cognitiveos/__init__.py`, tag, PyPI artifact, and GitHub Release use `0.2.0`.
- Release reliability: GitHub Release can create, update, and re-upload artifacts; PyPI publish skips existing versions and retries transient registry errors.
- Fresh clone dependencies: `pip install -e ".[dev,benchmark]"` declares benchmark-only `ijson`.
- Memory projection: `MEMORY.MD` renders pinned/profile memory first, then capped compressed dream super-nodes with `projection_policy_version`.
- Dream explainability: dream run JSON includes effective config, candidate explanations, cluster explanations, skipped unions, entity gate decisions, and projected memory ids.
- Bootstrap safety: no generated `.cognitiveos` artifacts, runtime `MEMORY.MD`, local user paths, or real email addresses are tracked.

## Commands

```bash
python scripts/check_0_2_readiness.py
python -m pytest
python -m ruff check .
```
