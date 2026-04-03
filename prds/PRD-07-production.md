# PRD-07: Production Hardening

> Module: DEF-DTP | Priority: P2  
> Depends on: PRD-04, PRD-05, PRD-06  
> Status: ⬜ Not started

## Objective
The module supports repeatable robustness campaigns, audit-grade reporting, failure handling, and release validation without drifting away from the paper-faithful attack implementation.

## Context (from paper)
The paper proves the vulnerability. Production hardening makes that result operational for ANIMA: repeated campaigns, exports, report bundles, runtime safeguards, and documented parity boundaries.

**Paper reference:** §V and §VI because production readiness depends on preserving the measured attack behavior and its stated limitations.  
Key paper wording: `urgent needs for robust defenses`.

## Acceptance Criteria
- [ ] Batch runner executes attack campaigns over a manifest of cases and predictors
- [ ] Every run emits audit logs with:
  - correct paper ID
  - predictor
  - attack objective
  - query count
  - perturbation magnitude
- [ ] Final report exports parity summary against Table I and II targets
- [ ] Runtime guards fail clearly when datasets, maps, or checkpoints are missing
- [ ] Test: `uv run pytest tests/test_campaign.py tests/test_release_artifacts.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `scripts/run_campaign.py` | batch robustness campaign runner | §V | ~220 |
| `scripts/export_report.py` | export parity and audit reports | §V / §VI | ~180 |
| `src/anima_def_dtp/runtime_guards.py` | missing-asset and config validation | — | ~120 |
| `tests/test_campaign.py` | campaign smoke tests | — | ~120 |
| `tests/test_release_artifacts.py` | report/export tests | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- predictor list
- dataset manifest
- attack objective grid

### Outputs
- `artifacts/campaigns/<run_id>/results.jsonl`
- `artifacts/campaigns/<run_id>/parity_report.md`
- `artifacts/campaigns/<run_id>/tables/*.csv`

### Algorithm
```python
def run_campaign(manifest, predictors, objectives):
    for case in manifest:
        for predictor in predictors:
            for objective in objectives:
                yield run_single_case(case, predictor, objective)
```

## Dependencies
```toml
pandas = ">=2.2"
jinja2 = ">=3.1"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| campaign manifests | small text/json | `artifacts/manifests/` | generated locally |
| release docs | small | `artifacts/campaigns/` | generated locally |

## Test Plan
```bash
uv run pytest tests/test_campaign.py tests/test_release_artifacts.py -v
uv run python scripts/run_campaign.py --help
uv run python scripts/export_report.py --help
```

## References
- Paper: §V, §VI
- Depends on: PRD-04, PRD-05, PRD-06
- Feeds into: final release validation
