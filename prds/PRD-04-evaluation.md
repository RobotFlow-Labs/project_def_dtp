# PRD-04: Evaluation & Reproduction

> Module: DEF-DTP | Priority: P0  
> Depends on: PRD-02, PRD-03  
> Status: ⬜ Not started

## Objective
The repo reproduces the paper's evaluation protocol, comparison baselines, tables, and figures with an explicit parity report documenting any metric gaps.

## Context (from paper)
The paper's value is established through benchmarked attack success and degradation results, especially Table I, Table II, and the comparative analysis against PSO and SBA. Reproduction must make these comparisons first-class outputs rather than ad hoc notebook work.

**Paper reference:** §V-A, §V-B, §V-C, Fig. 4-6, Table I-II  
Key paper wording: `100 randomly sampled scenarios per dataset` and `fixed computational budget of 1,000 model queries per scenario`.

## Acceptance Criteria
- [ ] 100-sample evaluation manifests are generated deterministically for nuScenes and Apolloscape
- [ ] Intention misclassification evaluation computes left/right/front/rear offsets and ASR
- [ ] Degradation evaluation computes ADE, FDE, MR, ORR, and perturbation MSE
- [ ] PSO and SBA baselines are wrapped under the same sample manifests and query budget
- [ ] Scripted report generates:
  - Table I
  - Table II
  - comparison plot similar to Fig. 4
  - convergence plot similar to Fig. 6
- [ ] Test: `uv run pytest tests/test_metrics.py tests/test_protocol.py tests/test_report.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_def_dtp/evaluation/metrics.py` | ADE, FDE, MR, ORR, directional offsets, MSE | §IV-A / §V-B | ~180 |
| `src/anima_def_dtp/evaluation/protocol.py` | 100-sample protocol and query-budget runner | §V-A | ~220 |
| `src/anima_def_dtp/evaluation/baselines.py` | PSO and SBA wrappers under shared protocol | §V-A / Fig. 4 | ~180 |
| `scripts/reproduce_paper.py` | end-to-end reproduction entrypoint | §V-B / §V-C | ~200 |
| `tests/test_metrics.py` | metric correctness tests | — | ~120 |
| `tests/test_protocol.py` | sample-manifest and budget tests | — | ~120 |
| `tests/test_report.py` | report artifact smoke tests | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- `AttackResult` bundles from PRD-02 / PRD-03
- sample manifest with 100 cases per dataset
- predictor and baseline registry

### Outputs
- `artifacts/reports/table_i.csv`
- `artifacts/reports/table_ii.csv`
- `artifacts/plots/fig4_comparison.png`
- `artifacts/plots/fig6_convergence.png`
- `artifacts/reports/parity_report.md`

### Algorithm
```python
# Paper Section V — benchmark reproduction
def reproduce_table_i(cases, predictor, attacker) -> dict:
    # left, right, front, rear, ASR
    ...

def reproduce_table_ii(cases, predictor, attacker) -> dict:
    # ADE, FDE, MR, ORR, perturbation MSE
    ...
```

## Dependencies
```toml
pandas = ">=2.2"
matplotlib = ">=3.8"
seaborn = ">=0.13"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| sample manifests | 100 cases per dataset | `artifacts/sample_manifests/` | generated locally |
| attack/eval outputs | per experiment | `artifacts/runs/` | local generation |

## Test Plan
```bash
uv run pytest tests/test_metrics.py tests/test_protocol.py tests/test_report.py -v
uv run python scripts/reproduce_paper.py --help
```

## References
- Paper: §V-A to §V-C, Table I, Table II, Fig. 4-6
- Reference impl: `repositories/DTP-Attack/blackbox_utils.py`, `repositories/DTP-Attack/prediction/evaluate/`
- Depends on: PRD-02, PRD-03
- Feeds into: PRD-07
