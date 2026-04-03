# DEF-DTP — Pipeline Map

This map ties the paper's method and the upstream reference repo to the planned ANIMA implementation.

## Canonical Flow
1. Load benchmark scenario window `H_t` and target-agent observation `X_t`.
2. Evaluate attack objective:
   - intention misclassification: lateral / longitudinal directional offsets
   - prediction degradation: ADE / FDE threshold crossing
3. Convert objective to a binary adversarial criterion `c(.)`.
4. Run decision-based boundary walking:
   - random adversarial initialization
   - orthogonal step
   - forward step
   - dynamic `delta` and `epsilon` updates
5. Query predictor through a model adapter without gradients.
6. Serialize adversarial trace, query count, metrics, and figures.
7. Reproduce Tables I and II and comparative plots against PSO / SBA baselines.

## Paper-to-Code Mapping
| Paper Component | Paper Ref | Reference Repo | Planned ANIMA Path | Owning PRD |
|---|---|---|---|---|
| Historical tensor `H_t`, target trace `X_t`, future `F_t`, prediction `P_t` | §III-A | `prediction/dataset/*.py` | `src/anima_def_dtp/types.py`, `src/anima_def_dtp/data/` | PRD-01 |
| Directional objective `d_int` | §IV-A Eq. 1 | `prediction/attack/loss.py` | `src/anima_def_dtp/objectives.py` | PRD-02 |
| Error objective `d_err` / ADE / FDE | §IV-A Eq. 2-4 | `prediction/attack/loss.py`, `prediction/evaluate/utils.py` | `src/anima_def_dtp/objectives.py`, `src/anima_def_dtp/evaluation/metrics.py` | PRD-02, PRD-04 |
| Binary adversarial criterion `c(.)` | §IV-B Eq. 5 | implicit in `boundary_attack.isadversarial` + thresholding | `src/anima_def_dtp/criteria.py` | PRD-02 |
| Decision-based optimization objective | §IV-C Eq. 7 | `prediction/attack/boundary_attack.py` | `src/anima_def_dtp/attack/boundary.py` | PRD-02 |
| Boundary walking algorithm | Algorithm 1 / §IV-C | `prediction/attack/boundary_attack.py` | `src/anima_def_dtp/attack/boundary.py` | PRD-02 |
| Multi-frame attack orchestration | Fig. 3 / §IV | `prediction/attack/backbone.py`, `prediction/model/utils.py` | `src/anima_def_dtp/attack/engine.py` | PRD-02 |
| Dataset windowing and JSON offline generator | repo implementation detail | `prediction/dataset/generate.py` | `src/anima_def_dtp/data/windows.py` | PRD-01 |
| Predictor abstraction | experimental setup | `prediction/model/base/interface.py` | `src/anima_def_dtp/predictors/base.py` | PRD-03 |
| Grip++ adapter | §V-A | `prediction/model/GRIP/` | `src/anima_def_dtp/predictors/grip.py` | PRD-03 |
| Trajectron++ adapter | §V-A | `prediction/model/Trajectron/` | `src/anima_def_dtp/predictors/trajectron.py` | PRD-03 |
| Table I intention metrics | Table I | `blackbox_utils.py` | `src/anima_def_dtp/evaluation/intention.py` | PRD-04 |
| Table II degradation metrics | Table II | `blackbox_utils.py`, `prediction/evaluate/` | `src/anima_def_dtp/evaluation/degradation.py` | PRD-04 |
| Comparative baselines PSO / SBA | §V-A / Fig. 4 | not isolated in repo | `src/anima_def_dtp/baselines/` | PRD-04 |
| Reproduction report and plots | Fig. 4-6 / Tables I-II | `image/` and evaluation scripts | `scripts/reproduce_paper.py`, `artifacts/reports/` | PRD-04 |
| Service layer | ANIMA packaging | none | `src/anima_def_dtp/service/` | PRD-05 |
| ROS2 bridge | ANIMA runtime | none | `src/anima_def_dtp/ros2/`, `launch/` | PRD-06 |
| Batch campaign / production export | ANIMA ops | none | `scripts/run_campaign.py`, `scripts/export_report.py` | PRD-07 |

## Files the Current Scaffold Must Replace
| Current File / Package | Problem | Planned Fix |
|---|---|---|
| `src/anima_bishamonten/` | wrong module identity | rename to `src/anima_def_dtp/` |
| `pyproject.toml` | wrong project name | rename package to `anima-def-dtp` |
| `configs/default.toml` | wrong codename and wrong arXiv ID | set `DEF-DTP` and `2603.26462` |
| `PRD.md` | stale scaffold content | replace with paper-accurate build plan |

## Reproduction Milestones
| Milestone | Exit Condition |
|---|---|
| M1 Paper-faithful attack core | Binary criteria, boundary walk, and predictor adapters produce adversarial traces on saved scenario windows |
| M2 Benchmark reproduction | Tables I and II reproduced within agreed tolerance on 100-scenario protocol |
| M3 ANIMA integration | Attack engine callable through API, Docker, and ROS2 wrappers |
| M4 Production hardening | Batch campaigns, audit logs, failure handling, and final validation report complete |
