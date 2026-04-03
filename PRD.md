# DEF-DTP: DTP-Attack — Implementation PRD
## ANIMA Wave-7 Defense Module

**Status:** PRD suite generated  
**Version:** 0.2  
**Date:** 2026-04-03  
**Paper:** DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction  
**Correct Paper Link:** https://arxiv.org/abs/2603.26462  
**Incorrect Scaffold Link To Remove:** https://arxiv.org/abs/2503.15832  
**Repo:** https://github.com/eclipse-bot/DTP-Attack  
**Functional Name:** `def_dtp`  
**Target Python Package:** `anima-def-dtp` / `src/anima_def_dtp/`  
**Stack:** ATLAS

## 1. Executive Summary
DEF-DTP reproduces DTP-Attack as a paper-faithful, decision-based black-box attack harness for trajectory prediction systems, then packages it for ANIMA as a robustness-evaluation component. The core deliverable is not a novel model training stack; it is a reproducible attack pipeline that turns binary adversarial criteria into minimal-perturbation trajectory attacks against Grip++ and Trajectron++ on nuScenes and Apolloscape.

The immediate objective is to match the paper's attack behavior and benchmark ranges before any platform adaptation. The current scaffold is materially wrong in two ways: it references the wrong paper (`2503.15832`) and carries copied `BISHAMONTEN` naming. PRD-01 fixes that foundation so later work is not built on bad metadata.

## 2. Paper Verification Status
- [x] Correct arXiv paper verified: `2603.26462`, submitted March 27, 2026
- [x] GitHub repo confirmed accessible and already present in `repositories/DTP-Attack/`
- [x] Paper read completely
- [ ] Reference repo executed end-to-end on staged data
- [ ] Datasets staged in ANIMA shared volume
- [ ] Predictor checkpoints staged in ANIMA shared volume
- [ ] Tables I and II reproduced locally
- [x] Key red flag identified: stale scaffold points to unrelated paper `2503.15832`
- **Verdict:** VALID PAPER / INVALID SCAFFOLD METADATA / REPRODUCTION WORK REQUIRED

## 3. What We Take From The Paper
- The exact attack framing: binary adversarial criteria over trajectory outputs, not score-based optimization.
- The exact optimization style: random adversarial initialization followed by orthogonal and forward boundary-walking steps.
- The exact evaluation scope: nuScenes and Apolloscape, 100 random samples per dataset, Grip++ and Trajectron++ predictors, plus Trajectron++(map) on nuScenes.
- The exact thresholding regime:
  - `theta_err_ADE = 7.5 m` for nuScenes and `3.5 m` for Apolloscape
  - `theta_err_FDE = 17.5 m` for nuScenes and `7.5 m` for Apolloscape
  - `theta_int_lateral = 2.0 m`
  - `theta_int_longitudinal = 3.0 m`
- The exact algorithm constants from Algorithm 1 and §V-A:
  - `delta = 1.0`
  - `epsilon = 0.1`
  - orthogonal adjustment `x0.95`
  - forward adjustment `x0.9`
  - `max_iter = 1000`
- The exact reported targets:
  - intention ASR range `41%` to `81%`
  - degradation increase `1.9x` to `4.2x`
  - perturbation MSE `0.12 m` to `0.45 m`

## 4. What We Skip
- FQA as a first-class reproduction target. It exists in the repo but is not part of the paper's main reported benchmark tables.
- New defensive methods. This module first reproduces the attack faithfully; defense research comes after parity.
- Training new predictor architectures from scratch inside the first milestone. The attack is the paper contribution; backbone retraining is only a checkpoint acquisition fallback.
- Sensor-specific adaptation in the first milestone. Real-world ZED/LiDAR ingestion belongs after benchmark parity.

## 5. What We Adapt
- Replace repo-global scripts and hard-coded paths with an ANIMA package and typed configs.
- Rename copied scaffold identity from `BISHAMONTEN` to `DEF-DTP`.
- Separate paper-faithful CUDA parity from later MLX/runtime adaptation.
- Turn one-off repo scripts into reusable ANIMA components:
  - typed data windows
  - attack objectives and criteria
  - predictor adapter layer
  - evaluation harness
  - API / Docker / ROS2 wrappers

## 6. Architecture
### 6.1 Canonical Inputs
- Historical scenario tensor `H_t`: `Tensor[LI, N, D]`
- Target observed trace `X_t`: `Tensor[LI, 2]`
- Ground-truth future trace `F_t`: `Tensor[LO, 2]`
- Context `I_t`: optional map / scene metadata

### 6.2 Canonical Outputs
- Adversarial trace `X_t*`: `Tensor[LI, 2]`
- Binary adversarial decision `c(.) in {0, 1}`
- Prediction outputs `P_t`: `Tensor[LO, N, 2]`
- Evaluation artifact bundle:
  - per-case metrics
  - query count
  - perturbation magnitude
  - reproduction tables / figures

### 6.3 Planned Module Layout
- `src/anima_def_dtp/config.py`
- `src/anima_def_dtp/types.py`
- `src/anima_def_dtp/data/`
- `src/anima_def_dtp/objectives.py`
- `src/anima_def_dtp/criteria.py`
- `src/anima_def_dtp/attack/boundary.py`
- `src/anima_def_dtp/attack/engine.py`
- `src/anima_def_dtp/predictors/`
- `src/anima_def_dtp/evaluation/`
- `src/anima_def_dtp/service/`
- `src/anima_def_dtp/ros2/`
- `scripts/reproduce_paper.py`
- `scripts/run_campaign.py`

### 6.4 Predictor Support
- In-scope for parity:
  - Grip++
  - Trajectron++
  - Trajectron++ with maps on nuScenes
- Out of scope for milestone 1:
  - FQA parity
  - new backbones

## 7. Implementation Phases
### Phase 1 — Foundation Correction + Paper-Faithful Core
- [ ] Correct arXiv / codename / package identity across project files
- [ ] Implement typed trajectory windows, thresholds, and attack criteria
- [ ] Implement decision-based boundary walker matching Algorithm 1

### Phase 2 — Predictor Adapters + Benchmark Reproduction
- [ ] Load Grip++ and Trajectron++ through stable ANIMA interfaces
- [ ] Reproduce Table I and Table II protocols on 100-sample subsets
- [ ] Record parity gaps against the paper

### Phase 3 — ANIMA Productization
- [ ] FastAPI service + Docker image
- [ ] ROS2 bridge and launch flow
- [ ] Batch campaign runner and exportable reports

## 8. Datasets
| Dataset | Size | URL | Phase Needed |
|---|---:|---|---|
| nuScenes | not stated in paper | https://www.nuscenes.org/nuscenes | Phase 2 |
| Apolloscape trajectory prediction | not stated in paper | http://apolloscape.auto/trajectory.html | Phase 2 |
| NGSIM helper pipeline | optional / non-paper | https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm | Optional |

## 9. Dependencies on Other Wave Projects
| Needs output from | What it provides |
|---|---|
| None required for core reproduction | DEF-DTP is self-contained once datasets and checkpoints are staged |

## 10. Success Criteria
- The module loads the correct paper metadata everywhere (`2603.26462`) and removes copied `BISHAMONTEN` naming from package/config planning.
- Grip++ and Trajectron++ predictor adapters execute the decision-based attack loop on saved benchmark windows without gradient access.
- Reproduction targets are achieved within agreed tolerance:
  - intention ASR stays inside the paper's `41%` to `81%` range
  - attack-induced ADE increase lands within `10%` of the paper's `1.9x` to `4.2x` range
  - perturbation MSE remains inside the paper's `0.12 m` to `0.45 m` range
- API and ROS2 wrappers call the same core engine without changing attack semantics.

## 11. Risk Assessment
- The biggest current risk is metadata corruption from the copied scaffold. If not corrected first, every downstream artifact will cite the wrong paper.
- The paper does not publish checkpoint URLs, so reproduction may block on backbone checkpoint acquisition or retraining.
- The reference repo hard-codes CUDA. A direct MLX adaptation before CUDA parity would create a moving target and break paper-faithful validation.
- The paper's physical-constraint statement references the SA-Attack vehicle dynamics model, but the published repo mostly operationalizes stealth through proximity and evaluation, so parity must be defined by observed behavior, not by over-interpreting missing implementation detail.

## 12. Build Plan
| PRD# | Deliverable | Purpose | Status |
|---|---|---|---|
| [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | correct metadata, package, schemas, datasets | ⬜ |
| [PRD-02](prds/PRD-02-core-model.md) | Core Attack Engine | objectives, criteria, boundary walking | ⬜ |
| [PRD-03](prds/PRD-03-inference.md) | Inference & Adapters | predictor loading, CLI, attack execution | ⬜ |
| [PRD-04](prds/PRD-04-evaluation.md) | Evaluation & Reproduction | tables, figures, baselines, parity report | ⬜ |
| [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | service exposure and reproducible containerization | ⬜ |
| [PRD-06](prds/PRD-06-ros2-integration.md) | ROS2 Integration | runtime bridge for ANIMA stacks | ⬜ |
| [PRD-07](prds/PRD-07-production.md) | Production Hardening | campaigns, exports, operational validation | ⬜ |

## 13. Immediate Execution Order
1. Fix the project identity and config drift.
2. Implement the binary criteria and boundary-walking core.
3. Load Grip++ and Trajectron++ via stable predictor adapters.
4. Reproduce Tables I and II before any MLX or sensor adaptation.
