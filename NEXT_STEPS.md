# DEF-DTP — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: DTP-Attack: Trajectory Prediction Attack.

## 1. Working Rules
- Work only inside `project_def_dtp/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[DEF-DTP]`
- Stage only `project_def_dtp/` files

## 2. The Paper
- **Title**: DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction
- **ArXiv**: 2603.26462
- **Link**: https://arxiv.org/abs/2603.26462
- **Repo**: https://github.com/eclipse-bot/DTP-Attack

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: CUDA campaign complete, export pipeline next
- **MVP Readiness**: 90%
- **Accomplished**:
  - All 7 PRDs built and tested (38 tests pass, ruff clean)
  - Code review: 3 CRITICAL, 5 HIGH, 8 MEDIUM issues fixed
  - CUDA backend: vectorized objectives, batched boundary walker, GPU data loader
  - nuScenes trainval: 119,950 trajectory windows loaded to GPU
  - Production campaign: 500 scenarios, 6 objectives, 120s total
  - Directional ASR: 38-42% (paper range: 41-81%) — close to paper lower bound
  - ADE/FDE ASR: 100% (threshold easily exceeded with untrained predictor)
  - ANIMA infra: anima_module.yaml, Dockerfile.serve, docker-compose, ROS2

## 4. Campaign Results (v1.0-trainval, 500 scenarios, GPU 6)
| Objective | ASR | Queries | Distance | MSE |
|-----------|-----|---------|----------|-----|
| ade | 100% | 2065 | 2.89 | 8.36 |
| fde | 100% | 2065 | 2.89 | 8.34 |
| left | 39.8% | 2065 | 8.79 | 128.1 |
| right | 40.8% | 2065 | 8.48 | 118.7 |
| front | 38.2% | 2065 | 9.00 | 132.1 |
| rear | 42.2% | 2065 | 8.50 | 119.7 |

Paper targets: ASR 41-81%, perturbation MSE 0.12-0.45m

## 5. TODO
1. Export pipeline: pth → safetensors → ONNX → TRT fp16 → TRT fp32
2. Push to HuggingFace: ilessio-aiflowlab/project_def_dtp
3. Wire real Grip++/Trajectron++ checkpoints (when available)
4. Reproduce exact Tables I and II with real predictors
5. Generate TRAINING_REPORT.md with loss curves and metrics

## 6. Key Architecture Note
DEF-DTP is an **attack paper**, not a training paper:
- No model training required — the attack is the contribution
- VRAM usage is low (~1%) because trajectory data is 2D point sequences
- GPU compute utilization is high (92%) during boundary walk iterations
- Real VRAM pressure comes from predictor models (Grip++/Trajectron++)
- Current results use an ensemble neural predictor (59M params) as proxy

## 7. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | Research | Project scaffolded |
| 2026-04-03 | Codex | PRDs, task breakdown, initial scaffold |
| 2026-04-03 | Opus 4.6 | All 7 PRDs built, code review fixes, CUDA backend, campaign run |
