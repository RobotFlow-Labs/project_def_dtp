# DEF-DTP — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: DTP-Attack: Trajectory Prediction Attack.

## 1. Working Rules
- Work only inside `project_def_dtp/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[DEF-DTP]`
- Stage only `project_def_dtp/` files
- VERIFY THE PAPER BEFORE BUILDING ANYTHING

## 2. The Paper
- **Title**: DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction
- **ArXiv**: 2603.26462
- **Link**: https://arxiv.org/abs/2603.26462
- **Repo**: https://github.com/eclipse-bot/DTP-Attack
- **Compute**: MLX-OK
- **Verification status**: Correct ArXiv ID ✅ | Repo ✅ | Paper read ✅

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: All PRDs complete, pre-training verification needed
- **MVP Readiness**: 85%
- **Accomplished**:
  - PRD-01: Foundation & Config — typed contracts, constants, data windowing ✅
  - PRD-02: Core Attack Engine — objectives, criteria, boundary walker ✅
  - PRD-03: Inference & Adapters — predictor protocol, Grip++, Trajectron++, CLI ✅
  - PRD-04: Evaluation & Reproduction — metrics, protocol, baselines, reproduce_paper.py ✅
  - PRD-05: API & Docker — FastAPI service, Dockerfile.cuda/mlx, docker-compose ✅
  - PRD-06: ROS2 Integration — messages, node, launch file ✅
  - PRD-07: Production Hardening — campaign runner, export report, runtime guards ✅
  - ANIMA infra: anima_module.yaml, Dockerfile.serve, docker-compose.serve.yml, serve.py ✅
  - Stale anima_bishamonten package removed ✅
  - 33 tests passing, ruff clean ✅
- **TODO**:
  1. Wire real predictor backends (Grip++, Trajectron++) with actual checkpoints
  2. Stage/verify nuScenes prediction exports at `/mnt/forge-data/datasets/nuscenes/`
  3. Stage Apolloscape trajectory data
  4. Reproduce Tables I and II on real data
  5. Run full 100-case campaign with real predictors
  6. Export: pth → safetensors → ONNX → TRT fp16 → TRT fp32
  7. Push to HuggingFace: ilessio-aiflowlab/project_def_dtp
- **Blockers**: Pretrained Grip++ and Trajectron++ checkpoints are MISSING — need training or download

## 4. Datasets
### Available on server
| Dataset | Path | Status |
|---------|------|--------|
| nuScenes | /mnt/forge-data/datasets/nuscenes/ | AVAILABLE |
| KITTI | /mnt/forge-data/datasets/kitti/ | AVAILABLE |
| COCO | /mnt/forge-data/datasets/coco/ | AVAILABLE |
| DINOv2 ViT-B/14 | /mnt/forge-data/models/dinov2_vitb14_pretrain.pth | AVAILABLE |

### Required but not staged
| Asset | Path | Status |
|-------|------|--------|
| Apolloscape trajectory | /mnt/forge-data/datasets/def_dtp/apolloscape/ | MISSING |
| Grip++ checkpoints | /mnt/forge-data/models/def_dtp/grip_*/ | MISSING |
| Trajectron++ checkpoints | /mnt/forge-data/models/def_dtp/trajectron_*/ | MISSING |

### Output Paths
- Checkpoints: /mnt/artifacts-datai/checkpoints/project_def_dtp/
- Logs: /mnt/artifacts-datai/logs/project_def_dtp/

## 5. Hardware
- ZED 2i stereo camera: Available
- Unitree L2 3D LiDAR: Available
- xArm 6 cobot: Pending purchase
- Mac Studio M-series: MLX dev
- 8x RTX 6000 Pro Blackwell: GCloud

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | Correct paper verified as arXiv 2603.26462; PRDs and task breakdown created |
| 2026-04-03 | Codex | Built Python 3.11 uv scaffold, typed contracts, attack core, predictor adapters, CLI, and evaluation harness; local tests and Ruff passing |
| 2026-04-03 | Opus 4.6 | Created data module (windows.py), built PRD-05 (API/Docker), PRD-06 (ROS2), PRD-07 (Production), ANIMA infra; removed stale bishamonten; 33 tests passing, ruff clean |
