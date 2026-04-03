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
- **Phase**: Autopilot build in progress
- **MVP Readiness**: 38%
- **Accomplished**: PRD suite + task suite generated; correct paper identified; reference repo audited; package migrated to `anima_def_dtp`; Python 3.11 + uv lock established; attack core, adapter seams, CLI, and evaluation harness scaffold implemented and passing local tests
- **TODO**:
  1. Stage benchmark datasets and predictor checkpoints
  2. Replace injected/test backends with real Grip++ and Trajectron++ loading paths
  3. Wire sample-manifest discovery and 100-case protocol execution
  4. Reproduce Tables I and II
  5. Add Docker / ROS2 / API integration slices
- **Blockers**: Missing benchmark datasets and pretrained predictor checkpoints

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| nuScenes prediction export | — | https://www.nuscenes.org/nuscenes | exported prediction windows | Phase 2 |
| Apolloscape trajectory prediction | — | http://apolloscape.auto/trajectory.html | prediction windows | Phase 2 |

### Check shared volume first
/Volumes/AIFlowDev/RobotFlowLabs/datasets

### Download
`bash scripts/download_data.sh`

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
