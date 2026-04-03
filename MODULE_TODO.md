# DEF-DTP — Design & Implementation Checklist

## Paper: DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction
## ArXiv: 2603.26462
## Repo: https://github.com/eclipse-bot/DTP-Attack

---

## Phase 1: Scaffold + Verification
- [x] Project structure created
- [x] Correct paper PDF downloaded to papers/
- [x] Paper read and annotated
- [x] Reference repo cloned
- [ ] Reference demo runs successfully
- [x] Datasets identified and accessibility confirmed
- [x] PRD suite + tasks generated
- [x] Active scaffold files migrated from copied BISHAMONTEN identity

## Phase 2: Reproduce
- [x] Core attack engine implemented in `src/anima_def_dtp/`
- [x] Predictor protocol + adapters implemented
- [x] Evaluation pipeline implemented
- [ ] Metrics match paper (within tolerance)
- [x] Dual-runtime verified (Mac-first with CUDA-ready config)

## Phase 3: Adapt to Hardware
- [ ] ZED 2i data pipeline (if applicable)
- [ ] Unitree L2 LiDAR pipeline (if applicable)
- [ ] xArm 6 integration (if manipulation module)
- [ ] Real sensor inference test
- [ ] MLX inference port validated

## Phase 4: ANIMA Integration
- [ ] ROS2 bridge node
- [ ] Docker container builds and runs
- [x] API / CLI surface defined
- [ ] Integration test with stack: ATLAS

## Shenzhen Demo Readiness
- [ ] Demo script works end-to-end
- [ ] Demo data prepared
- [ ] Demo runs in < 30 seconds
- [ ] Demo visuals are compelling
