# DEF-DTP — ANIMA Module

> **DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction**
> Paper: [arXiv:2603.26462](https://arxiv.org/abs/2603.26462)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
Defense

## Status
- [x] Paper read + ASSETS.md created
- [x] PRD-01 through PRD-07 drafted
- [x] Foundation package migration to `anima_def_dtp`
- [x] Core attack engine
- [x] Predictor adapters
- [x] Local CLI + evaluation harness scaffold
- [ ] Benchmark reproduction with real datasets/checkpoints
- [ ] API / ROS2 / production wrappers

## Quick Start
```bash
cd project_def_dtp
uv venv .venv --python python3.11 && uv sync --extra dev
uv run pytest tests/ -v
uv run python scripts/reproduce_paper.py --help
```

## Runtime Targets
- Local Mac development: MLX / CPU compatible scaffolding
- Future GPU server reproduction: CUDA-oriented runtime paths and dependency hooks are predeclared in `pyproject.toml` and `configs/default.toml`

## License
MIT — AIFLOW LABS LIMITED
