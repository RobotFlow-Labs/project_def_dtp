# DEF-DTP

## Paper
**DTP-Attack: Trajectory Prediction Attack**
arXiv: https://arxiv.org/abs/2503.15832

## Module Identity
- Codename: DEF-DTP
- Domain: Defense
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_def_dtp/
├── pyproject.toml
├── configs/
├── src/anima_def_dtp/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── CLAUDE.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [DEF-DTP]
