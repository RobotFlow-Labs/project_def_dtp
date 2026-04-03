# PRD-05: API & Docker

> Module: DEF-DTP | Priority: P1  
> Depends on: PRD-03, PRD-04  
> Status: ⬜ Not started

## Objective
The paper-faithful attack engine is exposed through a deterministic service and a reproducible Docker image without altering benchmark semantics.

## Context (from paper)
The paper itself is not an API paper, but ANIMA needs a callable service surface for robustness campaigns, regression testing, and downstream orchestration. The service must remain a thin wrapper over the validated attack core.

**Paper reference:** indirectly grounded in §IV and §V because the service wraps the same attack / evaluation contracts.  
Key paper wording: `real-world scenarios` and `black-box access`.

## Acceptance Criteria
- [ ] FastAPI schemas accept a benchmark window and attack objective configuration
- [ ] Service can run:
  - single-case attack
  - single-case evaluation
  - report generation trigger
- [ ] Docker image contains all runtime dependencies, health checks, and mounted asset paths
- [ ] Smoke test: `curl /healthz` returns OK after container start
- [ ] Test: `uv run pytest tests/test_api.py tests/test_container_contract.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_def_dtp/service/schemas.py` | request / response models | §III / §IV | ~120 |
| `src/anima_def_dtp/service/api.py` | FastAPI endpoints | §IV / §V | ~220 |
| `src/anima_def_dtp/service/deps.py` | predictor registry and config loading | §V-A | ~120 |
| `Dockerfile` | container image | — | ~70 |
| `docker-compose.yml` | local orchestration | — | ~60 |
| `tests/test_api.py` | endpoint tests | — | ~120 |
| `tests/test_container_contract.py` | health and env contract tests | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- `POST /attack`: scenario window + attack objective + predictor selection
- `POST /evaluate`: run configuration + sample manifest

### Outputs
- JSON attack result bundle
- JSON evaluation summary
- path references to generated artifacts

### Algorithm
```python
app = FastAPI()

@app.post("/attack")
def run_attack(req: AttackRequest) -> AttackResponse:
    engine = build_attack_engine(req)
    result = engine.run_case(req.case)
    return AttackResponse.from_result(result)
```

## Dependencies
```toml
fastapi = ">=0.111"
uvicorn = ">=0.30"
orjson = ">=3.10"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| runtime config | small | `configs/default.toml` | local |
| model mounts | variable | `/mnt/forge-data/models/def_dtp` | pre-stage manually |
| dataset mounts | variable | `/mnt/forge-data/datasets/def_dtp` | pre-stage manually |

## Test Plan
```bash
uv run pytest tests/test_api.py tests/test_container_contract.py -v
docker build -t def-dtp:local .
```

## References
- Paper: §IV, §V
- Depends on: PRD-03, PRD-04
- Feeds into: PRD-06, PRD-07
