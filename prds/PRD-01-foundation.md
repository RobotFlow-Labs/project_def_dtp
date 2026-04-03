# PRD-01: Foundation & Config

> Module: DEF-DTP | Priority: P0  
> Depends on: None  
> Status: ⬜ Not started

## Objective
The repo is correctly identified as DEF-DTP, the paper metadata is fixed to `2603.26462`, and the project has typed config, trajectory schemas, and dataset windowing primitives that match the paper and upstream repo.

## Context (from paper)
The paper defines the problem around historical windows `H_t`, target observations `X_t`, contextual information `I_t`, and future trajectories `F_t`, with dataset-specific `LI` / `LO` choices for nuScenes and Apolloscape. That makes a correct data contract the first blocker to any faithful implementation.

**Paper reference:** §III-A "Trajectory Prediction Formulation", §V-A "Experiment Set-up"  
Key paper wording: `historical state information spanning LI previous timesteps` and `nuScenes parameters to LI = 4 and LO = 12`.

## Acceptance Criteria
- [ ] `pyproject.toml`, `configs/default.toml`, and package naming are aligned to `DEF-DTP` and `2603.26462`
- [ ] Typed trajectory data models cover `H_t`, `X_t`, `F_t`, object masks, and optional map context
- [ ] Dataset windowing reproduces the repo's offline JSON framing for nuScenes and Apolloscape
- [ ] Test: `uv run pytest tests/test_config.py tests/test_types.py tests/test_windows.py -v` passes
- [ ] Benchmark constants for `LI`, `LO`, and thresholds are centralized in config and match the paper

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_def_dtp/config.py` | Pydantic settings for paper IDs, thresholds, paths | §V-A | ~140 |
| `src/anima_def_dtp/types.py` | Typed scenario, object, and trajectory containers | §III-A | ~180 |
| `src/anima_def_dtp/constants.py` | Dataset-specific `LI`, `LO`, thresholds | §V-A | ~80 |
| `src/anima_def_dtp/data/windows.py` | Offline JSON window slicing and sample manifest logic | §III-A | ~180 |
| `tests/test_config.py` | Metadata and settings validation | — | ~80 |
| `tests/test_types.py` | Shape and contract validation | — | ~120 |
| `tests/test_windows.py` | Dataset window generation parity checks | — | ~140 |
| `configs/default.toml` | Corrected module metadata | §V-A | ~40 |

## Architecture Detail (from paper)
### Inputs
- `H_t`: `Tensor[LI, N, D]`  # historical state tensor
- `X_t`: `Tensor[LI, 2]`  # target-agent observed coordinates
- `F_t`: `Tensor[LO, 2]`  # ground-truth future coordinates
- `I_t`: optional map or scene metadata

### Outputs
- `ScenarioWindow`: typed container carrying all trajectories and masks
- `PredictorRequest`: normalized predictor-facing input

### Algorithm
```python
# Paper Section III-A
from pydantic import BaseModel

class ObjectTrack(BaseModel):
    object_id: str
    observe_trace: list[list[float]]   # [LI, 2]
    future_trace: list[list[float]]    # [LO, 2]
    observe_mask: list[int]
    future_mask: list[int]
    object_type: int

class ScenarioWindow(BaseModel):
    observe_length: int
    predict_length: int
    time_step: float
    objects: dict[str, ObjectTrack]
    map_name: str | None = None
    scene_name: str | None = None
```

## Dependencies
```toml
pydantic = ">=2.7"
pydantic-settings = ">=2.2"
numpy = ">=1.26"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| nuScenes prediction exports | not stated | `/mnt/forge-data/datasets/def_dtp/nuscenes/prediction_{train,val,test}` | registration required at `https://www.nuscenes.org/nuscenes` |
| Apolloscape prediction exports | not stated | `/mnt/forge-data/datasets/def_dtp/apolloscape/prediction_{train,val,test}` | `http://apolloscape.auto/trajectory.html` |
| 100-sample manifests | small text files | `artifacts/sample_manifests/` | generated locally |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_types.py tests/test_windows.py -v
uv run ruff check src/ tests/
```

## References
- Paper: §III-A, §V-A
- Reference impl: `repositories/DTP-Attack/prediction/dataset/`
- Feeds into: PRD-02, PRD-03
