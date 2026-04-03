# PRD-03: Inference & Predictor Adapters

> Module: DEF-DTP | Priority: P0  
> Depends on: PRD-01, PRD-02  
> Status: ⬜ Not started

## Objective
The attack engine can query paper-matched predictor backbones through stable adapters and execute normal prediction, attack generation, and single-case CLI runs without exposing model internals to the core optimizer.

## Context (from paper)
The paper evaluates Grip++ and Trajectron++ on nuScenes and Apolloscape, plus map-enabled Trajectron++ on nuScenes. A faithful reproduction must preserve this experimental interface boundary while isolating predictor-specific loading and preprocessing from the paper's model-agnostic black-box attack core.

**Paper reference:** §V-A "Models & Datasets"  
Key paper wording: `Grip++`, `Trajectron++`, and `Trajectron++(m)`.

## Acceptance Criteria
- [ ] Predictor protocol abstracts `predict(window) -> predictions` without exposing gradients to the attack core
- [ ] Grip++ adapter loads checkpoints and produces attacked-agent future coordinates with expected shapes
- [ ] Trajectron++ adapter loads checkpoints, supports map and non-map modes, and produces attacked-agent future coordinates with expected shapes
- [ ] CLI supports:
  - `normal`
  - `attack`
  - `evaluate-case`
- [ ] Test: `uv run pytest tests/test_predictor_protocol.py tests/test_grip_adapter.py tests/test_trajectron_adapter.py tests/test_cli.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_def_dtp/predictors/base.py` | predictor interface and request / response contracts | §V-A | ~100 |
| `src/anima_def_dtp/predictors/grip.py` | Grip++ adapter | §V-A | ~200 |
| `src/anima_def_dtp/predictors/trajectron.py` | Trajectron++ adapter | §V-A | ~240 |
| `src/anima_def_dtp/cli.py` | command-line entry for attack and eval | §V-A | ~180 |
| `tests/test_predictor_protocol.py` | protocol conformance | — | ~80 |
| `tests/test_grip_adapter.py` | Grip shape and loading tests | — | ~120 |
| `tests/test_trajectron_adapter.py` | Trajectron shape and loading tests | — | ~140 |
| `tests/test_cli.py` | smoke tests for CLI surface | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- `ScenarioWindow`: typed benchmark case from PRD-01
- `PredictorConfig`: checkpoint path, dataset name, map mode, device

### Outputs
- `PredictionBundle.predict_traces`: `dict[str, Tensor[LO, 2]]`
- `PredictionBundle.observe_traces`: `dict[str, Tensor[LI, 2]]`
- `PredictionBundle.future_traces`: `dict[str, Tensor[LO, 2]]`

### Algorithm
```python
# Paper Section V-A — model adapters
class PredictorAdapter(Protocol):
    obs_length: int
    pred_length: int

    def predict(self, window: ScenarioWindow, perturbation: Tensor | None = None) -> PredictionBundle:
        ...

class GripAdapter:
    # Upstream predicted tensor comment: (N, 2, T, V)
    def predict(self, window, perturbation=None) -> PredictionBundle:
        ...

class TrajectronAdapter:
    def predict(self, window, perturbation=None) -> PredictionBundle:
        ...
```

## Dependencies
```toml
torch = ">=2.0"
numpy = ">=1.26"
typer = ">=0.12"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| Grip++ checkpoints | not stated | `/mnt/forge-data/models/def_dtp/grip_*` | reproduce or stage manually |
| Trajectron++ checkpoints | not stated | `/mnt/forge-data/models/def_dtp/trajectron_*` | reproduce or stage manually |
| nuScenes maps | not stated | `/mnt/forge-data/datasets/def_dtp/nuscenes/maps` | via nuScenes download |

## Test Plan
```bash
uv run pytest tests/test_predictor_protocol.py tests/test_grip_adapter.py tests/test_trajectron_adapter.py tests/test_cli.py -v
```

## References
- Paper: §V-A
- Reference impl: `repositories/DTP-Attack/prediction/model/GRIP/`, `repositories/DTP-Attack/prediction/model/Trajectron/`
- Depends on: PRD-01, PRD-02
- Feeds into: PRD-04, PRD-05
