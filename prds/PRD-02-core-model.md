# PRD-02: Core Attack Engine

> Module: DEF-DTP | Priority: P0  
> Depends on: PRD-01  
> Status: ⬜ Not started

## Objective
The project contains a paper-faithful implementation of DTP-Attack: attack objectives, binary adversarial criteria, and the decision-based boundary-walking optimizer from Algorithm 1.

## Context (from paper)
The contribution is the attack method itself: a decision-based black-box formulation that uses binary criteria instead of score-based optimization, then walks the adversarial boundary through orthogonal and forward steps.

**Paper reference:** §IV-A, §IV-B, §IV-C, Algorithm 1  
Key paper wording: `binary decision outputs` and `boundary walking algorithm`.

## Acceptance Criteria
- [ ] Intention objectives implement left / right / front / rear directional offsets
- [ ] Degradation objectives implement ADE and FDE thresholding
- [ ] Binary criterion `c(.)` matches the paper's thresholded objective activation
- [ ] Boundary walker implements random adversarial initialization, orthogonal step, forward step, and adaptive `delta` / `epsilon`
- [ ] Test: `uv run pytest tests/test_objectives.py tests/test_criteria.py tests/test_boundary.py -v` passes
- [ ] On a deterministic toy predictor, the boundary walker reduces `D(X_t, X_t*)` while maintaining adversarial validity

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_def_dtp/objectives.py` | `d_int`, `d_err`, ADE, FDE, directional offsets | §IV-A Eq. 1-4 | ~180 |
| `src/anima_def_dtp/criteria.py` | binary adversarial criterion `c(.)` | §IV-B Eq. 5 | ~100 |
| `src/anima_def_dtp/attack/boundary.py` | Algorithm 1 implementation | Algorithm 1 / §IV-C | ~260 |
| `src/anima_def_dtp/attack/engine.py` | multi-frame attack orchestration and result bundle | Fig. 3 / §IV | ~180 |
| `tests/test_objectives.py` | objective correctness checks | — | ~120 |
| `tests/test_criteria.py` | criterion thresholding checks | — | ~100 |
| `tests/test_boundary.py` | boundary walker convergence checks | — | ~160 |

## Architecture Detail (from paper)
### Inputs
- `X_t`: `Tensor[LI, 2]`  # original observed trajectory for the attacked agent
- `F_t`: `Tensor[LO, 2]`  # ground truth future
- `P_t`: `Tensor[LO, 2]`  # predicted future for attacked agent
- `criterion_type`: one of `{left, right, front, rear, ade, fde}`

### Outputs
- `AttackResult.perturbation`: `Tensor[LI, 2]`
- `AttackResult.query_count`: `int`
- `AttackResult.is_adversarial`: `bool`
- `AttackResult.distance_to_original`: `float`

### Algorithm
```python
# Paper Section IV-C — Algorithm 1
class BoundaryWalker:
    def __init__(self, delta: float = 1.0, epsilon: float = 0.1, max_iter: int = 1000):
        self.delta = delta
        self.epsilon = epsilon
        self.max_iter = max_iter

    def run(self, x_orig, predictor, criterion):
        x_adv = self._random_adversarial_init(x_orig, predictor, criterion)
        x_adv = self._forward_initialize(x_orig, x_adv, predictor, criterion)
        for _ in range(self.max_iter):
            x_adv = self._orthogonal_step(x_orig, x_adv, predictor, criterion)
            x_adv = self._forward_step(x_orig, x_adv, predictor, criterion)
        return x_adv
```

## Dependencies
```toml
numpy = ">=1.26"
torch = ">=2.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| Typed scenario windows | per case | generated in memory from `src/anima_def_dtp/data/windows.py` | local |
| Paper thresholds | tiny config | `configs/default.toml` and `src/anima_def_dtp/constants.py` | local |

## Test Plan
```bash
uv run pytest tests/test_objectives.py tests/test_criteria.py tests/test_boundary.py -v
```

## References
- Paper: §IV-A to §IV-C, Algorithm 1
- Reference impl: `repositories/DTP-Attack/prediction/attack/loss.py`, `repositories/DTP-Attack/prediction/attack/boundary_attack.py`, `repositories/DTP-Attack/prediction/attack/backbone.py`
- Depends on: PRD-01
- Feeds into: PRD-03, PRD-04
