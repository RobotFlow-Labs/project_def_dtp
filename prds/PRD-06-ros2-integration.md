# PRD-06: ROS2 Integration

> Module: DEF-DTP | Priority: P1  
> Depends on: PRD-05  
> Status: ⬜ Not started

## Objective
DEF-DTP can be invoked as a ROS2 component for ANIMA runtime experiments, replaying benchmark or live-proxied trajectory windows through the same validated attack service.

## Context (from paper)
The paper is benchmark-centric, not ROS-native. ROS2 is an ANIMA delivery layer that must preserve the same case schema, predictor selection, and attack objective semantics established in the paper-faithful implementation.

**Paper reference:** derived from the runtime use of the same `X_t`, `H_t`, `I_t`, and adversarial criteria.  
Key paper wording: `trajectory prediction systems are critical for autonomous vehicle safety`.

## Acceptance Criteria
- [ ] ROS2 node accepts trajectory windows and attack requests over typed topics or service calls
- [ ] Launch file starts the node with configurable predictor and asset paths
- [ ] Output topic publishes perturbation, query count, and attack metrics
- [ ] Replay test works on stored benchmark windows
- [ ] Test: `uv run pytest tests/test_ros2_contract.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_def_dtp/ros2/messages.py` | Python-side message contracts | §III-A | ~100 |
| `src/anima_def_dtp/ros2/node.py` | ROS2 wrapper node | §IV / §V | ~220 |
| `launch/def_dtp.launch.py` | launch entrypoint | — | ~80 |
| `tests/test_ros2_contract.py` | serialization and replay contract tests | — | ~120 |

## Architecture Detail (from paper)
### Inputs
- `TrajectoryWindowMsg`
- `AttackConfigMsg`

### Outputs
- `AttackResultMsg`
- optional `EvaluationSummaryMsg`

### Algorithm
```python
class DefDtpNode(Node):
    def __init__(self):
        super().__init__("def_dtp")
        self.subscription = self.create_subscription(...)
        self.publisher = self.create_publisher(...)
```

## Dependencies
```toml
rclpy = "system"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---:|---|---|
| benchmark replay cases | small JSON set | `artifacts/replay_cases/` | generated locally |

## Test Plan
```bash
uv run pytest tests/test_ros2_contract.py -v
python -m py_compile launch/def_dtp.launch.py
```

## References
- Paper: §III-A, §IV, §V
- Depends on: PRD-05
- Feeds into: PRD-07
