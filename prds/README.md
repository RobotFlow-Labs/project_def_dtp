# DEF-DTP PRD Suite

This directory contains the execution PRDs for reproducing and productizing the paper:

- `PRD-01-foundation.md`
- `PRD-02-core-model.md`
- `PRD-03-inference.md`
- `PRD-04-evaluation.md`
- `PRD-05-api-docker.md`
- `PRD-06-ros2-integration.md`
- `PRD-07-production.md`

## Execution Order
1. PRD-01 fixes scaffold drift and establishes typed data/config foundations.
2. PRD-02 implements the actual DTP-Attack method from the paper.
3. PRD-03 connects the attack engine to Grip++ and Trajectron++ predictors.
4. PRD-04 reproduces the paper's tables and comparison figures.
5. PRD-05, PRD-06, and PRD-07 wrap the paper-faithful core for ANIMA runtime use.

## Non-Negotiable Rule
Do not optimize for API or ROS2 convenience before the paper-faithful reproduction path is validated against the correct paper `2603.26462`.
