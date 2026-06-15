# SauRoN Documentation

SauRoN is a reinforcement-learning simulation in which differential-drive robots
learn lidar-based navigation: each robot must reach its assigned goal station in
a 2D level loaded from an SVG file, using only local sensing (lidar, goal
distance/bearing, own velocity). Training uses PPO (Proximal Policy
Optimization) with continuous actions.

## Contents

| Document | Covers |
| --- | --- |
| [01 — Architecture](01-architecture.md) | Components, training loop, data flow, multi-robot handling, checkpoints |
| [02 — Observation & Action Space](02-observation-action-space.md) | Input encodings, frame stacking, action semantics, robot kinematics, termination |
| [03 — PPO Algorithm](03-ppo-algorithm.md) | Network architecture, the PPO update, and all implementation tricks |
| [04 — Reward Function](04-reward-function.md) | Every reward term (active and inactive), the geodesic flow field, the custom-reward interface |
| [05 — Level Format](05-level-format.md) | The SVG level contract: walls, spawns, goals, parsing rules |
| [06 — CLI Reference](06-cli-reference.md) | All `main.py` flags with defaults |

Contributor-facing notes (test-run safety, repo conventions, known gotchas)
live in [`CLAUDE.md`](../CLAUDE.md) at the repository root.

## Quick start

```bash
uv sync                      # install dependencies (Python >= 3.12, managed with uv)
uv run main.py --mode train  # train with default settings (opens the rerun viewer)
uv run leveleditor.py        # visual level editor at http://127.0.0.1:8000
```
