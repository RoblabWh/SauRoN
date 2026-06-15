# Architecture

## Component map

```
main.py                         CLI entry point; all hyperparameters are argparse flags
│
├── Environment/Environment.py  RL-facing environment wrapper: step/reset API,
│   │                           observation extraction, reward calculation
│   └── Environment/Simulation.py
│       │                       Physics stepping, termination checks, level cycling
│       ├── Environment/LevelManager.py      loads/rotates levels, randomizes spawns
│       ├── Environment/SVGParser.py         parses SVG level files into geometry
│       ├── Environment/FlowField.py         geodesic distance fields (reward shaping)
│       └── Environment/Components/
│           ├── Robot.py        kinematics, lidar raycasting, observation frames
│           ├── Border.py       wall segments
│           └── Station.py      goal stations
│
├── PPO/
│   ├── Environment.py          train() / test() loops
│   ├── Algorithm.py            Actor, Critic, ActorCritic, PPO update
│   ├── SmallInput.py           default observation encoder
│   ├── BigInput.py             larger alternative encoder (--inputspace big)
│   └── CoolMemory.py           SwarmMemory: per-agent experience buffer
│
├── Visualization/RerunVisualizer.py   streams sim state to the rerun.io viewer
└── LevelEditor/ + leveleditor.py      browser-based visual level editor
```

There is no GUI toolkit; visualization is entirely through
[rerun.io](https://rerun.io) (`--visualization rerun`, or `none` for headless).

## Training loop

`PPO/Environment.py:train` drives everything:

1. **Reset.** `env.reset()` advances the `LevelManager` (which cycles through
   `--level_files` and randomizes robot spawns/goal assignments), rebuilds each
   robot's lidar state, and — when geodesic reward shaping is on — fetches the
   cached geodesic flow field for each robot's goal.
2. **Rollout.** For each step: the current policy samples actions for all
   active robots (`ppo.select_action`), `env.step(actions)` advances the
   simulation by one `--sim_time_step`, and each robot returns
   `(next_state, reward_dict, done, reached_goal)`. The named reward components
   are summed into a scalar and the transition is stored in `SwarmMemory`,
   keyed by agent id.
3. **Update.** As soon as the buffer holds at least `--update_experience`
   transitions (default 16384), `ppo.update(memory, batches)` runs — even
   mid-episode. The in-flight partial episodes are archived first
   (`memory.unroll_last_episode`) so no experience is lost, and collection
   continues afterwards with the same robots.
4. **Episode end.** The episode ends when every robot is done (crashed,
   arrived, or timed out) or `--steps` run out. The level manager is informed
   which robots reached their goals (used for curriculum/level rotation).

The update trigger is experience count, not episode count, so updates and
episode boundaries are decoupled.

## Multi-robot handling

A level can spawn several robots (a "swarm"); all share one policy. When a
robot finishes mid-episode it is deactivated and removed from the *compacted*
active lists — `active_ids` in the train loop maps each remaining list position
back to the robot's original slot so that experience keeps accumulating in the
right per-agent buffer (`PPO/Environment.py`, around line 111). The simulation
keeps running as long as at least one robot is active.

## Checkpointing

All checkpoints are written into `--ckpt_folder` as
`PPO_continuous_<name>.pth`:

| Checkpoint | When |
| --- | --- |
| `<model_name>_best` | every `--log_interval` episodes, when the goal-reached percentage beats the previous best |
| `<model_name>_solved` | when the goal-reached percentage reaches `--solved_percentage` (training stops) |
| `<model_name>_final` | on every training exit |
| `manuell` | manually: typing `s` + Enter in the terminal while training runs |

Note that the final checkpoint is written **on every exit** — a throwaway run
pointed at a real `--ckpt_folder` will overwrite real checkpoints (see
`CLAUDE.md` for the safe test-run recipe).

Metrics (losses, KL, clip fraction, explained variance, rewards, goal
percentage) go to TensorBoard event files in the same folder unless
`--tensorboard False`.
