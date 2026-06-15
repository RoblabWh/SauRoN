# CLI Reference

All settings are argparse flags on `main.py`; there is no config file. Run with
`uv run main.py <flags>`. Boolean flags take `True`/`False` (except
`--restore`, which is a plain switch).

> **Safety note (from `CLAUDE.md`):** `--mode train` **writes into
> `--ckpt_folder`** — `PPO_continuous_<model_name>_final.pth` on every exit and
> `_best.pth` during training, plus TensorBoard event files. A throwaway run
> pointed at a real folder silently overwrites real checkpoints. For test runs,
> isolate and run headless:
>
> ```bash
> uv run main.py --mode train --ckpt_folder /tmp/sauron_test \
>   --visualization none --tensorboard False \
>   --level_files <level.svg> --steps 200 --max_episodes 2
> ```

## Model & mode

| Flag | Default | Description |
| --- | --- | --- |
| `--ckpt_folder` | `./models/small` | folder for checkpoints and TensorBoard logs |
| `--model_name` | `model` | checkpoint base name (`PPO_continuous_<name>.pth`) |
| `--mode` | `train` | `train` or `test` (test loads `PPO_continuous_<model_name>.pth` and runs 100 deterministic episodes) |
| `--restore` | off | switch: load the checkpoint and continue training |

## Training

| Flag | Default | Description |
| --- | --- | --- |
| `--time_frames` | 4 | number of stacked past frames in the observation (not fully dynamic; the encoder hardcodes 4) |
| `--steps` | 5000 | max steps per episode |
| `--max_episodes` | inf | stop after this many episodes |
| `--update_experience` | 16384 | transitions collected before each PPO update |
| `--batches` | 32 | minibatches per epoch (minibatch size = update_experience / batches) |
| `--K_epochs` | 6 | optimization epochs per update |
| `--eps_clip` | 0.2 | PPO clipping ε |
| `--gamma` | 0.999 | discount factor; effective horizon ~1000 steps ≈ 40 m at the robot's ~0.04 m step length — must cover detour lengths in the larger levels |
| `--_lambda` | 0.97 | GAE λ (paired with the long γ horizon) |
| `--entropy_coeff` | 0.001 | entropy bonus coefficient in the actor loss |
| `--kl_early_stop` | True | abandon the whole update once approx. KL exceeds `--kl_target` |
| `--kl_target` | 0.02 | approximate-KL threshold for early stopping |
| `--lr` | 0.0003 | Adam learning rate (actor and critic) |
| `--lr_schedule` | `none` | `none` or `cosine` (cosine annealing, stepped per minibatch update) |
| `--lr_tmax` | 1000000 | T_max of the cosine schedule, in minibatch updates |
| `--torch_compile` | False | JIT-compile actor/critic with `torch.compile` (slow first update, faster afterwards) |
| `--inputspace` | `small` | observation encoder: `small` or `big` |
| `--image_size` | 256 | image side length for the (deprecated) image input style |
| `--action_std` | 0.5 | unused — the std is a learnable network parameter |

## Simulation

| Flag | Default | Description |
| --- | --- | --- |
| `--level_files` | all files in `svg/` | list of level filenames, relative to `svg/` (hardcoded curriculum lists in `main.py` can be uncommented instead) |
| `--sim_time_step` | 0.225 | seconds of simulated time per step |
| `--geodesic_reward` | True | shape the progress reward on geodesic (shortest obstacle-free path) distance instead of straight-line distance; observations are unaffected |
| `--flow_field_cell` | 0.1 | cell size in meters of the flow-field raster; must stay well below the narrowest passage width minus the robot diameter |

## Robot / sensors

| Flag | Default | Description |
| --- | --- | --- |
| `--number_of_rays` | 1081 | lidar rays per scan |
| `--field_of_view` | 270 | lidar field of view in degrees |
| `--lidar_d0` | 2.0 | scale of the inverse-depth lidar encoding `x = d0/(d0+d)`; an obstacle at `d0` meters encodes as 0.5 |
| `--goal_d0` | 10.0 | scale of the goal-distance encoding `x = d/(d+d0)`; 0 at the goal, 0.5 at `d0` meters, → 1 far away |
| `--has_pie_slice` | False | mount a reflector on the robot so other robots' lidar sees it (functional, not advised) |
| `--collide_other_targets` | False | robots collide with other robots' stations instead of passing through them (functional, not advised) |
| `--manually` | False | drive a robot with WASD for interactive collision checks (needs the rerun viewer — don't combine with `--visualization none`) |

## Visualization & managing

| Flag | Default | Description |
| --- | --- | --- |
| `--visualization` | `rerun` | `rerun`: stream to the rerun.io viewer; `none`: headless |
| `--rerun_save` | None | record to a `.rrd` file instead of spawning the live viewer (replay with `rerun <file>`) |
| `--rerun_memory_limit` | `25%` | in-RAM history budget of the live viewer (percentage or absolute, e.g. `2GB`) |
| `--viz_stride` | None | log the world state only every Nth sim step (default: 4 in train mode, 1 in test mode) |
| `--viz_lidar_stride` | 2 | log lidar point clouds only every Nth logged step |
| `--viz_lidar_decimation` | 3 | log only every Nth lidar ray |
| `--viz_flow_field` | False | debug overlay: render the geodesic flow-field raster and each robot's shortest path |
| `--step_delay` | 0.0 | sleep seconds per step in test mode (slows the sim down for watching) |
| `--tensorboard` | True | write TensorBoard event files into `--ckpt_folder` |
| `--print_interval` | 1 | print results every N episodes |
| `--log_interval` | 30 | episodes per TensorBoard log; also the window over which the solved percentage is computed |
| `--solved_percentage` | 0.99 | stop training (and save `_solved`) once this fraction of robots reaches the goal |

While training runs, typing `s` + Enter in the terminal saves the current
weights as `PPO_continuous_manuell.pth` at any time.
