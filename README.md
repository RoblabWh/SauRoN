# SauRoN

**SauRoN** (Simulation for autonomous robot navigation) is a simulation environment for reinforcement learning algorithms, which was developed within the research project ai arena. It is intended to help students to better understand and apply these methods during their studies.

The simulation is currently being further developed as part of a software project.

![Alt text](images/trained.gif?raw=true "tunnel level")

Agents in the simulation should learn to avoid obstacles and navigate towards their specific goals. They are using **laserscans**, **distance to goal**, **velocity** (linear and angular) and **orientation** as inputs for the neural network and the network outputs the new velocity commands.

## Installation

Clone the repository:

```
git clone git@github.com:RoblabWh/SauRoN.git
```

Install [uv](https://docs.astral.sh/uv/) on your system:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment (uv downloads a matching Python and all dependencies, including PyTorch with CUDA 12.8 support):

```
uv sync
```

For MPI-based multi-environment training you additionally need a system MPI implementation (e.g. `sudo apt install libopenmpi-dev`) and the optional extra:

```
uv sync --extra mpi
```

Now you should be ready to run your simulation.

## Levels

There are some levels provided as svg Files. Red dots mark possible agent positions while green dots mark possible goals for the agents. 

![Alt text](svg/tunnel.svg?raw=true "tunnel level")

## Usage

**Currently only PPO is implemented** 

The training can be started either using MPI for Multi Environment or without MPI to only train on a single level.

Run `main.py` via uv with desired arguments:

- `--mode` choose either `train` or `test` for Training or running a trained model
- `--ckpt_folder` Checkpoint Folder for the loaded and new models. Default : `./models`

There are several other options for the hyperparameters, vizualization and robot settings you can look up in the `main.py`.

The lidar scan is fed to the network as smooth inverse depth, `x = d0 / (d0 + d)` (`--lidar_d0`, default 2 m): close obstacles encode near 1, free space fades to 0, with no maximum sensor range — so arbitrarily large levels work without saturating the scan. The distance to the goal is squashed the same way via `--goal_d0`. Note that checkpoints trained with the previous linear distance encoding (including the formerly bundled pretrained model) are not compatible and need retraining.

The progress reward shapes on **geodesic distance** (shortest obstacle-free path to the goal) instead of straight-line distance: each level is rasterized to a grid (`--flow_field_cell`, default 0.1 m), inflated by the robot radius and flooded from every goal once per level (cached across episodes). This removes the local minima that made agents get stuck in alleys or push into walls when the goal lies behind them. The map knowledge is used at training time only — observations stay map-free, so the deployed policy needs no map. Disable with `--geodesic_reward False` to compare against Euclidean shaping, and use `--viz_flow_field True` to overlay the distance raster, the wall/inflation cells and each robot's shortest path in the rerun viewer for debugging.

To train run:

```uv run main.py --mode train```

To train with MPI (requires `uv sync --extra mpi`, see Installation):

```mpirun -n N_PROC ./.venv/bin/python main.py --mode train```


## Params

### Training Parameters:
`--restore`: If set to `True`, restores and continues training from previous checkpoint. **Default: `False`**

`--time_frames`: The number of timeframes (past states) that will be analyzed by the neural net. **Default: 4**

`--steps`: The number of steps in the environment per episode. **Default: 2500**

`--max_episodes`: The maximum number of episodes to run. **Default: `inf`**

`--update_experience`: How many experiences to use to update the policy. **Default: 20000**

`--batches`: The number of batches to use. **Default: 2**

`--action_std`: The constant standard deviation for the action distribution (Multivariate Normal). **Default: 0.5**

`--K_epochs`: The number of times to update the policy. **Default: 7**

`--eps_clip`: The epsilon value for p/q clipping. **Default: 0.2**

`--gamma`: The discount factor. **Default: 0.99**

`--lr`: The learning rate. Default: **0.0003**

`--input_style`: Choose between using images (image) or laser readings (laser). **Default: `laser`**

`--image_size`: The size of the image that is input to the neural net. **Default: 256**


### Simulation Settings:
`--level_files`: A list of level files as strings. **Default: [`'svg3_tareq2.svg'`]**

`--sim_time_step`: The time between steps. `Default: 0.1`


### Robot Settings:
`--number_of_rays`: The number of rays emitted by the laser. `Default: 1081`

`--field_of_view`: The lidar's field of view in degrees. `Default: 270`

`--collide_other_targets`: Determines whether the robot collides with targets of other robots (or passes through them). `Default: False`

`--manually`: Move the robot manually with wasd. `Default: False`


### Visualization and Managing Settings:

The simulation is visualized with the [rerun](https://rerun.io) viewer: it runs in its own process and renders on the GPU, so it costs the training loop almost nothing. In the viewer you can pause and scrub back through past episodes while training continues, toggle the lidar point clouds per robot (entity visibility), zoom/pan, and watch the per-robot reward plots next to the 2D world. During training, typing `s` + Enter in the terminal saves the current network weights (replaces the old save button).

`--visualization`: Choose the visualization mode: `none` (headless) or `rerun` (live viewer). **Default: `"rerun"`**

`--rerun_save`: Record the visualization to a `.rrd` file instead of spawning the live viewer; replay it later with `rerun <file>`. **Default: `None`**

`--rerun_memory_limit`: In-RAM history budget of the live viewer; the oldest events are dropped once it is exceeded. Percentage of system RAM (e.g. `25%`) or absolute (e.g. `2GB`). Two quirks of the viewer's garbage collector to be aware of: eviction works on data chunks, not episodes, so the oldest retained episodes may start mid-trajectory; and each cleanup pass frees **at least 25% of the used budget in one go** (by design, [rerun source](https://github.com/rerun-io/rerun/blob/0.33.0/crates/viewer/re_viewer/src/app.rs)), so the scrollback shrinks in big jumps rather than smoothly — with a large percentage budget those jumps are correspondingly large. An absolute limit like `4GB` makes the jumps smaller — but don't go far below `2GB`: the viewer's own UI and caches occupy tens of MB of the same budget, and below that floor the garbage collector compacts continuously no matter what. Log less data (`--viz_stride`, `--viz_lidar_stride`, `--viz_lidar_decimation`) for a longer window, or use `--rerun_save` for full history. **Default: `25%`**

`--viz_stride`: Log the world state only every Nth sim step. **Default: 4 in train mode, 1 in test mode**

`--viz_lidar_stride`: Log the lidar point clouds only every Nth logged step. **Default: 2**

`--viz_lidar_decimation`: Log only every Nth lidar ray (angular downsampling of the visualized point cloud). **Default: 3**

`--step_delay`: Sleep this many seconds per step in test mode to slow the simulation down for watching. **Default: 0**

`--tensorboard`: Use tensorboard. **Default: `True`**

`--print_interval`: How many episodes to print the results out. **Default: 1**

`--solved_percentage`: Stop training if objective is reached to this percentage. **Default: 0.99**

`--log_interval`: How many episodes to log into tensorboard. Also regulates how solved percentage is calculated. **Default: 30**