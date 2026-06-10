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

The pretrained model under `./pretrained` is stored with Git LFS — run `git lfs install && git lfs pull` to fetch it.

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

There is a trained network provided under `./pretrained`. It was trained with an older version of the 'big' input space, which is preserved as `--inputspace legacy`. Note that the `small` and `big` input spaces were widened (the lidar embedding and trunk grew from 8/32 to 128/128 features), so checkpoints trained with the previous `small`/`big` networks are no longer loadable — only the `legacy` pretrained model is kept compatible. If the installation was successful the following command should run the pretrained network:

```uv run main.py --ckpt_folder ./pretrained --model_name batch_8 --mode test --inputspace legacy --level_files 'kreuzung.svg'```

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
`--visualization`: Choose the visualization mode: none, single or all. **Default: `"single"`**

`--visualization_paused`: Start the visualization toggled to paused. **Default: `False`**

`--tensorboard`: Use tensorboard. **Default: `True`**

`--print_interval`: How many episodes to print the results out. **Default: 1**

`--solved_percentage`: Stop training if objective is reached to this percentage. **Default: 0.99**

`--log_interval`: How many episodes to log into tensorboard. Also regulates how solved percentage is calculated. **Default: 30**

`--render`: Whether to render the environment. **Default: `False`**

`--scale_factor`: The scale factor for the environment. **Default: 55**

`--display_normals`: Determines whether the normals of a wall are shown in the map. **Default: `True`**