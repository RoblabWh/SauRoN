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

Install Anaconda on your system:

[Installation Guide](https://docs.anaconda.com/anaconda/install/linux/)

Run the `install_conda_env.sh` script to create your anaconda environment with it. Now you should be ready to run your simulation.

## Levels

There are some levels provided as svg Files. Red dots mark possible agent positions while green dots mark possible goals for the agents. 

![Alt text](svg/tunnel.svg?raw=true "tunnel level")

## Usage

**Currently only PPO is implemented** 

The training can be started either using MPI for Multi Environment or without MPI to only train on a single level.

From your conda-environment run `main.py` with desired arguments:

- `--mode` choose either `train` or `test` for Training or running a trained model
- `--ckpt_folder` Checkpoint Folder for the loaded and new models. Default : `./models`

There are several other options for the hyperparameters, vizualization and robot settings you can look up in the `main.py`.

There is a trained network provided under `./models`. If the installation was successful the following command should run a pretrained network:
 
```python main.py --ckpt_folder ./pretrained --mode test```

To train run:

```python main.py --mode train```

To train with MPI:

```mpirun -n N_PROC python main.py --mode train```