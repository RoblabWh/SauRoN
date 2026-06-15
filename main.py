from PPO.Environment import train, test
from Environment.Environment import Environment
from utils import str2bool, check_args
import os
import argparse
from types import SimpleNamespace


def build_parser():
    """Single source of truth for all settings and their defaults.

    Both entrypoints use this: the CLI (`python main.py ...`) via parse_args(),
    and the notebook (`startSimulation`) via default_args()."""
    parser = argparse.ArgumentParser(description='SauRoN Simulation')
    parser.add_argument('--ckpt_folder', default='./models/small', help='Location to save checkpoint models')
    parser.add_argument('--model_name', default='model', help='Name of the modelfile')
    parser.add_argument('--mode', default='train', help='choose train or test')

    # Train Parameters

    parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')
    parser.add_argument('--time_frames', type=int, default=4, help='Number of Timeframes (past States) which will be analyzed by neural net') # TODO not properly implemented
    parser.add_argument('--steps', type=int, default=5000, help='Steps in Environment per Episode')
    parser.add_argument('--max_episodes', type=float, default="inf", help='Maximum Number of Episodes')
    parser.add_argument('--update_experience', type=int, default=16384, help='how many experiences to update the policy')
    parser.add_argument('--batches', type=int, default=32, help='number of minibatches per epoch (minibatch size = update_experience / batches)')
    parser.add_argument('--action_std', type=float, default=0.5, help='constant std for action distribution (Multivariate Normal)') # TODO currently not used
    parser.add_argument('--_lambda', type=float, default=0.97, help='lambda for advantage calculation (paired with the long gamma horizon)')
    parser.add_argument('--K_epochs', type=int, default=6, help='update the policy K times')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
    parser.add_argument('--entropy_coeff', type=float, default=0.001, help='coefficient of the entropy bonus in the actor loss')
    parser.add_argument('--kl_early_stop', type=str2bool, default=False, help='stop the policy update early once approx KL exceeds --kl_target')
    parser.add_argument('--kl_target', type=float, default=0.02, help='approximate KL divergence threshold for early stopping')
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'cosine'], help='learning rate schedule (cosine annealing, stepped per minibatch update)')
    parser.add_argument('--lr_tmax', type=int, default=1_000_000, help='T_max (in minibatch updates) of the cosine annealing schedule')
    parser.add_argument('--torch_compile', type=str2bool, default=False, help='JIT-compile actor/critic with torch.compile (slow first update, faster afterwards)')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor (effective horizon ~1000 steps / ~40 m at the robots step length of ~0.04 m -- must cover detour lengths in the larger levels)')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--inputspace', default='small', help='big or small') # image not advised to use but functional
    parser.add_argument('--image_size', type=float, default=256, help='size of the image that goes into the neural net')

    # Simulation settings

    parser.add_argument('--level_files', type=str, nargs='+', default=['maze1.svg', 'maze2.svg', 'maze3.svg', 'maze4.svg', 'new1.svg'], help='List of level files as strings')
    parser.add_argument('--sim_time_step', type=float, default=0.225, help='Time between steps') #.125
    parser.add_argument('--geodesic_reward', type=str2bool, default=True,
                        help='Shape the progress reward on geodesic (shortest obstacle-free path) distance to the goal instead of straight-line distance. Removes the local minima at walls between robot and goal. Observations are unaffected')
    parser.add_argument('--flow_field_cell', type=float, default=0.1,
                        help='Cell size in meters of the rasterized level grid for the geodesic flow field; must stay well below the narrowest passage width minus the robot diameter')

    # Level generator defaults (apply to gen: entries in --level_files, e.g.
    # "gen:maze" or "gen:rooms?passage=1.0&density=80&seed=42"; per-token ?key=value
    # overrides beat these globals. A fresh level is generated every time a gen:
    # entry is selected, i.e. every episode.)

    parser.add_argument('--gen_robots_min', type=int, default=4, help='Generated levels: lower bound of the random robot count (1-4)')
    parser.add_argument('--gen_robots_max', type=int, default=4, help='Generated levels: upper bound of the random robot count (1-4)')
    parser.add_argument('--gen_pool', type=str2bool, default=True, help='Generated levels: place an orange spawn pool (2*robots+1 circles, randomized start/goal pairs) instead of fixed numbered pairs')
    parser.add_argument('--gen_passage', type=float, default=1.2, help='Generated levels: minimum corridor/door width in meters (robot is 0.35 m wide; the flow field blocks passages much under 0.8 m)')
    parser.add_argument('--gen_density', type=float, default=50, help='Generated levels: clutter 0-100. open/pillars: obstacle count; maze: wall density (lower = more shortcuts); rooms: room count')
    parser.add_argument('--gen_arena', type=float, nargs=2, default=[620, 280], metavar=('W', 'H'), help='Generated levels: arena size in px (28.35 px = 1 m)')
    parser.add_argument('--gen_seed', type=int, default=None, help='Generated levels: base seed for a reproducible level sequence (seed+counter per generation); default None = fresh random level each time')

    # Robot settings

    parser.add_argument('--number_of_rays', type=int, default=1081, help='The number of Rays emittet by the laser')
    parser.add_argument('--field_of_view', type=int, default=270, help='The lidars field of view in degree')
    parser.add_argument('--lidar_d0', type=float, default=2.0,
                        help='Scale of the inverse-depth lidar encoding x = d0 / (d0 + d); an obstacle at d0 meters encodes as 0.5')
    parser.add_argument('--goal_d0', type=float, default=10.0,
                        help='Scale of the goal-distance encoding x = d / (d + d0); 0 at the goal, 0.5 at d0 meters, -> 1 far away')
    parser.add_argument('--has_pie_slice', type=str2bool, default='False',
                        help='Determines if an Object is places on top of the robot to reflect other robots lidar') # not advised to use but functional
    parser.add_argument('--collide_other_targets', type=str2bool, default=False,
                        help='Determines whether the robot collides with targets of other robots (or passes through them)') # not advised to use but functional
    parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=False,
                        help='Moving robot manually with wasd')

    # Visualization & Managing settings

    parser.add_argument('--visualization', type=str, default="rerun", help="Visualization mode. none: headless; rerun: stream the simulation to the rerun viewer")
    parser.add_argument('--rerun_save', type=str, default=None, help='Record the visualization to a .rrd file instead of spawning the live viewer (replay with: rerun <file>)')
    parser.add_argument('--rerun_memory_limit', type=str, default='25%', help="In-RAM history budget of the live viewer; oldest events are dropped beyond it. Percentage of system RAM (e.g. '25%%') or absolute (e.g. '2GB')")
    parser.add_argument('--viz_stride', type=int, default=None, help='Log the world state only every Nth sim step (default: 4 in train mode, 1 in test mode)')
    parser.add_argument('--viz_lidar_stride', type=int, default=2, help='Log the lidar point clouds only every Nth logged step')
    parser.add_argument('--viz_lidar_decimation', type=int, default=3, help='Log only every Nth lidar ray (angular downsampling of the visualized point cloud)')
    parser.add_argument('--viz_flow_field', type=str2bool, default=False, help='Debug overlay: render the geodesic flow-field raster (free cells colored by distance, walls and inflation band) and each robots shortest path in the rerun viewer')
    parser.add_argument('--step_delay', type=float, default=0.0, help='Sleep this many seconds per step in test mode to slow the simulation down for watching')
    parser.add_argument('--tensorboard', type=str2bool, default=True, help='Use tensorboard')
    parser.add_argument('--print_interval', type=int, default=1, help='how many episodes to print the results out')
    parser.add_argument('--solved_percentage', type=float, default=0.99, help='stop training if objective is reached to this percentage')
    parser.add_argument('--log_interval', type=int, default=30, help='how many episodes to log into tensorboard. Also regulates how solved percentage is calculated')
    return parser


def default_args():
    """All CLI defaults as a namespace -- the notebook entrypoint starts from these
    so it always trains with the exact same settings as `python main.py`."""
    return build_parser().parse_args([])


def run(args, env):
    """The single train/test dispatch shared by the CLI and the notebook."""
    if args.mode == 'train':
        train(args.model_name, env, inputspace=args.inputspace, solved_percentage=args.solved_percentage,
              max_episodes=args.max_episodes, max_timesteps=args.steps, update_experience=args.update_experience,
              _lambda=args._lambda, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
              gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder,
              restore=args.restore, log_interval=args.log_interval, scan_size=args.number_of_rays,
              batches=args.batches, tensorboard=args.tensorboard,
              entropy_coeff=args.entropy_coeff, kl_early_stop=args.kl_early_stop, kl_target=args.kl_target,
              lr_schedule=args.lr_schedule, lr_tmax=args.lr_tmax, torch_compile=args.torch_compile)
    elif args.mode == 'test':
        test(args.model_name, env, inputspace=args.inputspace,
             _lambda=args._lambda, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
             gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder, test_episodes=100,
             scan_size=args.number_of_rays, step_delay=args.step_delay, torch_compile=args.torch_compile)


def startSimulation(args, level_files, createReward, selectLevel):
    """Programmatic entrypoint for the notebooks.

    Starts from the CLI defaults (default_args()), overlays the settings the
    notebook provides in `args`, injects the level list and the custom reward /
    curriculum callbacks, then runs the exact same train/test path as the CLI.

    :param args: dict of overrides (only the keys you set differ from main's defaults)
    :param level_files: list of level files used as training environments
    :param createReward: custom reward function (see Environment.createReward)
    :param selectLevel: custom curriculum function (see LevelManager)
    """
    settings = vars(default_args())
    settings.update(args)            # notebook overrides win over the CLI defaults
    settings['level_files'] = level_files
    ns = SimpleNamespace(**settings)

    if not os.path.exists(ns.ckpt_folder):
        os.makedirs(ns.ckpt_folder)
    check_args(ns)

    env = Environment(ns, reward_func=createReward, curriculum_func=selectLevel)
    run(ns, env)


if __name__ == '__main__':
    args = build_parser().parse_args()
    if not os.path.exists(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)
    check_args(args)
    print(args)

    env = Environment(args)
    run(args, env)
