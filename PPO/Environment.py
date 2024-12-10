from PPO.Algorithm import PPO
from PPO.SwarmMemory import SwarmMemory
from PPO.CoolMemory import SwarmMemory as CoolSwarmMemory
from utils import Logger
import numpy as np
import torch
import time
from utils import statesToObservationsTensor,statesToObservationsNumpy, torchToNumpy


def train(env_name, env, solved_percentage, inputspace, max_episodes, max_timesteps,
          update_experience, _lambda, K_epochs, eps_clip, gamma, lr,
          betas, ckpt_folder, restore, tensorboard, scan_size=121, log_interval=10,
          batches=1, advantages_func=None):

    # Tensorboard
    logger = Logger(ckpt_folder, log_interval)
    logger.set_logging(tensorboard)
    best_reward = 0
    best_objective_reached = 0

    #memory = SwarmMemory(env.getNumberOfRobots())
    memory = CoolSwarmMemory(env.getNumberOfRobots())

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'

    ppo = PPO(scan_size=scan_size, inputspace=inputspace, lr=lr,
              betas=betas, gamma=gamma, _lambda=_lambda, K_epochs=K_epochs, eps_clip=eps_clip,
              logger=logger, restore=restore, ckpt=ckpt, advantages_func=advantages_func)

    env.setUISaveListener(ppo, ckpt_folder, env_name)

    training_counter = 0

    i_episode = 1
    level_idx = 0
    levels = len(env.getLevelManager())
    starttime = time.time()

    # training loop
    while i_episode < (max_episodes + 1):
        logger.episode = i_episode
        #states = env.reset()
        next_states = env.reset()
        level_idx += 1

        logger.set_number_of_agents(env.getNumberOfRobots())
        # memory.robotsCount = env.getNumberOfRobots()
        # memory.init()
        memory.unroll_last_episode(env.getNumberOfRobots())

        for t in range(max_timesteps):
            current_states = next_states
            # Run old policy
            actions, action_logprob = ppo.select_action(statesToObservationsTensor(current_states))

            next_states, rewards, dones, reached_goals = env.step(torchToNumpy(actions))

            # memory.insertObservations(o_laser, o_orientation, o_distance, o_velocity)
            unrolled_rewards = [sum([value for value in reward.values()]) for reward in rewards]
            memory.add(statesToObservationsNumpy(current_states), statesToObservationsNumpy(next_states), actions, action_logprob, unrolled_rewards, dones)

            logger.add_objective(reached_goals)
            logger.add_reward(rewards)
            logger.add_step_agents(len(rewards))

            if len(memory) >= update_experience:
                
                print('{}. training with {} experiences'.format(training_counter, len(memory)), flush=True)
                # memory.copyMemory()
                ppo.update(memory, batches)
                print('Time: {}'.format(time.time() - starttime), flush=True)
                starttime = time.time()
                training_counter += 1
                env.updateTrainingCounter(training_counter)

            if env.is_done():
                env.update_level_manager(goals_reached=reached_goals)
                break

        if i_episode % log_interval == 0:
            running_reward, objective_reached = logger.log()
            print(f'Percentage of objective reached: {objective_reached:.4f}', flush=True)
            if objective_reached >= solved_percentage:
                print(f"\nPercentage of: {objective_reached:.2f} reached!", flush=True)
                ppo.saveCurrentWeights(f"{env_name}_solved")
                print('Save as solved!!', flush=True)
                break

            if objective_reached > best_objective_reached:
                best_objective_reached = objective_reached
                ppo.saveCurrentWeights(f"{env_name}_best")
                print(
                    f'Best performance at training {training_counter}.',
                    flush=True)

        # if training_counter >= 100:
        #     break

        i_episode += 1
        # if len(memory) > 0:
        #     memory.copyMemory()
        # memory.clear_episode()

    ppo.saveCurrentWeights(f"{env_name}_final")

    if tensorboard:
        logger.close()


def test(env_name, env, render, inputspace, _lambda,
         K_epochs, eps_clip, gamma, lr, betas, ckpt_folder, test_episodes,
         scan_size=121, advantages_func=None):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    print('Load checkpoint from {}'.format(ckpt))

    ppo = PPO(scan_size, inputspace, lr, betas, gamma, _lambda,
              K_epochs, eps_clip, restore=True, ckpt=ckpt, logger=None,
              advantages_func=advantages_func)

    episode_reward, time_step = 0, 0
    avg_episode_reward, avg_length = 0, 0

    # test
    for i_episode in range(1, test_episodes+1):
        states = env.reset()
        while True:
            time_step += 1
            observations = statesToObservationsTensor(states)

            # Run old policy
            actions = ppo.select_action_certain(observations)

            states, rewards, dones, _ = env.step(torchToNumpy(actions))

            episode_reward += sum([sum([value for value in reward.values()]) for reward in rewards])

            if render:
                env.render()

            if env.is_done():
                print('Episode {} \t Length: {} \t Reward: {}'.format(i_episode, time_step, episode_reward))
                avg_episode_reward += episode_reward
                avg_length += time_step
                time_step, episode_reward = 0, 0
                break

    print('Test {} episodes DONE!'.format(test_episodes))
    print('Avg episode reward: {} | Avg length: {}'.format(avg_episode_reward/test_episodes, avg_length/test_episodes))