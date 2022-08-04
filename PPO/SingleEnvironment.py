from PPO.PPOAlgorithm import PPO
import numpy as np
import torch

class Memory:   # collected from old policy
    def __init__(self):
        self.states = []
        self.laser = []
        self.orientation = []
        self.distance = []
        self.velocity = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.laser[:]
        del self.orientation[:]
        del self.distance[:]
        del self.velocity[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]

def train(env_name, env, render, solved_reward,
    max_episodes, max_timesteps, update_timestep, action_std, K_epochs, eps_clip,
    gamma, lr, betas, ckpt_folder, restore, scan_size=121, tb=False, print_interval=10, save_interval=100):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    memory = Memory()

    ppo = PPO(scan_size, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt)

    running_reward, avg_length, time_step = 0, 0, 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        states = env.reset(0)
        for t in range(max_timesteps):
            time_step += 1

            # Run old policy
            actions = ppo.select_action(states, memory)

            states, rewards, dones, _ = env.step(actions)

            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)

            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += np.sum(rewards)
            if render:
                env.render()

            if env.is_done():
                break
        avg_length += t

        if running_reward > (print_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            print('Save a checkpoint!')
            break

        if i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            print('Save a checkpoint!')

        if i_episode % print_interval == 0:
            avg_length = int(avg_length / print_interval)
            running_reward = int((running_reward / print_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))

            if tb:
                writer.add_scalar('scalar/reward', running_reward, i_episode)
                writer.add_scalar('scalar/length', avg_length, i_episode)

            running_reward, avg_length = 0, 0