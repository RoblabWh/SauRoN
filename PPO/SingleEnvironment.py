from PPO.PPOAlgorithm import PPO
from torch.utils.data import Dataset, DataLoader
from utils import Logger
import numpy as np
import torch


class SwarmMemory():
    def __init__(self, robotsCount):
        self.robotMemory = [Memory() for _ in range(robotsCount)]
        self.currentTerminalStates = [False for _ in range(robotsCount)]

    def __getitem__(self, item):
        return self.robotMemory[item]

    # Gets relative Index according to currentTerminalStates
    def getRelativeIndices(self):
        relativeIndices = []
        for i in range(len(self.currentTerminalStates)):
            if not self.currentTerminalStates[i]:
                relativeIndices.append(i)

        return relativeIndices

    def insertState(self, laser, orientation, distance, velocity):
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].states.append([laser[i], orientation[i], distance[i], velocity[i]])

    def insertAction(self, action):
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].actions.append(action[i])

    def insertReward(self, reward):
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].rewards.append(reward[i])

    def insertLogProb(self, logprob):
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].logprobs.append(logprob[i])

    def insertReachedGoal(self, reachedGoal, isTerminal):
        terminalGoal = np.logical_and(reachedGoal, isTerminal)
        relativeIndices = self.getRelativeIndices()
        for idx in np.where(isTerminal)[0]:
            self.robotMemory[relativeIndices[idx]].reached_goal.append(terminalGoal[idx])

    def insertIsTerminal(self, isTerminal):
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].is_terminals.append(isTerminal[i])
            if isTerminal[i]:
                self.currentTerminalStates[relativeIndices[i]] = True

        # check if currentTerminalStates is all True
        if all(self.currentTerminalStates):
            self.currentTerminalStates = [False for _ in range(len(self.currentTerminalStates))]

    def getStatesOfAllRobots(self):
        laser = []
        orientation = []
        distance = []
        velocity = []
        for robotmemory in self.robotMemory:
            for state in robotmemory.states:
                laser.append(state[0])
                orientation.append(state[1])
                distance.append(state[2])
                velocity.append(state[3])

        return [torch.stack(laser), torch.stack(orientation), torch.stack(distance), torch.stack(velocity)]

    def getActionsOfAllRobots(self):
        actions = []
        for robotmemory in self.robotMemory:
            for action in robotmemory.actions:
                actions.append(action)

        return actions

    def getLogProbsOfAllRobots(self):
        logprobs = []
        for robotmemory in self.robotMemory:
            for logprob in robotmemory.logprobs:
                logprobs.append(logprob)

        return logprobs

    def clear_memory(self):
        for memory in self.robotMemory:
            memory.clear_memory()

    def __len__(self):
        length = 0
        for memory in self.robotMemory:
            length += len(memory)
        return length


class Memory:   # collected from old policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.reached_goal = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.reached_goal[:]
        del self.logprobs[:]

    def __len__(self):
        return len(self.states)

def train(env_name, env, solved_percentage, input_style,
          max_episodes, max_timesteps, update_experience, action_std, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, restore, tensorboard, scan_size=121, log_interval=10, batch_size=1):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    # Tensorboard
    logger = Logger(log_dir=ckpt_folder, update_interval=log_interval)
    if tensorboard:
        logger.set_logging(True)

    memory = SwarmMemory(env.getNumberOfRobots())

    ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, logger=logger, restore=restore, ckpt=ckpt)
    env.setUISaveListener(ppo, ckpt_folder, env_name)

    #logger.build_graph(ppo.policy.actor, ppo.policy.device)
    #logger.build_graph(ppo.policy.critic, ppo.policy.device)

    running_reward, avg_length, time_step = 0, 0, 0
    best_reward = 0
    # training loop
    for i_episode in range(1, max_episodes+1):
        states = env.reset(0)
        logger.set_episode(i_episode)
        for t in range(max_timesteps):
            time_step += 1

            # Run old policy
            actions = ppo.select_action(states, memory)

            states, rewards, dones, reachedGoals = env.step(actions)

            memory.insertReward(rewards)
            #memory.insertReachedGoal(reachedGoals, dones) not used just now
            memory.insertIsTerminal(dones)

            logger.log_objective(reachedGoals)

            if len(memory) >= update_experience:
                print('Train Network at Episode {}'.format(i_episode))
                ppo.update(memory, batch_size)
                memory.clear_memory()
                time_step = 0

            running_reward += np.mean(rewards)

            if env.is_done():
                break
        avg_length += t

        if logger.percentage_objective_reached() > solved_percentage:
            print(f"Percentage of: {logger.percentage_objective_reached():.2f} reached!")
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            print('Save as solved!')
            break

        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = (running_reward / log_interval)

            if running_reward > best_reward:
                best_reward = running_reward
                torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
                print(f'Best performance with avg reward of {best_reward:.2f} saved at episode {i_episode}.')
                print(f'Percentage of objective reached: {logger.percentage_objective_reached():.4f}')

            logger.scalar_summary('reward', running_reward)
            logger.scalar_summary('Avg Steps', avg_length)
            logger.summary_objective()
            logger.summary_actor_output()
            logger.summary_loss()

            if not tensorboard:
                print(f'Episode: {i_episode}, Avg reward: {running_reward:.2f}, Avg steps: {avg_length:.2f}')

            running_reward, avg_length = 0, 0

    if tensorboard:
        logger.close()


def test(env_name, env, render, action_std, input_style, K_epochs, eps_clip, gamma, lr, betas, ckpt_folder, test_episodes, scan_size=121):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    print('Load checkpoint from {}'.format(ckpt))

    memory = SwarmMemory(env.getNumberOfRobots())

    ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore=True, ckpt=ckpt, logger=None)

    episode_reward, time_step = 0, 0
    avg_episode_reward, avg_length = 0, 0

    # test
    for i_episode in range(1, test_episodes+1):
        states = env.reset(0)
        while True:
            time_step += 1

            # Run old policy
            actions = ppo.select_action_certain(states, memory)

            states, rewards, dones, _ = env.step(actions)
            memory.insertIsTerminal(dones)

            episode_reward += np.sum(rewards)

            if render:
                env.render()

            if env.is_done():
                print('Episode {} \t Length: {} \t Reward: {}'.format(i_episode, time_step, episode_reward))
                avg_episode_reward += episode_reward
                avg_length += time_step
                memory.clear_memory()
                time_step, episode_reward = 0, 0
                break

    print('Test {} episodes DONE!'.format(test_episodes))
    print('Avg episode reward: {} | Avg length: {}'.format(avg_episode_reward/test_episodes, avg_length/test_episodes))