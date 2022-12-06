from PPO.Algorithm import PPO
from utils import Logger, mpi_reduce, mpi_comm_split
from mpi4py import MPI
import numpy as np
import torch
import time

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

class SwarmMemory():
    def __init__(self, robotsCount = 0):
        self.robotsCount = robotsCount
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
        self.robotMemory = self.robotMemory[:self.robotsCount]
        self.currentTerminalStates = self.currentTerminalStates[:self.robotsCount]
        for memory in self.robotMemory:
            memory.clear_memory()

    def __add__(self, other):
        new_memory = SwarmMemory()
        new_memory.robotMemory += self.robotMemory
        new_memory.currentTerminalStates += self.currentTerminalStates
        if other is not None:
            new_memory.robotMemory += other.robotMemory
            new_memory.currentTerminalStates += other.currentTerminalStates
        return new_memory

    def __iadd__(self, other):
        if other is not None:
            self.robotMemory += other.robotMemory
            self.currentTerminalStates += other.currentTerminalStates
        return self

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

    # Tensorboard
    logger = Logger(ckpt_folder)
    if MPI_RANK == 0:
        logger.set_logging(tensorboard)

    memory = SwarmMemory(env.getNumberOfRobots())

    ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, logger=logger)
    env.setUISaveListener(ppo, ckpt_folder, env_name)

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    if restore:
        restored = None
        if MPI_RANK == 0:
            print('Load checkpoint from {}'.format(ckpt), flush=True)
            restored = torch.load(ckpt, map_location=lambda storage, loc: storage)
        pretrained_model = MPI_COMM.bcast(restored)
        #print(f'Rank {mpi_rank} len model {len(pretrained_model)}', flush=True)
        ppo.policy.load_state_dict(pretrained_model)

    #logger.build_graph(ppo.policy.actor, ppo.policy.device)
    #logger.build_graph(ppo.policy.critic, ppo.policy.device)

    mpi_comm_alive = MPI_COMM.Split()
    training_counter = 0

    best_reward = 0
    objective_reached = 0

    max_episodes += 1
    i_episode = 1
    starttime = time.time()
    # training loop
    while i_episode < max_episodes:
        states = env.reset()
        # logger.set_episode(i_episode)
        logger.set_number_of_agents(env.getNumberOfRobots())

        for t in range(max_timesteps):

            # Run old policy
            actions = ppo.select_action(states, memory)
            #print(f'actions {actions}', flush=True)

            states, rewards, dones, reachedGoals = env.step(actions)

            logger.add_reward(np.mean(rewards))

            memory.insertReward(rewards)
            #memory.insertReachedGoal(reachedGoals, dones) not used just now
            memory.insertIsTerminal(dones)

            logger.add_objective(reachedGoals)

            if len(memory) >= update_experience:
                mpi_comm_alive = mpi_comm_split(mpi_comm_alive, True)
                mpi_rank = mpi_comm_alive.Get_rank()
                
                mpi_reduce(memory, mpi_comm_alive)
                if mpi_rank == 0:
                    print('Train Network {} with {} Experiences'.format(training_counter, len(memory)), flush=True)
                    print('Time: {}'.format(time.time() - starttime), flush=True)
                    starttime = time.time()
                    ppo.update(memory, batch_size)
                memory.clear_memory()
                training_counter += 1
                env.updateTrainingCounter(training_counter)
                pth = mpi_comm_alive.bcast(ppo.policy.state_dict())
                if mpi_rank != 0:
                    ppo.policy.load_state_dict(pth)

                if training_counter % log_interval == 0:
                    means = logger.get_means()
                    logger_count = mpi_comm_alive.allreduce(means[0])
                    if logger_count > 0:
                        running_reward = mpi_comm_alive.reduce(means[1])
                        objective_reached = mpi_comm_alive.allreduce(means[2]) / logger_count
                        steps = mpi_comm_alive.reduce(means[3])
                        actor_mean_linvel = mpi_comm_alive.reduce(means[4])
                        actor_mean_angvel = mpi_comm_alive.reduce(means[5])
                        actor_var_linvel = mpi_comm_alive.reduce(means[6])
                        actor_var_angvel = mpi_comm_alive.reduce(means[7])
                        if mpi_rank == 0:
                            steps /= logger_count
                            actor_mean_linvel /= logger_count
                            actor_mean_angvel /= logger_count
                            actor_var_linvel /= logger_count
                            actor_var_angvel /= logger_count
                            running_reward /= logger_count

                            if running_reward > best_reward:
                                best_reward = running_reward
                                torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}_best.pth'.format(env_name))
                                print(f'Best performance with avg reward of {best_reward:.2f} saved at training {training_counter}.', flush=True)
                                print(f'Percentage of objective reached: {objective_reached:.4f}', flush=True)

                            logger.summary_reward(running_reward, training_counter)
                            logger.summary_objective(objective_reached, training_counter)
                            logger.summary_steps(steps, training_counter)
                            logger.summary_actor_output(actor_mean_linvel, actor_mean_angvel, actor_var_linvel, actor_var_angvel, training_counter)
                            logger.summary_loss(training_counter)

                            if not tensorboard:
                                print(f'Training: {training_counter}, Avg reward: {running_reward:.2f}, Steps: {steps:.2f}', flush=True)
                    
                    if means[0]:
                        logger.clear_summary()

            if env.is_done():
                break

        logger.add_steps(t)
        logger.log_episode(i_episode)

        if objective_reached >= solved_percentage:
            if mpi_rank == 0:
                print(f"Percentage of: {objective_reached:.2f} reached!", flush=True)
                torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}_solved.pth'.format(env_name))
                print('Save as solved!', flush=True)
            break

        i_episode += 1

    torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}_{}_ended.pth'.format(env_name, MPI_RANK))
    mpi_comm_alive = mpi_comm_split(mpi_comm_alive, False)
    mpi_comm_alive.Disconnect()

    if MPI_RANK == 0 and tensorboard:
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
        states = env.reset()
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