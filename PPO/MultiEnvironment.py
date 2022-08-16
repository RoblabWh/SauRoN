import os

from PPO.PPOAlgorithm import PPO
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import concurrent
import numpy as np
import torch
import ctypes
import sys
from Environment.Environment import Environment
from PyQt5.QtWidgets import QApplication



shared_array_laser_np = None
shared_array_distance_np = None
shared_array_orientation_np = None
shared_array_velocity_np = None
shared_array_action_np = None
shared_array_reward_np = None
shared_array_logprob_np = None
shared_array_terminal_np = None
def create_shared_memory_nparray(numOfProcesses, numOfRobots, learning_size, timesteps):
    global shared_array_laser_np
    global shared_array_distance_np
    global shared_array_orientation_np
    global shared_array_velocity_np
    global shared_array_action_np
    global shared_array_reward_np
    global shared_array_logprob_np
    global shared_array_terminal_np

    size_of_laser = 1081
    size_of_distance = 1
    size_of_orientation = 2
    size_of_velocity = 2
    size_of_action = 2
    size_of_reward = 1
    size_of_logprob = 1
    size_of_terminal = 1

    array_size_laser = numOfProcesses * numOfRobots * size_of_laser * learning_size * timesteps
    array_size_distance = numOfProcesses * numOfRobots * size_of_distance * learning_size * timesteps
    array_size_orientation = numOfProcesses * numOfRobots * size_of_orientation * learning_size * timesteps
    array_size_velocity = numOfProcesses * numOfRobots * size_of_velocity * learning_size * timesteps
    array_size_action = numOfProcesses * numOfRobots * size_of_action * learning_size * timesteps
    array_size_reward = numOfProcesses * numOfRobots * size_of_reward * learning_size * timesteps
    array_size_logprob = numOfProcesses * numOfRobots * size_of_logprob * learning_size * timesteps
    array_size_terminal = numOfProcesses * numOfRobots * size_of_terminal * learning_size * timesteps

    np_data_type = np.float32
    shape_laser = (numOfProcesses, learning_size, timesteps, size_of_laser)
    shape_distance = (numOfProcesses, learning_size, timesteps, size_of_distance)
    shape_orientation = (numOfProcesses, learning_size, timesteps, size_of_orientation)
    shape_velocity = (numOfProcesses, learning_size, timesteps, size_of_velocity)
    shape_action = (numOfProcesses, learning_size, timesteps, size_of_action)
    shape_reward = (numOfProcesses, learning_size, timesteps, size_of_reward)
    shape_logprob = (numOfProcesses, learning_size, timesteps, size_of_logprob)
    shape_terminal = (numOfProcesses, learning_size, timesteps, size_of_terminal)

    shared_array_laser = multiprocessing.RawArray(ctypes.c_float, array_size_laser)
    shared_array_distance = multiprocessing.RawArray(ctypes.c_float, array_size_distance)
    shared_array_orientation = multiprocessing.RawArray(ctypes.c_float, array_size_orientation)
    shared_array_velocity = multiprocessing.RawArray(ctypes.c_float, array_size_velocity)
    shared_array_action = multiprocessing.RawArray(ctypes.c_float, array_size_action)
    shared_array_reward = multiprocessing.RawArray(ctypes.c_float, array_size_reward)
    shared_array_logprob = multiprocessing.RawArray(ctypes.c_float, array_size_logprob)
    shared_array_terminal = multiprocessing.RawArray(ctypes.c_float, array_size_terminal)

    shared_array_laser_np = np.ndarray(shape_laser, dtype=np_data_type, buffer=shared_array_laser)
    shared_array_distance_np = np.ndarray(shape_distance, dtype=np_data_type, buffer=shared_array_distance)
    shared_array_orientation_np = np.ndarray(shape_orientation, dtype=np_data_type, buffer=shared_array_orientation)
    shared_array_velocity_np = np.ndarray(shape_velocity, dtype=np_data_type, buffer=shared_array_velocity)
    shared_array_action_np = np.ndarray(shape_action, dtype=np_data_type, buffer=shared_array_action)
    shared_array_reward_np = np.ndarray(shape_reward, dtype=np_data_type, buffer=shared_array_reward)
    shared_array_logprob_np = np.ndarray(shape_logprob, dtype=np_data_type, buffer=shared_array_logprob)
    shared_array_terminal_np = np.ndarray(shape_terminal, dtype=np_data_type, buffer=shared_array_terminal)

    print("#####Shared Memory#####")
    print("Num of Processes: {}".format(numOfProcesses))
    print("Num of Robots: {}".format(numOfRobots))
    print("Stack size: {}".format(learning_size))
    print("Time size : {}".format(timesteps))
    print("")
    print("Laser spape: {} with size: {}".format(shape_laser, size_of_laser))
    print("Distance spape: {} with size: {}".format(shape_distance, size_of_distance))
    print("Orientation spape: {} with size: {}".format(shape_orientation, size_of_orientation))
    print("Velocity spape: {} with size: {}".format(shape_velocity, size_of_velocity))
    print("Action spape: {} with size: {}".format(shape_action, size_of_action))
    print("Reward spape: {} with size: {}".format(shape_reward, size_of_reward))
    print("Logprob spape: {} with size: {}".format(shape_logprob, size_of_logprob))
    print("Terminal spape: {} with size: {}".format(shape_terminal, size_of_terminal))
    print("")
    print("")
    print("#############################")

def getNumOfProcesses(len):
    global args_
    cores = multiprocessing.cpu_count()
    print("Cores available: {}".format(cores))
    print("Num of levelfiles: {}".format(len))
    if len > cores:
        return cores
    else:
        return len

class SwarmMemory():
    def __init__(self, processID, robotsCount):
        self.processID = processID
        self.robotMemory = [Memory(self.processID, i) for i in range(robotsCount)]
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
    def __init__(self, processID, robotID):
        self.processID = processID
        self.robotID = robotID
        self.states = [shared_array_laser_np[processID][robotID], shared_array_distance_np[processID][robotID],
                       shared_array_orientation_np[processID][robotID], shared_array_velocity_np[processID][robotID]]
        self.actions = [shared_array_action_np[processID][robotID]]
        self.rewards = [shared_array_reward_np[processID][robotID]]
        self.is_terminals = [shared_array_terminal_np[processID][robotID]]
        self.logprobs = [shared_array_logprob_np[processID][robotID]]

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]

    def __len__(self):
        return len(self.states)


def train(env_name, render, solved_reward, input_style,
          max_episodes, max_timesteps, update_experience, action_std, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, restore, scan_size=121, print_interval=10, save_interval=100, batch_size=1,
          numOfRobots=4, args=None):
    args_ = args

    numOfProcesses = getNumOfProcesses(len(args_.level_files))
    print("Starting {} processes!\n".format(numOfProcesses))

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    create_shared_memory_nparray(numOfProcesses, numOfRobots, update_experience, 4)

    print("Start parallel training")
    print("####################")


    multiprocessing.set_start_method('spawn', force=True)
    futures = []
    pool = ProcessPoolExecutor(max_workers=numOfProcesses)
    for i in range(0, numOfProcesses):
        futures.append(pool.submit(runMultiprocessPPO, args=(i, max_episodes, env_name, max_timesteps, render,
                                                             print_interval, solved_reward, ckpt_folder, scan_size,
                                                             action_std, input_style, lr, betas, gamma, K_epochs,
                                                             eps_clip, restore, ckpt, args_)))

    done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    print("####################")
    print("Done!")

    #print("{}:{}".format(done, not_done))
    pool.shutdown()



def runMultiprocessPPO(args):
    processID, max_episodes, env_name, max_timesteps, render, print_interval, solved_reward, ckpt_folder, scan_size, \
    action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore, ckpt, args_obj = args

    app = None
    env = None
    ppo = None
    memory = None
    try:
        if processID == 0:
            app = QApplication(sys.argv)
        else:
            app = None

        env = Environment(app, args_obj, args_obj.time_frames, processID)


        ckpt = ckpt_folder + '/PPO_continuous_' + env_name + '.pth'

        ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt)

        memory = SwarmMemory(processID, env.getNumberOfRobots())

    except Exception as e:
        print(e)

    running_reward, avg_length, time_step = 0, 0, 0
    best_reward = 0
    print("Starting training loop of Process #{}".format(processID))
    # training loop
    update_experience = 1000
    try:
        for i_episode in range(1, max_episodes + 1):
            states = env.reset(0)
            for t in range(max_timesteps):
                time_step += 1

                # Run old policy
                actions = ppo.select_action(states, memory)

                states, rewards, dones, _ = env.step(actions)

                memory.insertReward(rewards)
                memory.insertIsTerminal(dones)

                #print("Test0")
                if len(memory) >= update_experience:
                #    print("0")
                    ppo.update(memory, batch_size)
                #    print("1")
                    memory.clear_memory()
                #    print("2")
                    time_step = 0

                #print("Test")
                running_reward += np.mean(rewards)
                #print("Test2")
                if render:
                    env.render()
                #print("Test3")
                if env.is_done():
                    break
                #print("Test4")
            avg_length += t

            if running_reward > (print_interval * solved_reward):
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
                print('Save a checkpoint!')
                break

            # if i_episode % save_interval == 0:
            #     torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            #     print('Save a checkpoint!')

            if i_episode % print_interval == 0:
                avg_length = int(avg_length / print_interval)
                running_reward = (running_reward / print_interval)

                if running_reward > best_reward:
                    best_reward = running_reward
                    torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
                    print('Save a checkpoint!')

                print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))

                running_reward, avg_length = 0, 0
    except Exception as e:
        print(e)

def test(env_name, env, render, action_std, input_style, K_epochs, eps_clip, gamma, lr, betas, ckpt_folder, test_episodes, scan_size=121):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    print('Load checkpoint from {}'.format(ckpt))

    memory = SwarmMemory(env.getNumberOfRobots())

    ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore=True, ckpt=ckpt)

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