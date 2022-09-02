import os

from PPO.PPOAlgorithm import PPO
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures.process import ProcessPoolExecutor
import concurrent
import numpy as np
import torch
import ctypes
import sys
from Environment.Environment import Environment
from PyQt5.QtWidgets import QApplication
import signal, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

shared_array_laser = None
shared_array_distance = None
shared_array_orientation = None
shared_array_velocity = None
shared_array_action = None
shared_array_reward = None
shared_array_logprob = None
shared_array_terminal = None
shared_array_signal = None
shared_array_counter = None

shared_array_laser_np = None
shared_array_distance_np = None
shared_array_orientation_np = None
shared_array_velocity_np = None
shared_array_action_np = None
shared_array_reward_np = None
shared_array_logprob_np = None
shared_array_terminal_np = None
shared_array_signal_np = None
shared_array_counter_np = None


def create_shared_memory_nparray(numOfProcesses, numOfRobots, learning_size, timesteps):
    global shared_array_laser_np
    global shared_array_distance_np
    global shared_array_orientation_np
    global shared_array_velocity_np
    global shared_array_action_np
    global shared_array_reward_np
    global shared_array_logprob_np
    global shared_array_terminal_np
    global shared_array_signal_np
    global shared_array_counter_np

    global shared_array_laser
    global shared_array_distance
    global shared_array_orientation
    global shared_array_velocity
    global shared_array_action
    global shared_array_reward
    global shared_array_logprob
    global shared_array_terminal
    global shared_array_signal
    global shared_array_counter


    size_of_laser = 1081
    size_of_distance = 1
    size_of_orientation = 2
    size_of_velocity = 2
    size_of_action = 2
    size_of_reward = 1
    size_of_logprob = 1
    size_of_terminal = 1

    array_size_laser = numOfProcesses * numOfRobots * size_of_laser * learning_size * timesteps * 4 #Sizeof(float)
    array_size_distance = numOfProcesses * numOfRobots * size_of_distance * learning_size * timesteps * 4 #Sizeof(float)
    array_size_orientation = numOfProcesses * numOfRobots * size_of_orientation * learning_size * timesteps * 4 #Sizeof(float)
    array_size_velocity = numOfProcesses * numOfRobots * size_of_velocity * learning_size * timesteps * 4 #Sizeof(float)
    array_size_action = numOfProcesses * numOfRobots * size_of_action * learning_size * 4 #Sizeof(float)
    array_size_reward = numOfProcesses * numOfRobots * size_of_reward * learning_size * 4 #Sizeof(float)
    array_size_logprob = numOfProcesses * numOfRobots * size_of_logprob * learning_size * 4 #Sizeof(float)
    array_size_terminal = numOfProcesses * numOfRobots * size_of_terminal * learning_size * 4#Sizeof(float)
    array_size_signal = numOfProcesses * 4 #SizeOf(int)
    num_of_numpy = 5 #Number of different numpy arrays
    array_size_counter = numOfProcesses * numOfRobots * num_of_numpy * 4 #4 because of np.int32

    shape_laser = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_laser)
    shape_distance = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_distance)
    shape_orientation = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_orientation)
    shape_velocity = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_velocity)
    shape_action = (numOfProcesses, numOfRobots, learning_size, size_of_action)
    shape_reward = (numOfProcesses, numOfRobots, learning_size, size_of_reward)
    shape_logprob = (numOfProcesses, numOfRobots, learning_size, size_of_logprob)
    shape_terminal = (numOfProcesses, numOfRobots, learning_size, size_of_terminal)
    shape_signal = (numOfProcesses,)
    shape_counter = (numOfProcesses, numOfRobots, num_of_numpy)

    try:
        shared_array_laser = shared_memory.SharedMemory(create=True, size=array_size_laser, name="shared_array_laser")
    except Exception as e:
        shared_array_laser = shared_memory.SharedMemory(name="shared_array_laser")
    try:
        shared_array_distance = shared_memory.SharedMemory(create=True, size=array_size_distance, name="shared_array_distance")
    except Exception as e:
        shared_array_distance = shared_memory.SharedMemory(name="shared_array_distance")
    try:
        shared_array_orientation = shared_memory.SharedMemory(create=True, size=array_size_orientation, name="shared_array_orientation")
    except Exception as e:
        shared_array_orientation = shared_memory.SharedMemory(name="shared_array_orientation")
    try:
        shared_array_velocity = shared_memory.SharedMemory(create=True, size=array_size_velocity, name="shared_array_velocity")
    except Exception as e:
        shared_array_velocity = shared_memory.SharedMemory(name="shared_array_velocity")
    try:
        shared_array_action = shared_memory.SharedMemory(create=True, size=array_size_action, name="shared_array_action")
    except Exception as e:
        shared_array_action = shared_memory.SharedMemory(name="shared_array_action")
    try:
        shared_array_reward = shared_memory.SharedMemory(create=True, size=array_size_reward,  name="shared_array_reward")
    except Exception as e:
        shared_array_reward = shared_memory.SharedMemory(name="shared_array_reward")
    try:
        shared_array_logprob = shared_memory.SharedMemory(create=True, size=array_size_logprob, name="shared_array_logprob")
    except Exception as e:
        shared_array_logprob = shared_memory.SharedMemory(name="shared_array_logprob")
    try:
        shared_array_terminal = shared_memory.SharedMemory(create=True, size=array_size_terminal, name="shared_array_terminal")
    except Exception as e:
        shared_array_terminal = shared_memory.SharedMemory(name="shared_array_terminal")
    try:
        shared_array_signal = shared_memory.SharedMemory(create=True, size=array_size_signal, name="shared_array_signal")
    except Exception as e:
        shared_array_signal = shared_memory.SharedMemory(name="shared_array_signal")
    try:
        shared_array_counter = shared_memory.SharedMemory(create=True, size=array_size_counter, name="shared_array_counter")
    except Exception as e:
        shared_array_counter = shared_memory.SharedMemory(name="shared_array_counter")



    np_data_type = np.float32
    shared_array_laser_np = np.ndarray(shape_laser, dtype=np_data_type, buffer=shared_array_laser.buf)
    shared_array_distance_np = np.ndarray(shape_distance, dtype=np_data_type, buffer=shared_array_distance.buf)
    shared_array_orientation_np = np.ndarray(shape_orientation, dtype=np_data_type, buffer=shared_array_orientation.buf)
    shared_array_velocity_np = np.ndarray(shape_velocity, dtype=np_data_type, buffer=shared_array_velocity.buf)
    shared_array_action_np = np.ndarray(shape_action, dtype=np_data_type, buffer=shared_array_action.buf)
    shared_array_reward_np = np.ndarray(shape_reward, dtype=np_data_type, buffer=shared_array_reward.buf)
    shared_array_logprob_np = np.ndarray(shape_logprob, dtype=np_data_type, buffer=shared_array_logprob.buf)
    shared_array_terminal_np = np.ndarray(shape_terminal, dtype=np.bool, buffer=shared_array_terminal.buf)
    shared_array_signal_np = np.ndarray(shape_signal, dtype=np.int32, buffer=shared_array_signal.buf)
    shared_array_counter_np = np.ndarray(shape_counter, dtype=np.int32, buffer=shared_array_counter.buf)

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
    print("Signal spape: {}".format(shape_signal))
    print("Counter spape: {}".format(shape_counter))
    print("")
    print("")
    print("#############################")

def handler(signum, frame):
    print('Signal handler called with signal', signum)
    close_shm()


def setupSignalHandler():
    signal.signal(signal.SIGALRM, handler)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGSEGV, handler)
    signal.signal(signal.SIGINT, handler)


def close_shm():
    global shared_array_laser
    global shared_array_distance
    global shared_array_orientation
    global shared_array_velocity
    global shared_array_action
    global shared_array_reward
    global shared_array_logprob
    global shared_array_terminal
    global shared_array_counter
    global shared_array_signal

    shared_array_laser.close()
    shared_array_distance.close()
    shared_array_orientation.close()
    shared_array_velocity.close()
    shared_array_action.close()
    shared_array_reward.close()
    shared_array_logprob.close()
    shared_array_terminal.close()
    shared_array_counter.close()
    shared_array_signal.close()

    shared_array_laser.unlink()
    shared_array_distance.unlink()
    shared_array_orientation.unlink()
    shared_array_velocity.unlink()
    shared_array_action.unlink()
    shared_array_reward.unlink()
    shared_array_logprob.unlink()
    shared_array_terminal.unlink()
    shared_array_counter.unlink()
    shared_array_signal.unlink()


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
    def __init__(self, processID=-1, robotsCount=0):
        self.processID = processID
        self.train = 1
        if processID == -1:
            self.train = 0
            self.loadFromShm()
        self.robotMemory = [Memory(processID=self.processID, robotID=num) for num in range(robotsCount)]
        self.currentTerminalStates = [False for _ in range(robotsCount)]

    def __getitem__(self, item):
        return self.robotMemory[item]

    def copyToShm(self):
        for robot in self.robotMemory:
            robot.copyToShm()

    def loadFromShm(self):
        for robot in self.robotMemory:
            robot.loadFromShm()

    # Gets relative Index according to currentTerminalStates
    def getRelativeIndices(self):
        relativeIndices = []
        for i in range(len(self.currentTerminalStates)):
            if not self.currentTerminalStates[i]:
                relativeIndices.append(i)

        return relativeIndices

    def insertState(self, laser, orientation, distance, velocity):
        global shared_array_counter_np
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].states.append([laser[i], orientation[i], distance[i], velocity[i]])
            shared_array_counter_np[self.processID][relativeIndices[i]][0] += 1

    def insertAction(self, action):
        global shared_array_counter_np
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].actions.append(action[i])
            shared_array_counter_np[self.processID][relativeIndices[i]][1] += 1

    def insertReward(self, reward):
        global shared_array_counter_np
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].rewards.append(reward[i])
            shared_array_counter_np[self.processID][relativeIndices[i]][2] += 1

    def insertLogProb(self, logprob):
        global shared_array_counter_np
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].logprobs.append(logprob[i])
            shared_array_counter_np[self.processID][relativeIndices[i]][3] += 1

    def insertIsTerminal(self, isTerminal):
        global shared_array_counter_np
        relativeIndices = self.getRelativeIndices()
        for i in range(len(relativeIndices)):
            self.robotMemory[relativeIndices[i]].is_terminals.append(isTerminal[i])
            shared_array_counter_np[self.processID][relativeIndices[i]][4] += 1
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
        self.robotID = robotID
        self.processID = processID
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def loadFromShm(self):
        amount_of_state = shared_array_counter[self.processID][self.robotID][0]
        amount_of_action = shared_array_counter[self.processID][self.robotID][1]
        amount_of_rewards = shared_array_counter[self.processID][self.robotID][2]
        amount_of_terminals = shared_array_counter[self.processID][self.robotID][3]
        amount_of_logprobs = shared_array_counter[self.processID][self.robotID][4]

        self.states[0] = torch.from_numpy([self.processID][self.robotID][0:amount_of_state]).to(device)
        self.states[1] = torch.from_numpy(shared_array_orientation_np[self.processID][self.robotID][0:amount_of_state]).to(device)
        self.states[2] = torch.from_numpy(shared_array_distance_np[self.processID][self.robotID][0:amount_of_state]).to(device)
        self.states[3] = torch.from_numpy(shared_array_velocity_np[self.processID][self.robotID][0:amount_of_state]).to(device)
        self.actions = shared_array_action_np[self.processID][self.robotID][0:amount_of_action]
        self.rewards = shared_array_reward_np[self.processID][self.robotID][0:amount_of_rewards]
        self.is_terminals = shared_array_terminal_np[self.processID][self.robotID][0:amount_of_terminals]
        self.logprobs = shared_array_logprob_np[self.processID][self.robotID][0:amount_of_logprobs]
    def copyToShm(self):
        for i in range(len(self.states)):
            shared_array_laser_np[self.processID][self.robotID][i] = np.copy(self.states[i][0].detach().numpy())
            shared_array_orientation_np[self.processID][self.robotID][i] = np.copy(self.states[i][1].detach().numpy())
            shared_array_distance_np[self.processID][self.robotID][i] = np.copy(np.expand_dims(self.states[i][2].detach().numpy(), axis=-1))
            shared_array_velocity_np[self.processID][self.robotID][i] = np.copy(self.states[i][3].detach().numpy())


            shared_array_action_np[self.processID][self.robotID][i] = np.copy(np.ndarray(self.actions[i]))
            shared_array_reward_np[self.processID][self.robotID][i] = np.copy(np.ndarray(self.rewards[i]))
            shared_array_terminal_np[self.processID][self.robotID][i] = np.copy(np.ndarray(self.is_terminals[i]))
            shared_array_logprob_np[self.processID][self.robotID][i] = np.copy(np.ndarray(self.logprobs[i]))

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        shared_array_counter[self.processID][self.robotID][0] = 0
        shared_array_counter[self.processID][self.robotID][1] = 0
        shared_array_counter[self.processID][self.robotID][2] = 0
        shared_array_counter[self.processID][self.robotID][3] = 0
        shared_array_counter[self.processID][self.robotID][4] = 0

    def __len__(self):
        return len(self.states)
#class SwarmMemory():
 #   def __init__(self, processID=-1, robotsCount=-1):
 #       self.processID = processID
 #       self.robotMemory = [Memory(self.processID, i) for i in range(robotsCount)]
 #       self.currentTerminalStates = [False for _ in range(robotsCount)]
 #       self.stateCounter = 0
 #       self.actionCounter = 0
 #       self.rewardCounter = 0
 #       self.logprobCounter = 0
 #       self.terminalCounter = 0

 #   def __getitem__(self, item):
 #       return self.robotMemory[item]

    # Gets relative Index according to currentTerminalStates
 #   def getRelativeIndices(self):
 #       relativeIndices = []
 #       for i in range(len(self.currentTerminalStates)):
 #           if not self.currentTerminalStates[i]:
 #               relativeIndices.append(i)

 #       return relativeIndices

 #   def insertState(self, laser, orientation, distance, velocity):
 #       relativeIndices = self.getRelativeIndices()
 #       for i in range(len(relativeIndices)):
 #           shared_array_laser_np[self.processID][self.stateCounter] = laser[i]
 #           shared_array_distance_np[self.processID][self.stateCounter] = distance[i]
 #           shared_array_orientation_np[self.processID][self.stateCounter] = orientation[i]
 #           shared_array_velocity_np[self.processID][self.stateCounter] = velocity[i]
 #           self.stateCounter += 1

 #   def insertAction(self, action):
 #       relativeIndices = self.getRelativeIndices()
 #       for i in range(len(relativeIndices)):
 #           shared_array_laser_np[self.processID][self.actionCounter] = action[i]
 #           self.actionCounter += 1

 #   def insertReward(self, reward):
 #       relativeIndices = self.getRelativeIndices()
 #       for i in range(len(relativeIndices)):
 #           shared_array_laser_np[self.processID][self.rewardCounter] = reward[i]
 #           self.rewardCounter += 1

 #   def insertLogProb(self, logprob):
 #       relativeIndices = self.getRelativeIndices()
 #       for i in range(len(relativeIndices)):
 #           shared_array_laser_np[self.processID][self.logprobCounter] = logprob[i]

 #   def insertIsTerminal(self, isTerminal):
 #       relativeIndices = self.getRelativeIndices()
 #       for i in range(len(relativeIndices)):
 #           shared_array_terminal_np[self.processID][self.terminalCounter] = isTerminal[i]
 #           if isTerminal[i]:
 #               self.currentTerminalStates[relativeIndices[i]] = True

        # check if currentTerminalStates is all True
 #       if all(self.currentTerminalStates):
 #           self.currentTerminalStates = [False for _ in range(len(self.currentTerminalStates))]

 #   def getStatesOfAllRobots(self):
 #       return [torch.from_numpy(shared_array_laser_np[:, :, :, :]), torch.from_numpy(shared_array_orientation_np[:, :, :, :]),
 #               torch.from_numpy(shared_array_distance_np[:, :, :, :]), torch.from_numpy(shared_array_velocity_np)[:, :, :, :]]

 #   def getActionsOfAllRobots(self):
 #       temp = torch.from_numpy(shared_array_action_np)
 #       return temp

 #   def getLogProbsOfAllRobots(self):
 #       return shared_array_logprob_np

 #   def clear_memory(self):
 #       for memory in self.robotMemory:
 #           memory.clear_memory()

 #   def __len__(self):
 #       length = 0
 #       for memory in self.robotMemory:
 #           length += len(memory)
 #       return length


#class Memory:   # collected from old policy
  #  def __init__(self, processID, robotID):
  #      self.processID = processID
  #      self.robotID = robotID
  #      if self.processID == -1:
  #          self.states = [shared_array_laser_np[processID][robotID], shared_array_distance_np[processID][robotID],
  #                         shared_array_orientation_np[processID][robotID], shared_array_velocity_np[processID][robotID]]
  #          self.actions = [shared_array_action_np[processID][robotID]]
  #          self.rewards = [shared_array_reward_np[processID][robotID]]
  #          self.is_terminals = [shared_array_terminal_np[processID][robotID]]
  #          self.logprobs = [shared_array_logprob_np[processID][robotID]]
  #      else:
  #          self.states = [shared_array_laser_np, shared_array_distance_np, shared_array_orientation_np, shared_array_velocity_np]
  #          self.actions = shared_array_action_np
  #          self.rewards = shared_array_reward_np
  #          self.is_terminals = shared_array_terminal_np
  #          self.logprobs = shared_array_logprob_np



 #   def __len__(self):
 #       return len(self.states)


def train(env_name, render, solved_reward, input_style,
          max_episodes, max_timesteps, update_experience, action_std, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, restore, scan_size=121, print_interval=10, save_interval=100, batch_size=1,
          numOfRobots=4, args=None):
    args_ = args

    os.system("sudo mkdir && /mnt/ramdisk sudo mount -t tmpfs -o size=2G ramdisk /mnt/ramdisk")
    #os.system("sudo mkdir /mnt/ramdisk/weights")


    numOfProcesses = getNumOfProcesses(len(args_.level_files))
    print("Starting {} processes!\n".format(numOfProcesses))

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    tSteps = 4
    create_shared_memory_nparray(numOfProcesses, numOfRobots, update_experience, tSteps)

    print("Start parallel training")
    print("####################")

    try:
        multiprocessing.set_start_method('spawn', force=True)
        futures = []
        episodes_counter = 0
        timesteps_counter = 0
        #queue = [multiprocessing.Queue() for i in range(numOfProcesses)]
        pool = ProcessPoolExecutor(max_workers=numOfProcesses)
        for i in range(0, numOfProcesses):
            futures.append(pool.submit(runMultiprocessPPO, args=(i, max_episodes, env_name, max_timesteps, render,
                                                                 print_interval, solved_reward, ckpt_folder, scan_size,
                                                                 action_std, input_style, lr, betas, gamma, K_epochs,
                                                                 eps_clip, restore, ckpt, args_, numOfProcesses, update_experience, tSteps)))
        while True:
            for i in range(numOfProcesses):
                while shared_array_signal_np[i] == 0:
                    pass
            timesteps_counter += 1000
            if timesteps_counter == max_timesteps:
                timesteps_counter = 0
                if episodes_counter == max_episodes:
                    break
                episodes_counter += 1

            # Train
            memory = SwarmMemory(processID=-1, robotsCount=numOfRobots) #-1 load from shm
            pth = train_all(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore, ckpt, batch_size, memory)

            # Save .pth
            torch.save(pth, "/mnt/ramdisk/weights")

            for i in range(numOfProcesses):
                shared_array_signal_np[i] == 0

    except Exception as e:
        print(e)
    done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    print("####################")
    print("Done!")



    pool.shutdown()
    close_shm()
    exit()

def train_all(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore, ckpt, batch_size, memory):
    print("Start training!")
    ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt)
    ppo.update(memory, batch_size)
    print("Training done!")
    return ppo.old_policy.state_dict()



def init_shm_client(numOfProcesses, numOfRobots, learning_size, timesteps):
    global shared_array_laser
    global shared_array_distance
    global shared_array_orientation
    global shared_array_velocity
    global shared_array_action
    global shared_array_reward
    global shared_array_logprob
    global shared_array_terminal
    global shared_array_signal
    global shared_array_counter

    global shared_array_laser_np
    global shared_array_distance_np
    global shared_array_orientation_np
    global shared_array_velocity_np
    global shared_array_action_np
    global shared_array_reward_np
    global shared_array_logprob_np
    global shared_array_terminal_np
    global shared_array_signal_np
    global shared_array_counter_np


    size_of_laser = 1081
    size_of_distance = 1
    size_of_orientation = 2
    size_of_velocity = 2
    size_of_action = 2
    size_of_reward = 1
    size_of_logprob = 1
    size_of_terminal = 1


    array_size_laser = numOfProcesses * numOfRobots * size_of_laser * learning_size * timesteps * 4  # Sizeof(float)
    array_size_distance = numOfProcesses * numOfRobots * size_of_distance * learning_size * timesteps * 4  # Sizeof(float)
    array_size_orientation = numOfProcesses * numOfRobots * size_of_orientation * learning_size * timesteps * 4  # Sizeof(float)
    array_size_velocity = numOfProcesses * numOfRobots * size_of_velocity * learning_size * timesteps * 4  # Sizeof(float)
    array_size_action = numOfProcesses * numOfRobots * size_of_action * learning_size * 4  # Sizeof(float)
    array_size_reward = numOfProcesses * numOfRobots * size_of_reward * learning_size * 4  # Sizeof(float)
    array_size_logprob = numOfProcesses * numOfRobots * size_of_logprob * learning_size * 4  # Sizeof(float)
    array_size_terminal = numOfProcesses * numOfRobots * size_of_terminal * learning_size * 4  # Sizeof(float)
    array_size_signal = numOfProcesses * 4 #Sizeof(int)
    num_of_numpy = 5  # Number of different numpy arrays
    array_size_counter = numOfProcesses * numOfRobots * num_of_numpy * 4  # 4 because of np.int32

    shape_laser = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_laser)
    shape_distance = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_distance)
    shape_orientation = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_orientation)
    shape_velocity = (numOfProcesses, numOfRobots, learning_size, timesteps, size_of_velocity)
    shape_action = (numOfProcesses, numOfRobots, learning_size, size_of_action)
    shape_reward = (numOfProcesses, numOfRobots, learning_size, size_of_reward)
    shape_logprob = (numOfProcesses, numOfRobots, learning_size, size_of_logprob)
    shape_terminal = (numOfProcesses, numOfRobots, learning_size, size_of_terminal)
    shape_signal = (numOfProcesses,)
    shape_counter = (numOfProcesses, numOfRobots, num_of_numpy)

    shared_array_laser = shared_memory.SharedMemory(name="shared_array_laser")
    shared_array_distance = shared_memory.SharedMemory(name="shared_array_distance")
    shared_array_orientation = shared_memory.SharedMemory(name="shared_array_orientation")
    shared_array_velocity = shared_memory.SharedMemory(name="shared_array_velocity")
    shared_array_action = shared_memory.SharedMemory(name="shared_array_action")
    shared_array_reward = shared_memory.SharedMemory(name="shared_array_reward")
    shared_array_logprob = shared_memory.SharedMemory(name="shared_array_logprob")
    shared_array_terminal = shared_memory.SharedMemory(name="shared_array_terminal")
    shared_array_signal = shared_memory.SharedMemory(name="shared_array_signal")
    shared_array_counter = shared_memory.SharedMemory(name="shared_array_counter")

    np_data_type = np.float32
    shared_array_laser_np = np.ndarray(shape_laser, dtype=np_data_type, buffer=shared_array_laser.buf)
    shared_array_distance_np = np.ndarray(shape_distance, dtype=np_data_type, buffer=shared_array_distance.buf)
    shared_array_orientation_np = np.ndarray(shape_orientation, dtype=np_data_type, buffer=shared_array_orientation.buf)
    shared_array_velocity_np = np.ndarray(shape_velocity, dtype=np_data_type, buffer=shared_array_velocity.buf)
    shared_array_action_np = np.ndarray(shape_action, dtype=np_data_type, buffer=shared_array_action.buf)
    shared_array_reward_np = np.ndarray(shape_reward, dtype=np_data_type, buffer=shared_array_reward.buf)
    shared_array_logprob_np = np.ndarray(shape_logprob, dtype=np_data_type, buffer=shared_array_logprob.buf)
    shared_array_terminal_np = np.ndarray(shape_terminal, dtype=np.bool, buffer=shared_array_terminal.buf)
    shared_array_signal_np = np.ndarray(shape_signal, dtype=np.int32, buffer=shared_array_signal.buf)
    shared_array_counter_np = np.ndarray(shape_counter, dtype=np.int32, buffer=shared_array_counter.buf)


def runMultiprocessPPO(args):
    processID, max_episodes, env_name, max_timesteps, render, print_interval, solved_reward, ckpt_folder, scan_size, \
    action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore, ckpt, args_obj, numOfProcesses, \
    batch_size, tSteps = args

    global shared_array_signal_np
    try:
        print("Hello from Process #{}".format(processID))

        app = None
        env = None
        ppo = None
        memory = None

        #if processID == 0:
        app = QApplication(sys.argv)


        env = Environment(app, args_obj, args_obj.time_frames, processID)

        init_shm_client(numOfProcesses, env.getNumberOfRobots(), batch_size, tSteps)

        ckpt = ckpt_folder + '/PPO_continuous_' + env_name + '.pth'

        ppo = PPO(scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt)
        memory = SwarmMemory(processID=processID, robotsCount=env.getNumberOfRobots())


        running_reward, avg_length, time_step = 0, 0, 0
        best_reward = 0
        print("Starting training loop of Process #{}".format(processID))
        # training loop
        update_experience = 1000 #Shoudl be in args

        for i_episode in range(1, max_episodes + 1):
            states = env.reset(0)
            for t in range(max_timesteps):
                time_step += 1

                # Run old policy

                actions = ppo.select_action(states, memory)

                states, rewards, dones, _ = env.step(actions)

                memory.insertReward(rewards)
                memory.insertIsTerminal(dones)

                if len(memory) >= update_experience:
                    memory.copyToShm()
                    shared_array_signal_np[processID] = 1
                    while shared_array_signal_np[processID] == 1:
                        pass
                    memory.clear_memory()

                    #Load .pth
                    ppo.old_policy.load_state_dict(torch.load("/mnt/ramdisk/weights"))


                running_reward += np.mean(rewards)
                if render:
                    env.render()
                if env.is_done():
                    break

            avg_length += t
    except Exception as e:
        print("Exception from process #{}: {}".format(processID, e))

    print("End of process #{}".format(processID))


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
                time_step, episode_reward = 0, 0
                break

    print('Test {} episodes DONE!'.format(test_episodes))
    print('Avg episode reward: {} | Avg length: {}'.format(avg_episode_reward/test_episodes, avg_length/test_episodes))