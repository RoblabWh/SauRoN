from Environment.Simulation import Simulation

import math
import numpy as np


class Environment:
    """
    Defines the environment of the reinforcement learning algorithm
    """

    def __init__(self, app, args, timeframes, level):
        """
        :param app: PyQt5.QtWidgets.QApplication
        :param args: args defined in main
        :param timeframes: int -
            the amount of frames saved as a history by the robots to train the neural net
        """

        self.args = args
        self.steps = args.steps
        self.steps_left = args.steps

        self.simulation = Simulation(app, args, timeframes, level)
        self.timeframs = timeframes
        self.total_reward = 0.0
        self.done = False
        self.shape = np.asarray([0]).shape
        self.piFact = 1 / math.pi


    def get_observation(self, i):
        """
        Get observation from the i-th robot depending which mode is activated in the main

        - either the global state including the robots position, the orientation, linear and angular velocity
        and the goal position and the distance between the robot and the goal
        - or the local state including the data from the sonar, the linear and angular velocity,
        the orientation towards the goal and the distance to the goal

        :param i: i-th robot
        :return: returns either the global or the local (sonar) state
        """
        if self.args.mode == 'global':
            return np.asarray(self.simulation.robots[i].state)  # Pos, Geschwindigkeit, Zielposition
        #elif self.args.mode == 'sonar':
        else:
            reversed = True #determines the order of the state (reversed = false : current state in last place and the oldest at Index 0)
            #return np.asarray(self.simulation.robots[i].get_state_lidar(reversed))  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel
            return self.simulation.robots[i].get_state_lidar(reversed)  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel

    @staticmethod
    def get_actions():
        """
        :return:

        In a discrete action space this method returns all discrete actions that can be taken in the world
        e.g. action number 0 represents turning left.

        In a continuous action space this method returns the number of actions for linear and angular velocity.
        """

        return 2#[0, 1, 2, 3]  # Links, Rechts, Oben, Unten

    def is_done(self):
        """
        Checks whether all robots are done (either crashed with a wall, crashed with another robot, reached their goal or
        the steps have run out) so the simulation continues as long as at least one robot is still active or the steps
        have not been used up.
        :return: returns True if every robot has finished or the steps have run out
        """

        robotsDone = True
        for robot in self.simulation.robots:
            if robot.isActive():
                robotsDone = False
                break
        return self.steps_left <= 0 or robotsDone

    def step(self, actions, activations = None, proximity = None):
        """
        Executes a step in the environment and updates the simulation

        :param actions: list of all actions of every robot to take in this step
        :return: returns the new state of every robot in a list called robotsDataCurrentFrame
        """

        self.steps_left -= 1


        ######## Update der Simulation #######

        robotsTermination = self.simulation.update(actions, self.steps_left, activations, proximity)

        states = []
        rewards = []
        dones = []
        reachedPickups = []

        for i, termination in enumerate(robotsTermination):
            if termination != (None, None, None):
                state, reward, done, reachedPickup = self.extractRobotData(i, robotsTermination[i])
                states.append(state)
                rewards.append(reward)
                dones.append(done)
                reachedPickups.append(reachedPickup)
            else:
                # set robot to inactive if it has crashed with a wall or another robot
                pass

        return states, rewards, dones, reachedPickups

    def extractRobotData(self, i, terminations):
        """
        Extracts state and calculates reward for the i-th robot
        :param i: i-th robot
        :param terminations: tuple - terminations of a single robot
            (Boolean collision with walls or other robots, Boolean reached PickUp, Boolean runOutOfTime)
        :return: tuple (list next state, float reward, Boolean is robot done, Boolean has reached goal)
        """
        robot = self.simulation.robots[i]
        collision, reachedPickup, runOutOfTime = terminations

        ############ State Robot i ############
        next_state = self.get_observation(i)

        ############ Euklidsche Distanz und Orientierung ##############

        goal_pos_x = robot.getGoalX()
        goal_pos_y = robot.getGoalY()
        robot_pos_old_x = robot.getLastPosX()
        robot_pos_old_y = robot.getLastPosY()

        robot_pos_current_x = robot.getPosX()
        robot_pos_current_y = robot.getPosY()

        distance_old = math.sqrt((robot_pos_old_x - goal_pos_x) ** 2 +
                                 (robot_pos_old_y - goal_pos_y) ** 2)
        distance_new = math.sqrt(
            (robot_pos_current_x - goal_pos_x) ** 2 +
            (robot_pos_current_y - goal_pos_y) ** 2)

        ########### REWARD CALCULATION ################

        reward = self.createReward(robot, distance_new, distance_old, reachedPickup, collision)

        return [next_state, reward, not robot.isActive(), reachedPickup]

    def createReward(self, robot, dist_new, dist_old, reachedPickup, collision):
        """
        Creates a (sparse) reward based on the euklidian distance, if the robot has reached his goal and if the robot
        collided with a wall or another robot.

        :param robot: robot
        :param dist_new: the new distance (after the action has been taken)
        :param dist_old: the old distance (before the action has been taken)
        :param reachedPickup: True if the robot reached his goal in this step
        :param collision: True if the robot collided with a wall or another robot
        :return: returns the result of the fitness function
        """

        reward = 0
        r_arrival = 20
        r_collision = -15
        w_g = 1.5
        w_w = -0.01

        if reachedPickup:
            reward += r_arrival
        else:
            reward += w_g * (dist_old - dist_new)

        if collision:
            reward += r_collision

        abs_ang_vel = np.abs(robot.getAngularVelocity())
        if abs_ang_vel > 0.7:
            reward += w_w * abs_ang_vel

        return reward

    def reset(self, level):
        """
        Resets the simulation after each epoch
        :param level: int - defines the reset level
        """
        self.simulation.reset(level)
        self.steps_left = self.steps
        self.total_reward = 0.0
        self.done = False

        states = [self.get_observation(i) for i in range(self.simulation.getCurrentNumberOfRobots())]
        return states

    def setUISaveListener(self, observer):
        """
        Sets a Listener for the save net button to detect if it's been pressed
        :param observer: observing learning algorithm
        """
        if self.simulation.hasUI:
            self.simulation.simulationWindow.setSaveListener(observer)

    # returns number of robots in the simulation
    def getNumberOfRobots(self):
        return self.simulation.getCurrentNumberOfRobots()


