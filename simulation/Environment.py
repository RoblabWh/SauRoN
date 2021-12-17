import math
from simulation.Simulation import Simulation
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
        elif self.args.mode == 'sonar':
            reversed = self.args.load_christian
            return np.asarray(self.simulation.robots[i].get_state_lidar(reversed))  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel


    def getRobotsProximityCategoryAllObstacles(self, i):
        distances = self.simulation.robots[i].collisionDistances
        if len(distances) > 0:
            min = np.min(distances)
            if min > 2: return 0 #[1,0,0]
            elif min > 0.75: return 1 #[0,1,0]
            else: return 2 #[0,0,1]
        return 0

    def getRobotsProximityCategoryOnlyRobots(self, i):
        distances = self.simulation.robots[i].collisionDistancesRobots
        if len(distances) > 0:
            min = np.min(distances)
            if min > 3: return 0 #[1,0,0]
            elif min > 1.25: return 1 #[0,1,0]
            else: return 2 #[0,0,1]
        return 0

    def getAngle(self, i, laserRange=-45):
        angle_min = np.radians(laserRange)
        angle_increment = np.radians(0.25)
        return angle_min + (i * angle_increment)

    def pol2cart(self, rho, phi): #forward axis looking up (positive y)
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return [x, y]


    def getMinDistOnVirtualRoadway(self, i):
        scan = self.simulation.robots[i].distances
        return self.getMinDistOnVirtualRoadwayWithScan(i, scan)

    def getMinDistOnVirtualRoadwayWithScan(self, i, scan):
        # scan = self.simulation.robots[i].distances
        scanTransformed = []
        for j, point in enumerate(scan):
            angle = self.getAngle(j)
            scanTransformed.append(self.pol2cart(point, angle))
        scanTransformed = np.asarray(scanTransformed)
        x = np.logical_and(scanTransformed[:, 0] > -0.35, scanTransformed[:, 0] < 0.35)
        roadwayIndexSelection = np.argwhere(np.logical_and(x, scanTransformed[:, 1] > 0))

        min = np.min(scan[roadwayIndexSelection])
        if min > 3:
            if i == 0: print(0, self.steps- self.steps_left, min)
            return 0 #[1,0,0]
        elif min > 1.25:
            if i == 0: print(1, self.steps- self.steps_left, min)
            return 1 #[0,1,0]
        else:
            if i == 0: print(2, self.steps- self.steps_left, min)
            return 2 #[0,0,1]


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
        robotsDataCurrentFrame = []
        for i, termination in enumerate(robotsTermination):
            if termination != (None, None, None):
                robotsDataCurrentFrame.append(self.extractRobotData(i, robotsTermination[i]))
            else:
                robotsDataCurrentFrame.append((None, None, None))

        return robotsDataCurrentFrame

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
        next_state = np.expand_dims(next_state, axis=0)


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

        return (next_state, reward, not robot.isActive(), reachedPickup)

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

        distPos = 0.02#0.009#0.015
        distNeg = -0.002#0.002  # in Masterarbeit alles = 0 außer distPos (mit 0.1)
        oriPos = 0.0001#0.0001#.0003
        oriNeg = -0.0002#-0.00002#0.00002
        lastDistPos = 0.000005
        unblViewPos = 0.003

        deltaDist = dist_old - dist_new

        if deltaDist > 0:
            rewardDist = deltaDist * distPos
        else:
            rewardDist = deltaDist * distNeg #Ein Minus führt zu geringer Belohnung (ohne Minus zu einer geringen Strafe)

        angularDeviation = (abs(robot.angularDeviation * self.piFact) *-2) +1

        if angularDeviation > 0:
            rewardOrient = angularDeviation * oriPos
        else:
            rewardOrient = angularDeviation * oriNeg #Dieses Minus führt zu geringer Belohnung (ohne Minus zu einer geringen Strafe)

        unblockedViewDistance = (-0.1 / (robot.distances[int(len(robot.distances)/2)] * robot.maxDistFact)) * unblViewPos

        lastBestDistance = robot.bestDistToGoal
        distGoal = dist_new
        rewardLastDist = 0

        if distGoal < lastBestDistance:
            rewardLastDist = (lastBestDistance - distGoal) * lastDistPos
            robot.bestDistToGoal = distGoal

        if collision:
            reward = -1
        elif reachedPickup:
            reward = 1 #+ rewardDist + rewardOrient + unblockedViewDistance # + rewardLastDist
        else:
            reward = rewardDist + rewardOrient + unblockedViewDistance #+ rewardLastDist #+  unblockedViewDistance
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

    def setUISaveListener(self, observer):
        """
        Sets a Listener for the save net button to detect if it's been pressed
        :param observer: observing learning algorithm
        """
        if self.simulation.hasUI:
            self.simulation.simulationWindow.setSaveListener(observer)

