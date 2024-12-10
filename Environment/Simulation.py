import Environment.SVGParser as SVGParser
import Visualization.EnvironmentWindow as SimulationWindow
from Environment.LevelManager import LevelManager
import math, random
import numpy as np

closedFirst = False


class Simulation:
    """
    Defines the simulation with different levels for the robots to train in
    """

    def __init__(self, app, args):
        """
        Creates a simulation containing the robots, stations, levels and simulation window
        :param app: PyQt5.QtWidgets.QApplication
        :param args:
            args defined in main
        :param timeframes: int -
            the amount of frames saved as a history by the robots to train the neural net
        """
        global closedFirst
        closedFirst = False
        self.args = args
        self.levelManager = LevelManager(args.level_files)
        self.robots = None

        # Skalierungsparameter für Visualisierung
        self.scaleFactor = args.scale_factor
        self.steps = args.steps
        self.hasUI = app is not None
        self.episode = 0

        self.simulationWindow = None
        self.reset()

        if self.hasUI:
            self.simulationWindow = SimulationWindow.SimulationWindow(app, self.args, self.levelManager)
            self.simulationWindow.show()

        self.simTime = 0  # s
        self.simTimestep = args.sim_time_step  # s

    def reset(self):
        """
        Resets the simulation after each epoch
        :param level: int - defines the reset level
        """

        self.levelManager.load_level(self.args)
        self.robots = self.levelManager.robots

        # Resetting each Agent's position and orientation and randomize Stations
        rng_robot_positions = self.levelManager.get_randomized_robot_positions()
        rng_stations = self.levelManager.randomize_stations()
        for i, r in enumerate(self.robots):
            r.reset(self.levelManager.stations, rng_robot_positions[i], random.uniform(0, math.pi), self.levelManager.get_walls(), goalStation=rng_stations[i])

        # Resetting each Agent's Lidar
        for robot in self.robots:
            robot.resetLidar(self.robots)

        # Reprint the level if it has changed and the simulation window is open
        if self.hasUI and self.simulationWindow is not None:
            self.simulationWindow.levelChange(self.levelManager)

    def update(self, robotsTarVels, stepsLeft, activations, proximity):
        """
        updates the robots and checks the exit conditions of the current epoch
        :param robotsTarVels: List of tuples of target linear and angular velocity for each robot
        :param stepsLeft: steps left in current epoch
        :return: list of tuples for each robot -
            (Boolean collision with walls or other robots, Boolean reached PickUp, Boolean runOutOfTime)
        """

        # self.plotterWindow.plot(self.robot.getLinearVelocity(), self.simTime)
        # self.plotterWindow.plot(self.robot.getAngularVelocity(), self.simTime)
        self.simTime += self.simTimestep
        #time.sleep(self.simTimestep)
        # schöner ??!
        relativeIndices = []
        missing = 0
        for i, robot in enumerate(self.robots):
            if robot.isActive():
                relativeIndices.append(i - missing)
            else:
                missing += 1
                relativeIndices.append(None)
        ####

        for i, robot in enumerate(self.robots):
            if robot.isActive():
                tarLinVel, tarAngVel = robotsTarVels[relativeIndices[i]]
                self.robots[i].update(self.simTimestep, tarLinVel, tarAngVel)

        for i, robot in enumerate(self.robots):
            # watch this ?!
            if robot.isActive():
                robot.lidarReading(self.robots, stepsLeft, self.steps)

        robotsTerminations = []
        for robot in self.robots:
            if robot.isActive():
                collision = False
                reachedPickUp = False
                runOutOfTime = False

                if stepsLeft <= 0:
                    runOutOfTime = True

                if (np.min(robot.collisionDistances) <= robot.radius + 0.0):
                    collision = True

                if robot.collideWithTargetStationCircular():
                    reachedPickUp = True

                if collision or reachedPickUp or runOutOfTime:
                    robot.deactivate()

                robotsTerminations.append((collision, reachedPickUp, runOutOfTime))

            else:
                robotsTerminations.append((None, None, None))

        if self.hasUI:
            if self.simulationWindow is not None:
                for i, robot in enumerate(self.robots):
                    activationsR = activations[i] if activations is not None else None
                    self.simulationWindow.updateRobot(robot, i, activationsR)
                self.simulationWindow.updateTrafficLights(proximity)
                self.simulationWindow.paintUpdates()
                self.simulationWindow.updateInfotext(self.steps - stepsLeft, self.episode)
        return robotsTerminations

    def getCurrentNumberOfRobots(self):
        return len(self.robots)

    def updateTrainingCounter(self, counter):
        if self.hasUI:
            self.simulationWindow.updateTrainingInfotext(counter)

    # def isFarEnoughApart(self, stationPositions, randPos, minDist):
    #     """
    #     Checks whether random placed stations are far enough apart so the stations don't overlap
    #     and there is enough space for the robots to pass between them
    #
    #     :param stationPositions: (float, float) random X, random Y in the limits of the width and height of the ai-arena
    #     :param randPos: (float, float) another random X, random Y in the limits of the width and height of the ai-arena to check whether
    #     that position is far enough apart from the first one
    #     :param minDist: the minimum distance between the two stations
    #     :return: returns True if the distance between the two positions is greater than the minDist
    #
    #     """
    #     for pos in stationPositions:
    #         dist = math.sqrt((pos[0] - randPos[0]) ** 2 + (pos[1] - randPos[1]) ** 2)
    #         if (dist < minDist):
    #             return False
    #     return True
