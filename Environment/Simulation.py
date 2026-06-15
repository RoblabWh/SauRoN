#import Environment.SVGParser as SVGParser
from Environment.LevelManager import LevelManager
from Environment.FlowField import FlowFieldCache
import math, random
import numpy as np


class Simulation:
    """
    Defines the simulation with different levels for the robots to train in
    """

    def __init__(self, args, curriculum_func=None):
        """
        Creates a simulation containing the robots, stations and levels
        :param args:
            args defined in main
        :param curriculum_func:
            optional callback choosing the next level (see LevelManager)
        """
        self.args = args
        self.levelManager = LevelManager(args.level_files, curriculum_func=curriculum_func)
        self.robots = None
        # geodesic fields are memoized per (level, goal): station positions are
        # fixed within a level, so each field is built exactly once per run
        self.flowFields = FlowFieldCache() if getattr(args, 'geodesic_reward', True) else None

        self.steps = args.steps
        self.episode = 0

        self.viz = None
        if args.visualization == 'rerun':
            from Visualization.RerunVisualizer import RerunVisualizer
            self.viz = RerunVisualizer(args)

        self.reset()

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

        # Geodesic distance field per goal station: the reward shapes on
        # shortest-path distance instead of straight-line distance, so detours
        # around obstacles are paid correctly. Observations stay map-free.
        if self.flowFields is not None:
            walls = [(w.getStart(), w.getEnd()) for w in self.levelManager.get_walls()]
            circles = [(c.getPosX(), c.getPosY(), c.getRadius()) for c in self.levelManager.circleWalls]
            robotRadius = max(r.radius for r in self.robots)
            cell = getattr(self.args, 'flow_field_cell', 0.1)
            for robot in self.robots:
                station = robot.station
                # seed the zero-distance region with the actual success disk
                # (collideWithTargetStationCircular: center dist < 1.2 * r_station
                # - r_robot), not the station radius -- otherwise the shaping has
                # a zero-gradient plateau before arrival actually triggers
                seedRadius = max(cell, 1.2 * station.getRadius() - robot.radius)
                robot.flowField = self.flowFields.get(
                    self.levelManager.get_current_level(), walls, circles,
                    station.getPosX(), station.getPosY(), seedRadius,
                    robotRadius, cell,
                    store=not self.levelManager.current_is_generated)
                # high-water mark of the progress reward must start geodesic too
                startDist = robot.flowField.distance(robot.getPosX(), robot.getPosY())
                if startDist is not None:
                    robot.bestGoalDist = startDist
                    robot.bestGoalDistIsGeodesic = True

        # Log the (possibly changed) level geometry to the viewer
        if self.viz is not None:
            self.viz.log_level(self.levelManager, self.robots)
            if self.flowFields is not None:
                self.viz.log_flow_field(self.robots)

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

        if self.viz is not None:
            self.viz.log_step(self.robots)
        return robotsTerminations

    def getCurrentNumberOfRobots(self):
        return len(self.robots)

    def updateTrainingCounter(self, counter):
        pass

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
