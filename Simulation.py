import Robot
import SVGParser
from Station import Station
import SimulationWindow
import math, random
import numpy as np
from old.PlotterWindow import PlotterWindow
from Borders import ColliderLine, SquareWall

class Simulation:
    """
    Defines the simulation with different levels for the robots to train in
    """

    levelFiles = ['svg0.svg', 'Lab2.svg', 'Simple.svg', 'Zipper.svg', 'Funnel.svg', 'SwapSide.svg']


    def __init__(self, app, args, timeframes, level):
        """
        Creates a simulation containing the robots, stations, levels and simulation window
        :param app: PyQt5.QtWidgets.QApplication
        :param args:
            args defined in main
        :param timeframes: int -
            the amount of frames saved as a history by the robots to train the neural net
        """
        self.args = args
        # Skalierungsparameter für Visualisierung
        self.scaleFactor = args.scale_factor
        self.steps = args.steps
        self.hasUI = app is not None

        # Parameter width & length über args
        self.arenaWidth = args.arena_width
        self.arenaLength = args.arena_length

        #Wände der Simulation
        self.walls = []
        self.walls.append(ColliderLine(0, 0, self.arenaWidth, 0))
        self.walls.append(ColliderLine(self.arenaWidth, 0, self.arenaWidth, self.arenaLength))
        self.walls.append(ColliderLine(self.arenaWidth, self.arenaLength, 0, self.arenaLength))
        self.walls.append(ColliderLine(0, self.arenaLength, 0, 0))


        self.simulationWindow = None
        self.loadLevel(level)

        self.reset(level)

        if self.hasUI:
            self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, args, self.walls, self.circleWalls, self.arenaSize)
            self.simulationWindow.show()


        self.simTime = 0  # s
        self.simTimestep = args.sim_time_step  # s
        # self.simTimestep = 0.25  # FÜR CHRISTIANS NETZ
        # self.plotterWindow = PlotterWindow(app)


    def reset(self, level):
        """
        Resets the simulation after each epoch
        :param level: int - defines the reset level
        """
        if self.levelID != level:
            self.loadLevel(level)
            levelChanged = True
        else:
            levelChanged = False

        for i, s in enumerate(self.stations):
            s.setPos(self.level[2][i])
        for i, r in enumerate(self.robots):
            # r.reset(self.stations, self.level[level][0][i], self.level[level][1][i]+(random.uniform(0, math.pi)*self.noiseStrength[level]), self.level[level][3])
            r.reset(self.stations, self.level[0][i], self.level[1][i]+(random.uniform(0, math.pi)), self.level[3])

        for robot in self.robots:
            robot.resetLidar(self.robots)
        if self.hasUI and self.simulationWindow != None:
            if levelChanged:
                self.simulationWindow.setWalls(self.level[3])
                self.simulationWindow.setRobotRepresentation(self.robots)
                self.simulationWindow.setStations(self.stations)
                self.simulationWindow.setCircleWalls(self.circleWalls)




    def isFarEnoughApart(self, stationPositions, randPos, minDist):
        """
        Checks whether random placed stations are far enough apart so the stations don't overlap
        and there is enough space for the robots to pass between them

        :param stationPositions: (float, float) random X, random Y in the limits of the width and height of the ai-arena
        :param randPos: (float, float) another random X, random Y in the limits of the width and height of the ai-arena to check whether
        that position is far enough apart from the first one
        :param minDist: the minimum distance between the two stations
        :return: returns True if the distance between the two positions is greater than the minDist

        """
        for pos in stationPositions:
            dist = math.sqrt((pos[0]-randPos[0])**2+(pos[1]-randPos[1])**2)
            if(dist<minDist):
                return False
        return True


    def getGoalWidth(self):
        """
        only for rectangular Stations
        :return: float station width in meter
        """
        goalWidth = self.pickUp.getWidth()
        return goalWidth


    def getGoalLength(self):
        """
        only for rectangular Stations
        :return: float station length in meter
        """
        goalLength = self.pickUp.getLength()
        return goalLength


    def update(self, robotsTarVels, stepsLeft, activations):
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

        for i, robot in enumerate(self.robots):
            if robot.isActive() == True:
                tarLinVel, tarAngVel = robotsTarVels[i]
                self.robots[i].update(self.simTimestep, tarLinVel, tarAngVel)


        if self.args.mode == 'sonar':
            for i, robot in enumerate(self.robots):
                if robotsTarVels[i] != (None, None):
                    robot.lidarReading(self.robots, stepsLeft, self.steps)


        robotsTerminations = []
        for robot in self.robots:
            if robot.isActive():
                collision = False
                reachedPickUp = False
                runOutOfTime = False

                if stepsLeft <= 0:
                    runOutOfTime = True

                if(np.min(robot.collisionDistances) <= robot.radius + 0.0 ):
                # if(np.min(robot.distances)<=0.3): # FÜR CHRISTIANS NETZ
                    collision = True

                if robot.collideWithTargetStationCircular():
                    reachedPickUp = True

                if collision or reachedPickUp or runOutOfTime:
                    robot.deactivate()

                robotsTerminations.append((collision, reachedPickUp, runOutOfTime))

            else:
                robotsTerminations.append((None, None, None))

        if self.hasUI:
            if self.simulationWindow != None:
                for i, robot in enumerate(self.robots):
                    activationsR = activations[i] if activations is not None else None
                    self.simulationWindow.updateRobot(robot, i, self.steps-stepsLeft, activationsR)

        return robotsTerminations

    def showWindow(self, app):
        if not self.hasUI:
            self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, self.args, self.walls, self.circleWalls, self.arenaSize)
            self.simulationWindow.show()
            self.hasUI = True


    def closeWindow(self):
        if self.hasUI:
            self.simulationWindow.close()
            self.simulationWindow = None
            self.hasUI = False

    def getCurrentNumberOfRobots(self):
        return len(self.robots)

    def loadLevel(self, levelID):
        selectedLevel = SVGParser.SVGLevelParser(self.levelFiles[levelID], self.args)
        self.robots = selectedLevel.getRobots()
        if self.args.manually:
            self.robots = self.robots[0:self.args.numb_of_robots]

        self.stations = selectedLevel.getStations()
        self.walls = selectedLevel.getWalls()
        self.circleWalls = selectedLevel.getCircleWalls()
        self.level = (selectedLevel.getRobsPos(), selectedLevel.getRobsOrient(), selectedLevel.getStatsPos(), self.walls, self.circleWalls)
        self.levelID = levelID
        self.arenaSize = selectedLevel.getArenaSize()




    def getLevelName(self):
        levelNameSVG = self.levelFiles[self.levelID]
        levelName = levelNameSVG.split('.', 1)[0]
        return levelName