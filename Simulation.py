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

        # # Erstelle Stationen und Roboter
        # self.pickUp = Station(5, 1.2, 1, 0, self.scaleFactor)
        # self.pickUp2 = Station(1.15, 1.25, 1, 1, self.scaleFactor)
        # self.pickUp3 = Station(9, 0.8, 1, 2, self.scaleFactor)
        # self.pickUp4 = Station(13, 1.3, 1, 3, self.scaleFactor)
        # self.stations = [self.pickUp, self.pickUp2, self.pickUp3, self.pickUp4]
        #
        # self.robot = Robot.Robot((10.5, 8.8), 3.2*math.pi/2, self.pickUp, args, self.walls, self.stations)
        # self.robot2 = Robot.Robot((4, 8.6), 2.6*math.pi/2, self.pickUp2, args, self.walls, self.stations)
        # self.robot3 = Robot.Robot((1.1, 8.9), 3.6*math.pi/2, self.pickUp3, args, self.walls, self.stations)
        # self.robot4 = Robot.Robot((13.1, 3.9), 3.6*math.pi/2, self.pickUp4, args, self.walls, self.stations)
        #
        # self.robots = [self.robot, self.robot2, self.robot3, self.robot4]
        # if args.numb_of_robots<= 4:
        #     self.robots = self.robots[:args.numb_of_robots]
        #
        # #Level bestehend aus Positionen für Stationen und Robotern, deren Ausrichtung und Wänden werden erstellt
        # level00_robotPos = [(self.arenaWidth / 5, self.arenaLength - 3.5),
        #                      (self.arenaWidth / 5 * 2, self.arenaLength - 3.5),
        #                      (self.arenaWidth / 5 * 3, self.arenaLength - 3.5),
        #                      (self.arenaWidth / 5 * 4, self.arenaLength - 3.5)]
        # level01_robotPos = [(self.arenaWidth / 5, self.arenaLength - 4),
        #                     (self.arenaWidth / 5 * 2, self.arenaLength - 2.5),
        #                     (self.arenaWidth / 5 * 3, self.arenaLength - 2.5),
        #                     (self.arenaWidth / 5 * 4, self.arenaLength - 4)]
        # level02_robotPos = [(1.5, 1.5),
        #                     (self.arenaWidth -1.5, 1.5),
        #                     (self.arenaWidth -1.5, self.arenaLength -1.5),
        #                     (1.5,  self.arenaLength -1.5)]
        # level03b_robotPos = [(self.arenaWidth / 16 * 5, self.arenaLength - 1.5),
        #                     (self.arenaWidth / 5 * 2, self.arenaLength - 1.5),
        #                     (self.arenaWidth / 5 * 3, self.arenaLength - 1.5),
        #                     (self.arenaWidth / 16 * 11, self.arenaLength - 1.5)]
        # level03_robotPos = [(self.arenaWidth / 5 *2, 1),
        #                     (self.arenaWidth / 5   , 1),
        #                     (self.arenaWidth / 5 *4, self.arenaLength -1),
        #                     (self.arenaWidth / 5 *3,  self.arenaLength -1)]
        # level04_robotPos = level01_robotPos
        # level05_robotPos = level02_robotPos
        # level06_robotPos = level03_robotPos
        # level07_robotPos = [(self.arenaWidth /2, self.arenaLength /2 - 3.5),
        #                     (self.arenaWidth /2 +3.5, self.arenaLength /2),
        #                     (self.arenaWidth /2, self.arenaLength / 2 + 3.5),
        #                     (self.arenaWidth /2 -3.5, self.arenaLength / 2)]
        # lvl_funnle_robPos = [(self.arenaWidth / 5, self.arenaLength - 2),
        #                     (self.arenaWidth / 5 * 2, self.arenaLength - 1.5),
        #                     (self.arenaWidth / 5 * 3, self.arenaLength - 1.5),
        #                     (self.arenaWidth / 5 * 4, self.arenaLength - 2)]
        # lvl_narrowWayWithBays_robPos = [(2,4),(2,7),(20,4),(20,7)]
        # lvl_narrowWayEasy_robPos = [(20,5.5),(2,5.5),(20,4),(20,7)]
        #
        #
        # level00_robotOrient = [math.pi/2*3 for _ in range(4)]
        # level01_robotOrient = [math.pi/4 *7, math.pi/2 *3, math.pi/2 *3, math.pi/4 *5]
        # level02_robotOrient = [math.pi/4 , math.pi/4 *3, math.pi/4 *5, math.pi/4 *7]
        # level03b_robotOrient = [math.pi/2*3 for _ in range(4)]
        # level03_robotOrient = [math.pi/4 , math.pi/4 *3, math.pi/4 *5, math.pi/4 *7]
        # level04_robotOrient = level01_robotOrient
        # level05_robotOrient = level02_robotOrient
        # level06_robotOrient = level03_robotOrient
        # level07_robotOrient = [math.pi/2 , math.pi, math.pi/2 *3, 0]
        # lvl_narrowWayWithBays_robOrient = [0,0,math.pi, math.pi]
        # lvl_narrowWayEasy_robOrient = [math.pi,0,math.pi, math.pi]
        # lvl_funnle_robotOrient = [math.pi/4 *7, math.pi/2 *3, math.pi/2 *3, math.pi/4 *5]
        #
        #
        # level00_stationsPos = [(self.arenaWidth / 5, 3.5),
        #                         (self.arenaWidth / 5 * 2, 3.5),
        #                         (self.arenaWidth / 5 * 3, 3.5),
        #                         (self.arenaWidth / 5 * 4, 3.5)]
        # level01_stationsPos = [(self.arenaWidth / 2 - 3, 2.2),
        #                        (self.arenaWidth / 2 -1.15, 3.1),
        #                        (self.arenaWidth / 2 +1.15, 3.1),
        #                        (self.arenaWidth / 2 + 3, 2.2)]
        # level02_stationsPos = [(self.arenaWidth / 2 -1, self.arenaLength / 2 -1),
        #                        (self.arenaWidth / 2 +1, self.arenaLength / 2 -1),
        #                        (self.arenaWidth / 2 +1, self.arenaLength / 2 +1),
        #                        (self.arenaWidth / 2 -1, self.arenaLength / 2 +1)]
        # level03b_stationsPos = [(self.arenaWidth / 4, 1.5),
        #                        (self.arenaWidth / 5 * 2, 1.5),
        #                        (self.arenaWidth / 5 * 3, 1.5),
        #                        (self.arenaWidth / 4 * 3, 1.5)]
        # level03_stationsPos = [(self.arenaWidth / 5   , self.arenaLength -1),
        #                        (self.arenaWidth / 5 *2, self.arenaLength -1),
        #                        (self.arenaWidth / 5 *3, 1),
        #                        (self.arenaWidth / 5 *4, 1)]
        # level04_stationsPos = [level01_stationsPos[2], level01_stationsPos[3], level01_stationsPos[0], level01_stationsPos[1]]
        # level05_stationsPos = [level02_stationsPos[2], level02_stationsPos[3], level02_stationsPos[0], level02_stationsPos[1]]
        # level06_stationsPos = level03_stationsPos
        # level07_stationsPos = [(self.arenaWidth /2, self.arenaLength / 2 + 5),
        #                        (self.arenaWidth /2 -5, self.arenaLength / 2),
        #                        (self.arenaWidth /2, self.arenaLength /2 - 5),
        #                        (self.arenaWidth /2 +5, self.arenaLength /2)]
        # lvl_funnle_stationsPos = [(self.arenaWidth / 2 - 3, 2.2),
        #                        (self.arenaWidth / 2 -1.15, 1.8),
        #                        (self.arenaWidth / 2 +1.15, 1.8),
        #                        (self.arenaWidth / 2 + 3, 2.2)]
        # lvl_narrowWayWithBays_statPos = [(21,4),(21,7),(1,4),(1,7)]
        # lvl_narrowWayEasy_statPos = [(1,5.5),(21,5.5),(1,4),(1,7)]
        #
        # noWalls = self.walls
        # level03b_walls = self.walls + [ColliderLine(self.arenaWidth / 16, self.arenaLength / 2 + 0.5, self.arenaWidth / 5 * 2, self.arenaLength / 2 + .5),
        #                                ColliderLine(self.arenaWidth / 5 * 3, self.arenaLength / 2 + 0.5, self.arenaWidth / 16 * 15, self.arenaLength / 2 + .5),
        #                                ColliderLine(self.arenaWidth / 16, self.arenaLength / 2 - 0.2, self.arenaWidth / 5 * 2, self.arenaLength / 2 - .2),
        #                                ColliderLine(self.arenaWidth / 5 * 3, self.arenaLength / 2 - 0.2, self.arenaWidth / 16 * 15, self.arenaLength / 2 - .2),
        #                                ColliderLine(self.arenaWidth / 16, self.arenaLength / 2 + 0.5, self.arenaWidth / 16, self.arenaLength / 2 - .2),
        #                                ColliderLine(self.arenaWidth / 5 * 3, self.arenaLength / 2 + 0.5, self.arenaWidth / 5 * 3, self.arenaLength / 2 - .2),
        #                                ColliderLine(self.arenaWidth / 5 * 2, self.arenaLength / 2 - 0.2, self.arenaWidth / 5 * 2, self.arenaLength / 2 + .5),
        #                                ColliderLine(self.arenaWidth / 16 * 15, self.arenaLength / 2 - 0.2, self.arenaWidth / 16 * 15, self.arenaLength / 2 + .5)]
        #
        #
        # level06_walls = self.walls + [ColliderLine(0, self.arenaLength / 3, self.arenaWidth / 24 * 6, self.arenaLength / 3),
        #                               ColliderLine(self.arenaWidth / 24 * 6, self.arenaLength / 3, self.arenaWidth / 24 * 6, self.arenaLength / 3 * 2),
        #                               ColliderLine(self.arenaWidth / 24 * 9, self.arenaLength / 3, self.arenaWidth / 24 * 9, self.arenaLength / 3 * 2),
        #                               ColliderLine(self.arenaWidth / 24 * 9, self.arenaLength / 3, self.arenaWidth / 24 * 12, self.arenaLength / 3),
        #                               ColliderLine(self.arenaWidth / 2, 0, self.arenaWidth / 2, self.arenaLength),
        #                               ColliderLine(self.arenaWidth / 24 * 12, self.arenaLength / 3 * 2, self.arenaWidth / 24 * 15, self.arenaLength / 3 * 2),
        #                               ColliderLine(self.arenaWidth / 24 * 15, self.arenaLength / 3, self.arenaWidth / 24 * 15, self.arenaLength / 3 * 2),
        #                               ColliderLine(self.arenaWidth / 24 * 18, self.arenaLength / 3, self.arenaWidth / 24 * 18, self.arenaLength / 3 * 2),
        #                               ColliderLine(self.arenaWidth / 24 * 18, self.arenaLength / 3 * 2, self.arenaWidth, self.arenaLength / 3 * 2)]
        # centerW = self.arenaWidth/2
        # centerL = self.arenaLength/2
        # level07_walls = self.walls+[ColliderLine(0, centerL - 1, centerW - 1, centerL - 1),
        #                             ColliderLine(0, centerL + 1, centerW - 1, centerL + 1),
        #                             ColliderLine(centerW + 1, centerL - 1, self.arenaWidth, centerL - 1),
        #                             ColliderLine(centerW + 1, centerL + 1, self.arenaWidth, centerL + 1),
        #                             ColliderLine(centerW - 1, 0, centerW - 1, centerL - 1),
        #                             ColliderLine(centerW + 1, 0, centerW + 1, centerL - 1),
        #                             ColliderLine(centerW - 1, centerL + 1, centerW - 1, self.arenaLength),
        #                             ColliderLine(centerW + 1, centerL + 1, centerW + 1, self.arenaLength)]
        # lvl_funnle_walls = self.walls + [ColliderLine(1, 8.5, 8, 4), ColliderLine(21, 8.5 , 14, 4)]
        #
        # lvl_narrowWayWithBays_walls = [ColliderLine(0, 3.6, 5, 3.6),
        #                                 ColliderLine(5, 3.6, 5.25, 3),
        #                                 ColliderLine(6.75, 3, 7.25, 3.6),
        #                                 ColliderLine(7.25, 3.6, 14.75, 3.6),
        #                                 ColliderLine(14.75, 3.6, 15.25, 3),
        #                                 ColliderLine(16.75, 3, 17, 3.6),
        #                                 ColliderLine(17, 3.6, 22, 3.6),
        #                                 ColliderLine(0, 7.4, 5, 7.4),
        #                                 ColliderLine(5, 7.4, 5.25, 8),
        #                                 ColliderLine(6.25, 8, 6.5, 7.4),
        #                                 ColliderLine(6.5, 7.4, 15.5, 7.4),
        #                                 ColliderLine(15.5, 7.4, 15.75, 8),
        #                                 ColliderLine(16.75, 8, 17, 7.4),
        #                                 ColliderLine(17, 7.4, 22, 7.4)] + \
        #                               SquareWall(11, 3.85, 22, 1.7, 0).getBorders() + \
        #                               SquareWall(11, 7.15, 22, 1.7, 0).getBorders()
        # lvl_narrowWayEasy_walls = lvl_narrowWayWithBays_walls + \
        #                           [ColliderLine(0, 4.4, 0, 6.6), ColliderLine(22, 4.4, 22, 6.6)]
        #
        # self.noiseStrength = [0.1, 0.2, 0.33, 0.4, 0.15, 0.4, 0.2, 0.5, 0.15]
        # self.level = [#(level00_robotPos,level00_robotOrient,level00_stationsPos, noWalls),
        #               (level01_robotPos,level01_robotOrient,level01_stationsPos, noWalls),
        #               (level02_robotPos,level02_robotOrient,level02_stationsPos, noWalls),
        #               (level03_robotPos,level03_robotOrient,level03_stationsPos, noWalls),
        #               (level03b_robotPos,level03b_robotOrient,level03b_stationsPos, level03b_walls),
        #               (level04_robotPos,level04_robotOrient,level04_stationsPos, noWalls),
        #               # (level05_robotPos,level05_robotOrient,level05_stationsPos, noWalls),
        #               (level06_robotPos,level06_robotOrient,level06_stationsPos, level06_walls),
        #               (level07_robotPos,level07_robotOrient,level07_stationsPos, level07_walls)]
        #
        # # testlevel01 = ([(18,0.5),(19,0.666),(19.5,1.16), (18.3,1.4)],[2,1.8,1.9,2.2],[(1,9),(7.75,5.5),(9,6.5),(8,9)], SquareWall(9, 5, 3.8, 0.55, 40, True).getBorders() + noWalls)
        # # testlevel02 = ([(3,0.5),(2,0.666),(2.5,1.16), (3.3,1.4)],[1,0.8,0.9,1.2],[(19,9),(13.5,5),(12,6.5),(13,9)], SquareWall(12, 5, 3.8, 0.55, 130, True).getBorders() + noWalls)
        # testlevel03 = ([(3,0.5),(2,0.666),(2.5,1.16), (3.3,1.4)],[1,0.8,0.9,1.2],[(18,0.5),(19,2),(19.5,1.16), (20,2.6)], SquareWall(10, 1.6, 4, 0.55, 90, True).getBorders() + noWalls)
        # testlevel04 = ([(18,0.5),(19,0.666),(19.5,1.16), (18.3,1.4)],[1,0.8,0.9,1.2],[(3,0.5),(2,2),(2.5,1.16), (1,2.6)], SquareWall(10, 1.6, 4, 0.55, 90, True).getBorders() + noWalls)
        # funnle = (lvl_funnle_robPos, lvl_funnle_robotOrient, lvl_funnle_stationsPos,lvl_funnle_walls)
        # narrowWayWithBays = [lvl_narrowWayWithBays_robPos, lvl_narrowWayWithBays_robOrient, lvl_narrowWayWithBays_statPos, lvl_narrowWayWithBays_walls]
        # narrowWayEasy = [lvl_narrowWayEasy_robPos, lvl_narrowWayEasy_robOrient, lvl_narrowWayEasy_statPos, lvl_narrowWayEasy_walls]
        # # self.level[1] = narrowWayWithBays
        # # self.level[2] = narrowWayEasy
        # self.level[0] = funnle
        # # self.level[2] = testlevel03
        # # self.level[4] = testlevel04
        # self.level.append(testlevel03)
        # self.level.append(testlevel04)

        # level01SVG = SVGParser.SVGLevelParser("test_v3.svg", args)
        level01SVG = SVGParser.SVGLevelParser("laborNachbauAuschnitt_kleiner.svg", args)
        self.robots = level01SVG.getRobots()
        self.stations = level01SVG.getStations()
        self.walls = level01SVG.getWalls()
        self.level = [(level01SVG.getRobsPos(), level01SVG.getRobsOrient(), level01SVG.getStatsPos(), self.walls)]
        # self.noiseStrength = [0,0,0,0,0,0,0,0,0,0,0,0]

        self.simulationWindow = None

        self.reset(level)

        if self.hasUI:
            self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, args, self.walls)
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
        randomSim = False
        robotsPositions = []
        orientations = []

        if(randomSim):
            stationRadius = self.stations[0].getRadius()
            robotRadius = self.robots[0].getRadius()
            clearance = 0.25
            stationPosLimitsX = [stationRadius + clearance, self.arenaWidth - stationRadius - clearance]
            stationPosLimitsY = [stationRadius + clearance, self.arenaLength - stationRadius - clearance]
            stationDistance = stationRadius * 2 + 2 * robotRadius + clearance
            robotPosLimitsX = [robotRadius + clearance, self.arenaWidth - robotRadius - clearance]
            robotPosLimitsY = [robotRadius + clearance, self.arenaLength - robotRadius - clearance]
            robotDistance = robotRadius * 2 + clearance
            robotStationDistance = robotRadius * 2 + stationRadius + clearance

            stationPositions = [ (random.uniform(stationPosLimitsX[0], stationPosLimitsX[1]),
                                  random.uniform(stationPosLimitsY[0], stationPosLimitsY[1])) ]

            for i in range(len(self.stations)-1):

                while True:
                    randPos = (random.uniform(stationPosLimitsX[0], stationPosLimitsX[1]),
                               random.uniform(stationPosLimitsY[0], stationPosLimitsY[1]))
                    if(self.isFarEnoughApart(stationPositions, randPos, stationDistance)):
                        stationPositions.append(randPos)
                        break

            for i, s in enumerate(self.stations):
                s.setPos(stationPositions[i])
            for i in range(len(self.robots)):
                while True:
                    randPos = (random.uniform(robotPosLimitsX[0], robotPosLimitsX[1]),
                               random.uniform(robotPosLimitsY[0], robotPosLimitsY[1]))
                    if(self.isFarEnoughApart(stationPositions, randPos, robotStationDistance)):
                        if(self.isFarEnoughApart(robotsPositions, randPos, robotDistance)):
                            robotsPositions.append(randPos)
                            break
            orientations = [random.uniform(0, math.pi*2) for _ in range(len(self.robots))]
            for i, r in enumerate(self.robots):
                r.reset(self.stations, robotsPositions[i], orientations[i])

        else:
            for i, s in enumerate(self.stations):
                s.setPos(self.level[level][2][i])
            for i, r in enumerate(self.robots):
                # r.reset(self.stations, self.level[level][0][i], self.level[level][1][i]+(random.uniform(0, math.pi)*self.noiseStrength[level]), self.level[level][3])
                r.reset(self.stations, self.level[level][0][i], self.level[level][1][i]+(random.uniform(0, math.pi)), self.level[level][3])

        for robot in self.robots:
            robot.resetSonar(self.robots)
        if self.hasUI and self.simulationWindow != None:
            self.simulationWindow.setWalls(self.level[level][3])


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


    def update(self, robotsTarVels, stepsLeft):
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
                    robot.sonarReading(self.robots, stepsLeft, self.steps)


        robotsTerminations = []
        for robot in self.robots:
            if robot.isActive():
                collision = False
                reachedPickUp = False
                runOutOfTime = False

                if stepsLeft <= 0:
                    runOutOfTime = True

                if(np.min(robot.collisionDistances) <= robot.radius + 0.035 ):
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
                    self.simulationWindow.updateRobot(robot, i, self.steps-stepsLeft)

        return robotsTerminations

