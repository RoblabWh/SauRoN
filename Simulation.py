from PyQt5.QtCore import QTimer
import Robot
from Station import Station
import SimulationWindow
import math, random
import numpy as np
import time
from old.PlotterWindow import PlotterWindow
from Borders import CollidorLine

class Simulation:

    def __init__(self, app, args, timeframes):
        """
        Creates a simulation containing the robots, stations and simulation window
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

        self.walls = []
        self.walls.append(CollidorLine(0,0,self.arenaWidth, 0))
        self.walls.append(CollidorLine(self.arenaWidth, 0, self.arenaWidth, self.arenaLength))
        self.walls.append(CollidorLine(self.arenaWidth, self.arenaLength, 0, self.arenaLength))
        self.walls.append(CollidorLine(0,self.arenaLength, 0, 0))

        #TODO mehrere Robots mit eigenen Pickup stationen erstellen

        # Erstelle Stationen und Roboter
        self.pickUp = Station(5, 1.2, 1, 0, self.scaleFactor)
        self.pickUp2 = Station(1.15, 1.25, 1, 1, self.scaleFactor)
        self.pickUp3 = Station(9, 0.8, 1, 2, self.scaleFactor) #12, 4.1 gute ergebnisse mit trainierte Netz vom 19Jan  #2, 5.1 geht für das neuste
        self.pickUp4 = Station(13, 1.3, 1, 3, self.scaleFactor)
        self.stations = [self.pickUp, self.pickUp2, self.pickUp3, self.pickUp4]


        self.robot = Robot.Robot((10.5, 8.8), 3.2*math.pi/2, self.pickUp, args, timeframes, self.walls, self.stations)
        self.robot2 = Robot.Robot((4, 8.6), 2.6*math.pi/2, self.pickUp2, args, timeframes, self.walls, self.stations)
        self.robot3 = Robot.Robot((1.1, 8.9), 3.6*math.pi/2, self.pickUp3, args, timeframes, self.walls, self.stations)
        self.robot4 = Robot.Robot((13.1, 3.9), 3.6*math.pi/2, self.pickUp4, args, timeframes, self.walls, self.stations)

        self.robots = [self.robot, self.robot2, self.robot3, self.robot4]

        level00_robotPos = [(self.arenaWidth / 5, self.arenaLength - 3.5),
                             (self.arenaWidth / 5 * 2, self.arenaLength - 3.5),
                             (self.arenaWidth / 5 * 3, self.arenaLength - 3.5),
                             (self.arenaWidth / 5 * 4, self.arenaLength - 3.5)]
        level01_robotPos = [(self.arenaWidth / 5, self.arenaLength - 4),
                            (self.arenaWidth / 5 * 2, self.arenaLength - 2.5),
                            (self.arenaWidth / 5 * 3, self.arenaLength - 2.5),
                            (self.arenaWidth / 5 * 4, self.arenaLength - 4)]
        level02_robotPos = [(1.5, 1.5),
                            (self.arenaWidth -1.5, 1.5),
                            (self.arenaWidth -1.5, self.arenaLength -1.5),
                            (1.5,  self.arenaLength -1.5)]
        level03_robotPos = [(self.arenaWidth / 5 *2, 1),
                            (self.arenaWidth / 5   , 1),
                            (self.arenaWidth / 5 *4, self.arenaLength -1),
                            (self.arenaWidth / 5 *3,  self.arenaLength -1)]
        level04_robotPos = level01_robotPos
        level05_robotPos = level02_robotPos
        level06_robotPos = level03_robotPos
        level07_robotPos = [(self.arenaWidth /2, self.arenaLength /2 - 3.5),
                            (self.arenaWidth /2 +3.5, self.arenaLength /2),
                            (self.arenaWidth /2, self.arenaLength / 2 + 3.5),
                            (self.arenaWidth /2 -3.5, self.arenaLength / 2)]

        level00_robotOrient = [math.pi/2*3 for _ in range(4)]
        level01_robotOrient = [math.pi/4 *7, math.pi/2 *3, math.pi/2 *3, math.pi/4 *5]
        level02_robotOrient = [math.pi/4 , math.pi/4 *3, math.pi/4 *5, math.pi/4 *7]
        level03_robotOrient = [math.pi/4 , math.pi/4 *3, math.pi/4 *5, math.pi/4 *7]
        level04_robotOrient = level01_robotOrient
        level05_robotOrient = level02_robotOrient
        level06_robotOrient = level03_robotOrient
        level07_robotOrient = [math.pi/2 , math.pi, math.pi/2 *3, 0]


        level00_stationsPos = [(self.arenaWidth / 5, 3.5),
                                (self.arenaWidth / 5 * 2, 3.5),
                                (self.arenaWidth / 5 * 3, 3.5),
                                (self.arenaWidth / 5 * 4, 3.5)]
        level01_stationsPos = [(self.arenaWidth / 2 - 3, 2.2),
                               (self.arenaWidth / 2 -1.15, 3.1),
                               (self.arenaWidth / 2 +1.15, 3.1),
                               (self.arenaWidth / 2 + 3, 2.2)]
        level02_stationsPos = [(self.arenaWidth / 2 -1, self.arenaLength / 2 -1),
                               (self.arenaWidth / 2 +1, self.arenaLength / 2 -1),
                               (self.arenaWidth / 2 +1, self.arenaLength / 2 +1),
                               (self.arenaWidth / 2 -1, self.arenaLength / 2 +1)]
        level03_stationsPos = [(self.arenaWidth / 5   , self.arenaLength -1),
                               (self.arenaWidth / 5 *2, self.arenaLength -1),
                               (self.arenaWidth / 5 *3, 1),
                               (self.arenaWidth / 5 *4, 1)]
        level04_stationsPos = [level01_stationsPos[2], level01_stationsPos[3], level01_stationsPos[0], level01_stationsPos[1]]
        level05_stationsPos = [level02_stationsPos[2], level02_stationsPos[3], level02_stationsPos[0], level02_stationsPos[1]]
        level06_stationsPos = level03_stationsPos
        level07_stationsPos = [(self.arenaWidth /2, self.arenaLength / 2 + 5),
                               (self.arenaWidth /2 -5, self.arenaLength / 2),
                               (self.arenaWidth /2, self.arenaLength /2 - 5),
                               (self.arenaWidth /2 +5, self.arenaLength /2)]

        noWalls = self.walls
        level06_walls = self.walls + [CollidorLine(0,self.arenaLength/3, self.arenaWidth/24 *6, self.arenaLength/3),
                                      CollidorLine(self.arenaWidth/24 *6, self.arenaLength/3,self.arenaWidth/24 *6, self.arenaLength/3*2),
                                      CollidorLine(self.arenaWidth/24 *9, self.arenaLength/3, self.arenaWidth/24 *9, self.arenaLength/3 *2),
                                      CollidorLine(self.arenaWidth/24 *9, self.arenaLength/3, self.arenaWidth/24 *12, self.arenaLength/3),
                                      CollidorLine(self.arenaWidth/2, 0, self.arenaWidth/2, self.arenaLength),
                                      CollidorLine(self.arenaWidth/24 *12, self.arenaLength/3*2, self.arenaWidth/24 *15, self.arenaLength/3 *2),
                                      CollidorLine(self.arenaWidth/24 *15, self.arenaLength/3, self.arenaWidth/24 *15, self.arenaLength/3 *2),
                                      CollidorLine(self.arenaWidth/24 *18,self.arenaLength/3,self.arenaWidth/24 *18, self.arenaLength/3*2),
                                      CollidorLine(self.arenaWidth/24 * 18,self.arenaLength/3 * 2,self.arenaWidth, self.arenaLength/3 * 2)]
        centerW = self.arenaWidth/2
        centerL = self.arenaLength/2
        level07_walls = self.walls+[CollidorLine(0, centerL-1, centerW-1, centerL-1),
                                    CollidorLine(0, centerL+1, centerW-1, centerL+1),
                                    CollidorLine(centerW+1, centerL - 1, self.arenaWidth, centerL - 1),
                                    CollidorLine(centerW+1, centerL + 1, self.arenaWidth, centerL + 1),
                                    CollidorLine(centerW-1, 0 , centerW-1, centerL-1),
                                    CollidorLine(centerW+1, 0 , centerW+1, centerL-1),
                                    CollidorLine(centerW-1, centerL+1 , centerW-1, self.arenaLength),
                                    CollidorLine(centerW+1, centerL+1 , centerW+1, self.arenaLength)]

        self.noiseStrength = [0.1, 0.2, 0.33, 0.5, 0.66, 0.2, 0.5, 0.15]
        self.level = [(level00_robotPos,level00_robotOrient,level00_stationsPos, noWalls),
                      (level01_robotPos,level01_robotOrient,level01_stationsPos, noWalls),
                      (level02_robotPos,level02_robotOrient,level02_stationsPos, noWalls),
                      (level03_robotPos,level03_robotOrient,level03_stationsPos, noWalls),
                      (level04_robotPos,level04_robotOrient,level04_stationsPos, noWalls),
                      (level05_robotPos,level05_robotOrient,level05_stationsPos, noWalls),
                      (level06_robotPos,level06_robotOrient,level06_stationsPos, level06_walls),
                      (level07_robotPos,level07_robotOrient,level07_stationsPos, level07_walls)]
        self.simulationWindow = None

        self.reset(0)

        if self.hasUI:
            self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, args, self.walls)
            self.simulationWindow.show()



        self.simTime = 0  # s
        self.simTimestep = 0.1  # s



        # self.plotterWindow = PlotterWindow(app)

    def reset(self, level):
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
                r.reset(self.stations, self.level[level][0][i], self.level[level][1][i]+(random.uniform(0, math.pi)*self.noiseStrength[level]), self.level[level][3])

        for robot in self.robots:
            robot.resetSonar(self.robots)
        if self.hasUI and self.simulationWindow != None:
            self.simulationWindow.setWalls(self.level[level][3])


    def isFarEnoughApart(self, stationPositions, randPos, minDist):

        for pos in stationPositions:
            dist = math.sqrt((pos[0]-randPos[0])**2+(pos[1]-randPos[1])**2)
            if(dist<minDist):
                return False

        return True

    def getRobot(self):
        return self.robot

    def getPickUpStation(self):
        return self.pickUp

    def getDeliveryStation(self):
        return self.delivery

    def getGoalWidth(self):
        goalWidth = self.pickUp.getWidth()
        return goalWidth

    def getGoalLength(self):
        goalLength = self.pickUp.getLength()
        return goalLength

    def update(self, robotsTarVels, stepsLeft):
        """
        updates the robots and checks the exit conditions of the current epoch
        :param tarLinVel: int/ float -
            target linear velocity
        :param tarAngVel: int/ float -
            target angular velocity
        :return: tuple -
            (Boolean - out of area, Boolean - reached pickup, Boolean - reached Delivery)
        """
        # self.plotterWindow.plot(self.robot.getLinearVelocity(), self.simTime)
        # self.plotterWindow.plot(self.robot.getAngularVelocity(), self.simTime)
        # time.sleep(0.1)
        self.simTime += self.simTimestep

        for i, robot in enumerate(self.robots): #TODO überall die action Liste iterieren nicht die robeoter
            if robot.isActive() == True:
                tarLinVel, tarAngVel = robotsTarVels[i]
                self.robots[i].update(self.simTimestep, tarLinVel, tarAngVel)

        if self.args.mode == 'sonar':
            for i, robot in enumerate(self.robots):
                if robotsTarVels[i] != (None, None):
                    robot.sonarReading(self.robots, stepsLeft, self.steps)
        saveWeights = False
        if self.hasUI:
            if self.simulationWindow != 0:
                for i, robot in enumerate(self.robots):
                    self.simulationWindow.updateRobot(robot, i, self.steps-stepsLeft)



        robotsTerminations = []
        for robot in self.robots:
            if robot.isActive():
                collision = False
                reachedPickUp = False
                runOutOfTime = False

                if stepsLeft <= 0:
                    runOutOfTime = True


                # # nicht rechts oder links aus dem Fenster gehen
                # if (robot.getPosX() + robot.width) > self.arenaWidth \
                #         or (robot.getPosX()) < 0:
                #     collision = True
                #
                # # nicht oben oder unten aus dem Fenster gehen
                # if (robot.getPosY() + robot.length) > self.arenaLength or \
                #         (robot.getPosY()) < 0:
                #     collision = True

                # for dist in robot.distances:
                #     if dist<robot.getRadius()+0.05:
                #         collision = True
                if(np.min(robot.distances)<=robot.getRadius()+0.05):
                    collision = True
                # if any(d <= robot.getRadius()+0.05 for d in robot.distances):
                #     collision = True

                # Wenn der Roboter mit der PickUpStation kollidiert und sie als Ziel hat wird ein neues Ziel generiert
                # und reachedPickUp auf True gesetzt, ansonsten bleibt das alte Ziel
                #if robot.hasGoal(self.pickUp):
                if robot.collideWithTargetStationCircular():
                    reachedPickUp = True
                #     goal = (self.delivery.getPosX(), self.delivery.getPosY())
                # else:
                #     goal = (self.pickUp.getPosX(), self.pickUp.getPosY())
                # else:
                #     goal = (self.delivery.getPosX(), self.delivery.getPosY())
                #     if robot.collideWithStation(self.delivery):
                #         reachedDelivery = True

                if collision or reachedPickUp or runOutOfTime:
                    robot.deactivate()
                robotsTerminations.append((collision, reachedPickUp, runOutOfTime))
            else:
                robotsTerminations.append((None, None, None))

        return robotsTerminations

