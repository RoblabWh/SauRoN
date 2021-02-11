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
        self.pickUp2 = Station(1.15, 1.25, 1, 3, self.scaleFactor)
        self.pickUp3 = Station(9, 0.8, 1, 1, self.scaleFactor) #12, 4.1 gute ergebnisse mit trainierte Netz vom 19Jan  #2, 5.1 geht für das neuste
        self.pickUp4 = Station(13, 1.3, 1, 2, self.scaleFactor)
        self.stations = [self.pickUp, self.pickUp2, self.pickUp3, self.pickUp4]
        # self.stations = [self.pickUp, self.pickUp3]#, self.pickUp4]


        self.robot = Robot.Robot((10.5, 8.8), 3.2*math.pi/2, self.pickUp3, args, timeframes, self.walls, self.stations)
        self.robot2 = Robot.Robot((4, 8.6), 2.6*math.pi/2, self.pickUp, args, timeframes, self.walls, self.stations)
        self.robot3 = Robot.Robot((1.1, 8.9), 3.6*math.pi/2, self.pickUp4, args, timeframes, self.walls, self.stations)
        self.robot4 = Robot.Robot((13.1, 3.9), 3.6*math.pi/2, self.pickUp2, args, timeframes, self.walls, self.stations)

        self.robots = [self.robot2, self.robot, self.robot3, self.robot4]
        # self.robots = [self.robot2]#, self.robot]#, self.robot3]

        for robot in self.robots:
            robot.reset(self.stations)
        for robot in self.robots:
            robot.resetSonar(self.robots)

        if self.hasUI:
            self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, args)
            self.simulationWindow.show()

        self.simTime = 0  # s
        self.simTimestep = 0.1  # s



        # self.plotterWindow = PlotterWindow(app)

    def reset(self):
        randomSim = True
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


        for robot in self.robots:
            robot.resetSonar(self.robots)


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

