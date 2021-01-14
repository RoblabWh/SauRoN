# from PyQt5.QtCore import QTimer
import Robot
from Station import Station
# import SimulationWindow
import math, random
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
        self.pickUp = Station(5, 1.2, 0.75, 0.75, 0, self.scaleFactor)
        self.pickUp2 = Station(1, 1.25, 0.75, 0.75, 3, self.scaleFactor)
        self.pickUp3 = Station(9, 1.1, 0.75, 0.75, 1, self.scaleFactor)
        self.pickUp4 = Station(13, 1.3, 0.75, 0.75, 2, self.scaleFactor)
        self.stations = [self.pickUp, self.pickUp2, self.pickUp3, self.pickUp4]
        # self.stations = [self.pickUp, self.pickUp3]#, self.pickUp4]


        self.robot = Robot.Robot((10.5, 8.8), 3.2*math.pi/2, self.pickUp3, args, timeframes, self.walls, self.stations)
        self.robot2 = Robot.Robot((4, 8.6), 2.6*math.pi/2, self.pickUp, args, timeframes, self.walls, self.stations)
        self.robot3 = Robot.Robot((1.1, 8.9), 3.6*math.pi/2, self.pickUp4, args, timeframes, self.walls, self.stations)
        self.robot4 = Robot.Robot((13.1, 3.9), 3.6*math.pi/2, self.pickUp2, args, timeframes, self.walls, self.stations)


        robots = [self.robot2, self.robot, self.robot3, self.robot4]
        self.robots = []
        for i in range(args.nb_robots):
            self.robots.append(robots[i])

        for robot in self.robots:
            robot.reset()
        for robot in self.robots:
            robot.resetSonar(self.robots)

        # self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, args)
        # self.simulationWindow.show()

        self.simTime = 0  # s
        self.simTimestep = 0.1  # s



        # self.plotterWindow = PlotterWindow(app)

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

        # if self.simulationWindow != 0:
        #     for i, robot in enumerate(self.robots):
        #         self.simulationWindow.updateRobot(robot, i, self.steps-stepsLeft)



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
                if any(d <= robot.getRadius()+0.05 for d in robot.distances):
                    collision = True

                # Wenn der Roboter mit der PickUpStation kollidiert und sie als Ziel hat wird ein neues Ziel generiert
                # und reachedPickUp auf True gesetzt, ansonsten bleibt das alte Ziel
                #if robot.hasGoal(self.pickUp):
                if robot.collideWithTargetStation():
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

