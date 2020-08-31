from PyQt5.QtCore import QTimer
import Robot
from Station import Station
import SimulationWindow
import math


class Simulation:

    def __init__(self, app):
        # self.plotterWindow = PlotterWindow.PlotterWindow(app)
        self.simulationWindow = SimulationWindow.SimulationWindow(app)
        self.timer = QTimer()  # to create a thread that calls a function at intervals
        # self.timer.timeout.connect(self.update)  # the update function keeps getting called at intervals
        self.timer.setSingleShot(False)

        #  self.simulationWindow = simulationWindow
        self.simulationWindow.show()
        #  self.plotterWindow = plotterWindow

        self.simTime = 0  # s
        self.simTimestep = 1  # s

        self.pickUp = Station(800.0, 100.0, 50, 50, 0)
        self.delivery = Station(100, 100, 50, 50, 1)
        self.robot = Robot.Robot((400.0, 500.0), 3*math.pi/2, self.pickUp)

        # if self.simulationWindow != 0:
        #     self.simulationWindow.initRobot(meterToPixel(self.robot.getPosX()),
        #                                     meterToPixel(self.robot.getPosY()),
        #                                     self.robot.getDirection(),
        #                                     meterToPixel(self.robot.width),
        #                                     meterToPixel(self.robot.length))
        #
        #     self.simulationWindow.initPickUpStation(meterToPixel(self.pickUp.getPosX()),
        #                                             meterToPixel(self.pickUp.getPosY()),
        #                                             meterToPixel(self.pickUp.getWidth()),
        #                                             meterToPixel(self.pickUp.getLength()))
        #
        #     self.simulationWindow.initDeliveryStation(meterToPixel(self.delivery.getPosX()),
        #                                               meterToPixel(self.delivery.getPosY()),
        #                                               meterToPixel(self.delivery.getWidth()),
        #                                               meterToPixel(self.delivery.getLength()))
        #
        # self.timer.start(self.simTimestep * 1000)

        if self.simulationWindow != 0:
            self.simulationWindow.initRobot(self.robot.getPosX(),
                                            self.robot.getPosY(),
                                            self.robot.getDirection(),
                                            self.robot.width,
                                            self.robot.length)

            self.simulationWindow.initPickUpStation(self.pickUp.getPosX(),
                                                    self.pickUp.getPosY(),
                                                    self.pickUp.getWidth(),
                                                    self.pickUp.getLength())

            self.simulationWindow.initDeliveryStation(self.delivery.getPosX(),
                                                      self.delivery.getPosY(),
                                                      self.delivery.getWidth(),
                                                      self.delivery.getLength())

        self.timer.start(self.simTimestep * 1000)

    def getRobot(self):
        return self.robot

    def getPickUpStation(self):
        return self.pickUp

    def getDeliveryStation(self):
        return self.delivery

    def getGoalWidth(self):
        goalWidth = 0
        if self.robot.hasGoal(self.pickUp):
            goalWidth = self.pickUp.getWidth()
        if self.robot.hasGoal(self.delivery):
            goalWidth = self.delivery.getWidth()
        return goalWidth

    def getGoalLength(self):
        goalLength = 0
        if self.robot.hasGoal(self.pickUp):
            goalLength = self.pickUp.getLength()
        if self.robot.hasGoal(self.delivery):
            goalLength = self.delivery.getLength()
        return goalLength

    def update(self, vel):
        self.simTime += self.simTimestep
        outOfArea = False
        reachedPickUp = False
        reachedDelivery = False

        # nicht rechts oder links aus dem Fenster gehen
        if (self.robot.getPosX() + self.robot.width) > self.simulationWindow.width \
                or (self.robot.getPosX()) < 0:
            outOfArea = True

        # nicht oben oder unten aus dem Fenster gehen
        if (self.robot.getPosY() + self.robot.length) > self.simulationWindow.height or \
                (self.robot.getPosY()) < 0:
            outOfArea = True

        # Wenn der Roboter mit der PickUpStation kollidiert und sie als Ziel hat wird ein neues Ziel generiert
        # und reachedPickUp auf True gesetzt, ansonsten bleibt das alte Ziel
        if self.robot.hasGoal(self.pickUp):
            if self.robot.collideWithStation(self.pickUp):
                reachedPickUp = True
                goal = (self.delivery.getPosX(), self.delivery.getPosY())
            else:
                goal = (self.pickUp.getPosX(), self.pickUp.getPosY())
        else:
            goal = (self.delivery.getPosX(), self.delivery.getPosY())
            if self.robot.collideWithStation(self.delivery):
                reachedDelivery = True

        self.robot.update(self.simTimestep, vel, goal)

        if self.simulationWindow != 0:
            self.simulationWindow.updateRobot((self.robot.getPosX()),
                                              (self.robot.getPosY()),
                                              self.robot.getDirection())
        # if self.plotterWindow != 0:
        #     self.plotterWindow.plot(self.robot.getLinearVelocity(), self.simTime)

        return outOfArea, reachedPickUp, reachedDelivery


def meterToPixel(m):
    return 100 * m
