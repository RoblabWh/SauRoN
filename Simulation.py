from PyQt5.QtCore import QTimer
import Robot
from Station import Station
import SimulationWindow
import math


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
        # Erstelle Stationen und Roboter
        self.pickUp = Station(800.0, 100.0, 50, 50, 0)
        self.delivery = Station(100, 100, 50, 50, 1)
        self.robot = Robot.Robot((400.0, 500.0), 3*math.pi/2, self.pickUp, args, timeframes)
        # self.robot2 = Robot.Robot((700.0, 500.0), 3*math.pi/2, self.pickUp, args, timeframes)

        # Erstelle Liste aller Stationen und Roboter (Für Multiroboter Multistation Support!) TODO
        self.robots = [self.robot]
        self.stations = [self.pickUp, self.delivery]

        self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations)
        self.simulationWindow.show()

        self.simTime = 0  # s
        self.simTimestep = 0.25  # s

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

    def update(self, tarLinVel, tarAngVel):
        """
        updates the robots and checks the exit conditions of the current epoch
        :param tarLinVel: int/ float -
            target linear velocity
        :param tarAngVel: int/ float -
            target angular velocity
        :return: tuple -
            (Boolean - out of area, Boolean - reached pickup, Boolean - reached Delivery)
        """

        self.simTime += self.simTimestep
        #TODO hier Schleife für jeden Agenten

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

        self.robot.update(self.simTimestep, tarLinVel, tarAngVel, goal)

        if self.simulationWindow != 0:
            for i, robot in enumerate(self.robots):
                self.simulationWindow.updateRobot(robot, i)
        return outOfArea, reachedPickUp, reachedDelivery
