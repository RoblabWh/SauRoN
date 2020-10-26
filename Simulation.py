from PyQt5.QtCore import QTimer
import Robot
from Station import Station
import SimulationWindow
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
        self.pickUp = Station(6, 7.5, 1, 1, 0, self.scaleFactor)
        #self.pickUp = Station(random.randrange(1, 21), random.randrange(1, 9.0), 1, 1, 0, self.scaleFactor)
        self.delivery = Station(1, 1, 0.5, 0.5, 1, self.scaleFactor)
        self.robot = Robot.Robot((10.5, 8.0), 3*math.pi/2, self.pickUp, args, timeframes, self.walls)
        # self.robot2 = Robot.Robot((700.0, 500.0), 3*math.pi/2, self.pickUp, args, timeframes



        # Erstelle Liste aller Stationen und Roboter (Für Multiroboter Multistation Support!) TODO
        self.robots = [self.robot]
        self.stations = [self.pickUp, self.delivery]

        self.simulationWindow = SimulationWindow.SimulationWindow(app, self.robots, self.stations, args)
        self.simulationWindow.show()

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
        # self.plotterWindow.plot(self.robot.getLinearVelocity(), self.simTime)
        # self.plotterWindow.plot(self.robot.getAngularVelocity(), self.simTime)
        # time.sleep(0.1)
        self.simTime += self.simTimestep
        #TODO Parameter ist Liste an Actions für jeden Roboter


        #TODO Abbruchkriterien jedes Roboters prüfen (außer er hat bereits abgebrochen)
        outOfArea = False
        reachedPickUp = False
        reachedDelivery = False


        # nicht rechts oder links aus dem Fenster gehen
        if (self.robot.getPosX() + self.robot.width) > self.arenaWidth \
                or (self.robot.getPosX()) < 0:
            outOfArea = True

        # nicht oben oder unten aus dem Fenster gehen
        if (self.robot.getPosY() + self.robot.length) > self.arenaLength or \
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

        # TODO in Schleife bei mehreren Robotern (außer bei denen die bereits done sind)
        self.robot.update(self.simTimestep, tarLinVel, tarAngVel, goal)
        #TODO eigene Schleife bei mehreren Robotern (erst alle update dann in neuer Schleife das Sonar)
        if self.args.mode == 'sonar':
            self.robot.sonarReading()

        if self.simulationWindow != 0:
            for i, robot in enumerate(self.robots):
                self.simulationWindow.updateRobot(robot, i)
        #TODO das als Liste pro roboter zurückgeben
        return outOfArea, reachedPickUp, reachedDelivery
