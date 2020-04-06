import sys
import time
import random
import math
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
from collections import deque
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer
import keyboard


class PlotterWindow:

    def __init__(self, application):
        self.app = application
        self.win = pg.GraphicsWindow()

        self.datY = deque()
        self.datX = deque()
        self.maxLen = 1000 # max number of data points to show on graph

        self.p1 = self.win.addPlot(colspan=2)
        self.win.nextRow()

        self.curve1 = self.p1.plot()
        self.p1.setYRange(0, 10)
        self.p1.setLabel(axis='left', text='Geschwindigkeit in m/s')
        self.p1.setLabel(axis='bottom', text='Simulation Time in s')


    def plot(self, valueToPlotY, valueToPlotX):
        if len(self.datY) > self.maxLen:
            self.datY.popleft()  # remove oldest
        if len(self.datX) > self.maxLen:
            self.datX.popleft()
        self.datY.append(valueToPlotY)
        self.datX.append(valueToPlotX)
        self.curve1.setData(self.datX, self.datY)
        self.app.processEvents()


class SimulationWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simulation")
        self.width = 1000
        self.height = 600
        self.setGeometry(200, 100, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

        self.robotRepresentation = 0
        self.pickUpStation = 0
        self.deliveryStation = 0

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.begin(self)
        if self.pickUpStation != 0:
            self.pickUpStation.paint(painter)
        if self.deliveryStation != 0:
            self.deliveryStation.paint(painter)
        if self.robotRepresentation != 0:
            self.robotRepresentation.paint(painter)
        painter.end()

    def initRobot(self, robotStartPosX, robotStartPosY, robotStartDirection, robotWidth, robotLength):
        self.robotRepresentation = RobotRepresentation(robotStartPosX,
                                                       robotStartPosY,
                                                       robotStartDirection,
                                                       robotWidth,
                                                       robotLength)

    def initPickUpStation(self, posX, posY, width, length):
        self.pickUpStation = PickUpStation(posX, posY, width, length)

    def initDeliveryStation(self, posX, posY, width, length):
        self.deliveryStation = DeliveryStation(posX, posY, width, length)

    def updateRobot(self, posX, posY, direction):
        self.robotRepresentation.update(posX, posY, direction)
        self.repaint()



def meterToPixel(m):
    return 100 * m


class RobotRepresentation:

    def __init__(self, x, y, direction, width, height):
        self.width = width
        self.height = height

        self.lineColor = Qt.green
        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.white
        self.brushStyle = Qt.SolidPattern

        self.posX = x
        self.posY = y
        self.direction = direction

    def paint(self, painter):
        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawEllipse(self.posX, self.posY, self.width, self.height)
        painter.drawLine(self.posX + self.width / 2,
                         self.posY + self.height / 2,
                         self.posX + self.width / 2 + (self.width / 2) * math.cos(self.direction),
                         self.posY + self.height / 2 - (self.height / 2) * math.sin(- self.direction))

    def update(self, x, y, direction):
        self.posX = x
        self.posY = y
        self.direction = direction



class Simulation:

    def __init__(self, simulationWindow, plotterWindow):
        self.timer = QTimer()  # to create a thread that calls a function at intervals
        self.timer.timeout.connect(self.update)  # the update function keeps getting called at intervals
        self.timer.setSingleShot(False)

        self.simulationWindow = simulationWindow
        self.plotterWindow = plotterWindow

        self.simTime = 0         # s
        self.simTimestep = 0.01  # s

        self.robot = Robot(5.0, 5.0, 0.0)
        self.pickUp = PickUpStation(8.0, 1.0, 0.5, 0.5)
        self.delivery = DeliveryStation(1, 1, 0.5, 0.5)

        if self.simulationWindow != 0:
            self.simulationWindow.initRobot(meterToPixel(self.robot.getPosX()),
                                            meterToPixel(self.robot.getPosY()),
                                            self.robot.getDirection(),
                                            meterToPixel(self.robot.getWidth()),
                                            meterToPixel(self.robot.getLength()))

            self.simulationWindow.initPickUpStation(meterToPixel(self.pickUp.getPosX()),
                                                    meterToPixel(self.pickUp.getPosY()),
                                                    meterToPixel(self.pickUp.getWidth()),
                                                    meterToPixel(self.pickUp.getLength()))

            self.simulationWindow.initDeliveryStation(meterToPixel(self.delivery.getPosX()),
                                                      meterToPixel(self.delivery.getPosY()),
                                                      meterToPixel(self.delivery.getWidth()),
                                                      meterToPixel(self.delivery.getLength()))

        self.agent = Agent(self.robot)

        self.timer.start(self.simTimestep * 1000)


    def update(self):
        self.simTime += self.simTimestep

        # nicht rechts oder links aus dem Fenster gehen
        if meterToPixel(self.robot.getPosX() + self.robot.width) > self.simulationWindow.width or meterToPixel(self.robot.getPosX()) < 0:
            print("out width, posX: " + str(self.robot.getPosX()))
            self.robot.setPose(5,5)
            self.robot.linearVelocity = 0

        # nicht oben oder unten aus dem Fenster gehen
        if meterToPixel(self.robot.getPosY() + self.robot.length) > self.simulationWindow.height or meterToPixel(self.robot.getPosY()) < 0:
            print("out height, posY: " + str(self.robot.getPosY()))
            self.robot.setPose(5,5)
            self.robot.linearVelocity = 0

        self.agent.update(self.simTimestep)
        self.robot.update(self.simTimestep)

        if self.simulationWindow != 0:
            self.simulationWindow.updateRobot(meterToPixel(self.robot.getPosX()),
                                              meterToPixel(self.robot.getPosY()),
                                              self.robot.getDirection())
        if self.plotterWindow != 0:
            self.plotterWindow.plot(self.robot.getLinearVelocity(), self.simTime)



class Robot:

    def __init__(self, startPosX, startPosY, startDirection):
        self.width  = 0.5   # m
        self.length = 0.5   # m

        self.maxLinearVelocity      = 10   # m/s
        self.minLinearVelocity      =  0   # m/s
        self.maxLinearAcceleration  =  1   # m/s^2
        self.minLinearAcceleration  = -5   # m/s^2
        self.maxAngularVelocity     =  4   # rad/s
        self.minAngularVelocity     = -4   # rad/s
        self.maxAngularAcceleration =  2   # rad/s^2
        self.minAngularAcceleration = -2   # rad/s^2

        self.posX = startPosX
        self.posY = startPosY
        self.direction = startDirection

        self.linearVelocity = 0
        self.angularVelocity = 0

        self.targetLinearVelocity = 0
        self.targetAngularVelocity = 0


    def setTargetVelocity(self, newTargetLinearVelocity, newTargetAngularVelocity):
        self.targetLinearVelocity = newTargetLinearVelocity
        self.targetAngularVelocity = newTargetAngularVelocity


    def update(self, dt):
        self.compute_next_velocity(dt)
        self.posX += math.cos(self.direction) * self.linearVelocity * dt
        self.posY += math.sin(self.direction) * self.linearVelocity * dt
        self.direction += self.angularVelocity * dt

        # Tastatursteuerung des Roboters zu Testzwecken

#        if keyboard.is_pressed('right'):
#            self.posX += 0.05
#        if keyboard.is_pressed('left'):
#            self.posX -= 0.05
#        if keyboard.is_pressed('up'):
#            self.posY -= 0.05
#        if keyboard.is_pressed('down'):
#            self.posY += 0.05


    def compute_next_velocity(self, dt):

        # beschleunigen
        if self.linearVelocity < self.targetLinearVelocity:
            if self.linearVelocity > self.maxLinearVelocity:
                self.linearVelocity = self.maxLinearVelocity
            else:
                self.linearVelocity += self.maxLinearAcceleration * dt  # v(t) = v(t-1) + a * dt

        # bremsen
        elif self.linearVelocity > self.targetLinearVelocity:
            if self.linearVelocity < self.minLinearVelocity:
                self.linearVelocity = self.minLinearVelocity
            else:
                self.linearVelocity += self.minLinearAcceleration * dt

        # nach links drehen
        if self.angularVelocity < self.targetAngularVelocity:
            if self.angularVelocity > self.maxAngularVelocity:
                self.angularVelocity = self.maxAngularVelocity
            else:
                self.angularVelocity += self.maxAngularAcceleration * dt

        # nach rechts drehen
        elif self.angularVelocity > self.targetAngularVelocity:
            if self.angularVelocity < self.minAngularVelocity:
                self.angularVelocity = self.minAngularVelocity
            else:
                self.angularVelocity += self.minAngularAcceleration * dt

    def getPose(self):
        return (self.posX, self.posY, self.direction)

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def setPose(self, PosX, PosY):
        self.posX = PosX
        self.posY = PosY

    def getDirection(self):
        return self.direction

    def getVelocity(self):
        return (self.linearVelocity, self.angularVelocity)

    def getLinearVelocity(self):
        return self.linearVelocity

    def getAngularVelocity(self):
        return self.angularVelocity

    def getWidth(self):
        return self.width

    def getLength(self):
        return self.length


class PickUpStation:
    def __init__(self, posX, posY, width, length):
        self.posX = posX
        self.posY = posY

        self.width = width
        self.length = length

        self.lineColor = Qt.blue
        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.blue
        self.brushStyle = Qt.SolidPattern

    def paint(self, painter):
        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawRect(self.posX, self.posY, self.width, self.length)

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getWidth(self):
        return self.width

    def getLength(self):
        return self.length


class DeliveryStation:
    def __init__(self, posX, posY, width, length):
        self.posX = posX
        self.posY = posY

        self.width = width
        self.length = length

        self.lineColor = Qt.red
        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.red
        self.brushStyle = Qt.SolidPattern

    def paint(self, painter):
        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawRect(self.posX, self.posY, self.width, self.length)

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getWidth(self):
        return self.width

    def getLength(self):
        return self.length


class Agent:

    def __init__(self, robot):
        self.robot = robot
        self.targetLinearVelocity = 5
        self.targetAngularVelocity = 5
        self.counter = 0   # s

    def update(self, dt):
        self.counter += dt
        if self.counter > 1:
            self.counter = 0
            self.robot.setTargetVelocity(random.randint(0, 10),
                                         random.randint(-3, 3))


# Workaround for not getting error message
def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


def main():
    sys.excepthook = except_hook
    app = QApplication(sys.argv)

    plotterWindow = PlotterWindow(app)
    #plotterWindow = 0
    simulationWindow = SimulationWindow()
    #simulationWindow = 0

    simulation = Simulation(simulationWindow, plotterWindow)

    simulationWindow.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()