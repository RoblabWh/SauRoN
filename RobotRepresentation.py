from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets
import math


class RobotRepresentation:
    def __init__(self, x, y, direction, width, height, scaleFactor, mode, colorIndex):
        self.mode = mode
        self.scale = scaleFactor
        self.width = width * self.scale
        self.height = height * self.scale

        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.white
        self.brushStyle = Qt.SolidPattern
        self.radius = self.width / 2

        self.posX = x * self.scale
        self.posY = y * self.scale
        self.dirV = [0,0]
        self.direction = direction
        self.radarHits = []
        self.isActive = True

        self.lineColor = Qt.blue
        if colorIndex == 1:
            self.lineColor = Qt.red
        if colorIndex == 2:
            self.lineColor = Qt.darkGray
        if colorIndex == 3:
            self.lineColor = Qt.green
        if colorIndex == 4:
            self.lineColor = Qt.darkBlue
        if colorIndex == 5:
            self.lineColor = Qt.gray

    def paint(self, painter, sonarShowing):

        # if keyboard.is_pressed('p'):    # show sonar
        #     self.showSonar = True
        # if keyboard.is_pressed('o'):    # don't show sonar
        #     self.showSonar = False
        #
        # if keyboard.is_pressed('z'):
        #     self.showSimulation = True
        # if keyboard.is_pressed('t'):
        #     self.showSimulation = False
        if self.mode == 'sonar' and self.isActive:
            if sonarShowing:
                painter.setPen(QPen(self.lineColor, 1.5, Qt.DotLine))
                painter.setBrush(QBrush(self.lineColor, self.brushStyle))
                for i in range(0, len(self.radarHits)):
                    painter.drawLine(self.posX,
                                     self.posY,
                                     self.radarHits[i][0] * self.scale,
                                     self.radarHits[i][1] * self.scale)
                    painter.drawEllipse(self.radarHits[i][0] * self.scale - 3, self.radarHits[i][1] * self.scale - 3, 6, 6)

        painter.setPen(QPen(self.lineColor, self.thickness, Qt.DotLine))

        painter.drawLine(self.posX,
                         self.posY,
                         self.posX + 1.25 * self.radius * + (
                                     self.dirV[0] * math.cos(self.direction) - self.dirV[1] * math.sin(self.direction)),
                         self.posY + 1.25 * self.radius * + (
                                     self.dirV[0] * math.sin(self.direction) + self.dirV[1] * math.cos(self.direction)))


        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawEllipse(self.posX-self.radius, self.posY-self.radius, self.width, self.height)

        middlex = self.posX + self.radius
        middley = self.posY + self.radius

        painter.drawLine(self.posX,
                         self.posY,
                         self.posX + self.radius * math.cos(self.direction),
                         self.posY + self.radius * math.sin(self.direction))


        painter.setPen(QPen(Qt.red, self.thickness, self.lineStyle))
        painter.drawEllipse(self.posX-1, self.posY-1, 2, 2)




    def update(self, x, y, direction, radarHits, simShowing, isActive, dirV):
        if simShowing:
            self.posX = x * self.scale
            self.posY = y * self.scale
            self.direction = direction
            self.radarHits = radarHits
            self.isActive = isActive
            self.dirV = dirV


