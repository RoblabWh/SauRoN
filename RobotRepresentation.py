from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets
import math
import numpy as np


class RobotRepresentation:
    def __init__(self, x, y, direction, width, height, scaleFactor, mode, colorIndex):
        self.mode = mode
        self.scale = scaleFactor
        self.width = width #* self.scale
        self.height = height # * self.scale

        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.white
        self.brushStyle = Qt.SolidPattern
        self.radiusUnscaled = self.width / 2
        self.radius = self.radiusUnscaled

        self.hasPieSlice = True
        self.pieSliceBorders = None
        self.sensorPos = None

        self.posX = x * self.scale
        self.posY = y * self.scale
        self.dirV = [0,0]
        self.direction = direction
        self.radarHits = []
        self.isActive = True
        self.activations = [1 for _ in range(1081)] # TODO 1081 dynamisch einlesen

        brightness = 235 - (int((colorIndex * 39) / 255) * 80)
        self.lineColor = QColor.fromHsv((colorIndex * 39) % 255, 255, brightness)



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

                if self.hasPieSlice and self.sensorPos != None:
                    posX, posY = self.sensorPos
                    posX = posX * self.scale
                    posY = posY * self.scale
                else:
                    posX = self.posX
                    posY = self.posY


                for i in range(0, len(self.radarHits)):
                    alpha = 0 if self.activations[i] < 0.8 else 1
                    #alpha = 0 if self.activations[i] > 0.2 else 1
                    self.lineColor.setAlphaF(alpha)
                    painter.setPen(QPen(self.lineColor, 1.5, Qt.DotLine))
                    painter.setBrush(QBrush(self.lineColor, self.brushStyle))
                    painter.drawLine(posX,
                                     posY,
                                     self.radarHits[i][0] * self.scale,
                                     self.radarHits[i][1] * self.scale)
                    painter.drawEllipse(self.radarHits[i][0] * self.scale - 3, self.radarHits[i][1] * self.scale - 3, 6, 6)




        self.lineColor.setAlphaF(1)
        painter.setPen(QPen(self.lineColor, self.thickness, Qt.DotLine))

        painter.drawLine(self.posX,
                         self.posY,
                         self.posX + 1.25 * self.radius * (
                                     self.dirV[0] * math.cos(self.direction) - self.dirV[1] * math.sin(self.direction)),
                         self.posY + 1.25 * self.radius * (
                                     self.dirV[0] * math.sin(self.direction) + self.dirV[1] * math.cos(self.direction)))


        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawEllipse(self.posX - self.radius , self.posY - self.radius, self.width * self.scale, self.height * self.scale)

        middlex = self.posX + self.radius
        middley = self.posY + self.radius

        painter.drawLine(self.posX,
                         self.posY,
                         self.posX + self.radius * math.cos(self.direction),
                         self.posY + self.radius * math.sin(self.direction))


        painter.setPen(QPen(Qt.red, self.thickness, self.lineStyle))
        painter.drawEllipse(self.posX-1, self.posY-1, 2, 2)

        if self.hasPieSlice and self.pieSliceBorders != None:
            for border in self.pieSliceBorders:
                border.paint(painter, self.scale)




    def update(self, x, y, direction, radarHits, simShowing, isActive, dirV, activations, pieSliceBorders = None, sensorPos = None):
        if simShowing:
            self.posX = x * self.scale
            self.posY = y * self.scale
            self.direction = direction
            self.radarHits = radarHits
            self.isActive = isActive
            self.dirV = dirV
            self.pieSliceBorders = pieSliceBorders
            self.sensorPos = sensorPos

            if activations is not None:
                max = np.max(activations)
                min = np.min(activations)

                actives = (activations + (0-min)) * (1 / (max - min))

                self.activations = [actives[int(i/6)] for i in range(1081)] #activations.shape[0]
            else:
                self.activations = [1 for _ in range(1081)]  # TODO 1081 dynamisch einlesen

    def updateScale(self, scaleFactor):
        self.scale = scaleFactor
        self.radius = self.radiusUnscaled * self.scale

