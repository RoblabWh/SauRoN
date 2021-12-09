from PyQt5.QtGui import QBrush, QPen, QColor
from PyQt5.QtCore import Qt
import math
import numpy as np


class RobotRepresentation:
    def __init__(self, x, y, direction, width, height, scaleFactor, mode, colorIndex, args):
        self.args = args
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
        self.lidarHits = []
        self.isActive = True
        self.activations = [1 for _ in range(self.args.number_of_rays)]

        brightness = 235 - (int((colorIndex * 39) / 255) * 80)
        self.lineColor = QColor.fromHsv((colorIndex * 39) % 255, 255, brightness)
        self.lineColorNegAct = QColor.fromHsv((colorIndex * 39) % 255, 255, brightness-100)



    def paint(self, painter, sonarShowing):

        if self.mode == 'sonar' and self.isActive:
            if sonarShowing:

                if self.hasPieSlice and self.sensorPos != None:
                    posX, posY = self.sensorPos
                    posX = posX * self.scale
                    posY = posY * self.scale
                else:
                    posX = self.posX
                    posY = self.posY

                beta = 0.2 # determines which percentage of high and low activations are shown
                for i in range(0, len(self.lidarHits)):
                    self.lineColor.setAlphaF(1)
                    color = self.lineColor
                    if self.activations != None:
                        alphaPos = False if self.activations[i] < 1-beta else True
                        alphaNeg = False #if self.activations[i] > beta else True

                        if alphaNeg:
                            color = self.lineColorNegAct
                        elif not alphaPos:
                            self.lineColor.setAlphaF(0)
                            color = self.lineColor

                    #color = QColor.fromHsv(36, 255, int(i*(255/len(self.lidarHits))))
                    painter.setPen(QPen(color, 1.5, Qt.DotLine))
                    painter.setBrush(QBrush(color, self.brushStyle))

                    painter.drawLine(posX,
                                     posY,
                                     self.lidarHits[i][0] * self.scale,
                                     self.lidarHits[i][1] * self.scale)
                    painter.drawEllipse(self.lidarHits[i][0] * self.scale - 3, self.lidarHits[i][1] * self.scale - 3, 6, 6)




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




    def update(self, x, y, direction, lidarHits, simShowing, isActive, dirV, activations, pieSliceBorders = None, sensorPos = None):
        if simShowing:
            self.posX = x * self.scale
            self.posY = y * self.scale
            self.direction = direction
            self.lidarHits = lidarHits
            self.isActive = isActive
            self.dirV = dirV
            self.pieSliceBorders = pieSliceBorders
            self.sensorPos = sensorPos

            if activations is not None:
                max = np.max(activations)
                min = np.min(activations)

                actives = (activations + (0-min)) * (1 / (max - min))

                self.activations = [actives[int(i/6)] for i in range(self.args.number_of_rays)] #activations.shape[0]
            else:
                self.activations = None

    def updateScale(self, scaleFactor):
        self.scale = scaleFactor
        self.radius = self.radiusUnscaled * self.scale


