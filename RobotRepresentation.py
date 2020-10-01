from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer
import math


class RobotRepresentation:
    def __init__(self, x, y, direction, width, height, scaleFactor, mode):
        self.mode = mode
        self.scale = scaleFactor
        self.width = width * self.scale
        self.height = height * self.scale

        self.lineColor = Qt.green
        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.white
        self.brushStyle = Qt.SolidPattern
        self.radius = self.width / 2

        self.posX = x * self.scale
        self.posY = y * self.scale
        self.direction = direction
        self.radarHits = []

    def paint(self, painter):
        if self.mode == 'sonar':
            painter.setPen(QPen(Qt.darkMagenta, 1.5, Qt.DotLine))
            painter.setBrush(QBrush(Qt.darkMagenta, self.brushStyle))
            for i in range(0, len(self.radarHits)):
                painter.drawLine(self.posX,
                                 self.posY,
                                 self.radarHits[i][0] * self.scale,
                                 self.radarHits[i][1] * self.scale)
                painter.drawEllipse(self.radarHits[i][0] * self.scale - 3, self.radarHits[i][1] * self.scale - 3, 6, 6)


        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawEllipse(self.posX-self.radius, self.posY-self.radius, self.width, self.height)

        middlex = self.posX + self.radius
        middley = self.posY + self.radius

        painter.drawLine(self.posX,
                         self.posY,
                         self.posX + self.radius * math.cos(self.direction),
                         self.posY + self.radius * math.sin(self.direction))


    def update(self, x, y, direction, radarHits):
        self.posX = x * self.scale
        self.posY = y * self.scale
        self.direction = direction
        self.radarHits = radarHits

