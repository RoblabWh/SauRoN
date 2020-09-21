from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer
import math


class RobotRepresentation:
    def __init__(self, x, y, direction, width, height, scale):
        self.scale = scale
        self.width = width / self.scale
        self.height = height / self.scale

        self.lineColor = Qt.green
        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.fillColor = Qt.white
        self.brushStyle = Qt.SolidPattern
        self.radius = self.width / 2

        self.posX = x / self.scale
        self.posY = y / self.scale
        self.direction = direction

    def paint(self, painter):
        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawEllipse(self.posX, self.posY, self.width, self.height)

        middlex = self.posX + self.radius
        middley = self.posY + self.radius

        painter.drawLine(middlex,
                         middley,
                         middlex + self.radius * math.cos(self.direction),
                         middley + self.radius * math.sin(self.direction))

    def update(self, x, y, direction):
        self.posX = x / self.scale
        self.posY = y / self.scale
        self.direction = direction
