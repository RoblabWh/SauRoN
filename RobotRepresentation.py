from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer
import math

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