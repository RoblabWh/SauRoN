from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt


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
