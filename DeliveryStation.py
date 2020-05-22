from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer

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
