from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt
from Borders import CollidorLine


class Station:
    def __init__(self, posX, posY, radius, color, scaleFactor):
        self.scaleFactor = scaleFactor
        self.posX = posX
        self.posY = posY

        self.radius = radius

        self.color = Qt.blue
        if color == 1:
            self.color = Qt.red
        if color == 2:
            self.color = Qt.darkGray
        if color == 3:
            self.color = Qt.green
        if color == 4:
            self.color = Qt.darkBlue
        if color == 5:
            self.color = Qt.gray


        self.lineColor = self.color  # Qt.red Qt.blue
        self.fillColor = self.color  # Qt.red Qt.blue

        self.thickness = 2
        self.lineStyle = Qt.SolidLine
        self.brushStyle = Qt.SolidPattern



    # def __init__(self, posX, posY, width, length, color, scaleFactor):
    #     self.scaleFactor = scaleFactor
    #     self.posX = posX
    #     self.posY = posY
    #
    #     self.width = width
    #     self.length = length
    #
    #     self.color = Qt.blue
    #     if color == 1:
    #         self.color = Qt.red
    #     if color == 2:
    #         self.color = Qt.yellow
    #     if color == 3:
    #         self.color = Qt.green
    #     if color == 4:
    #         self.color = Qt.darkBlue
    #     if color == 5:
    #         self.color = Qt.gray
    #
    #
    #     self.lineColor = self.color  # Qt.red Qt.blue
    #     self.fillColor = self.color  # Qt.red Qt.blue
    #
    #     self.thickness = 2
    #     self.lineStyle = Qt.SolidLine
    #     self.brushStyle = Qt.SolidPattern
    #
    #     self.borders = [CollidorLine(posX+width, posY, posX, posY),
    #                     CollidorLine(posX, posY, posX, posY+length),
    #                     CollidorLine(posX, posY+length, posX+width, posY+length),
    #                     CollidorLine(posX+width, posY+length, posX+width, posY)]


    def paint(self, painter):
        painter.setPen(QPen(self.lineColor, self.thickness, self.lineStyle))
        painter.setBrush(QBrush(self.fillColor, self.brushStyle))
        painter.drawEllipse((self.posX-self.radius) * self.scaleFactor, (self.posY-self.radius) * self.scaleFactor, self.radius*2 * self.scaleFactor, self.radius*2 * self.scaleFactor)
        # painter.drawRect(self.posX * self.scaleFactor, self.posY * self.scaleFactor, self.width * self.scaleFactor, self.length * self.scaleFactor)

    def setPos(self, pos):
        self.posX, self.posY = pos

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getWidth(self):
        return self.width

    def getLength(self):
        return self.length

    def getRadius(self):
        return self.radius

    def reposition(self, posX, posY):
        self.posX = posX
        self.posY = posY
