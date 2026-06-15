from Visualization.RerunVisualizer import stationColorRGB

class Station:
    def __init__(self, posX, posY, radius, color):
        self.posX = posX
        self.posY = posY

        self.radius = radius

        self.color = stationColorRGB(color)

    def setPos(self, pos):
        self.posX, self.posY = pos

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getWidth(self):
        """
        only for rectangular Stations
        :return: float station width in meter
        """
        return self.width

    def getLength(self):
        """
        only for rectangular Stations
        :return: float station length in meter
        """
        return self.length

    def getRadius(self):
        return self.radius

    def setColor(self, i):
        self.color = stationColorRGB(i)
