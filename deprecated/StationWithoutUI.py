from Borders import ColliderLine


class Station:
    def __init__(self, posX, posY, width, length, color, scaleFactor):
        self.scaleFactor = scaleFactor
        self.posX = posX
        self.posY = posY

        self.width = width
        self.length = length


        self.borders = [ColliderLine(posX + width, posY, posX, posY),
                        ColliderLine(posX, posY, posX, posY + length),
                        ColliderLine(posX, posY + length, posX + width, posY + length),
                        ColliderLine(posX + width, posY + length, posX + width, posY)]


    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getWidth(self):
        return self.width

    def getLength(self):
        return self.length

    def reposition(self, posX, posY):
        self.posX = posX
        self.posY = posY
