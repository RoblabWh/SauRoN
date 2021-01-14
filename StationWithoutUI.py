from Borders import CollidorLine


class Station:
    def __init__(self, posX, posY, width, length, color, scaleFactor):
        self.scaleFactor = scaleFactor
        self.posX = posX
        self.posY = posY

        self.width = width
        self.length = length


        self.borders = [CollidorLine(posX+width, posY, posX, posY),
                        CollidorLine(posX, posY, posX, posY+length),
                        CollidorLine(posX, posY+length, posX+width, posY+length),
                        CollidorLine(posX+width, posY+length, posX+width, posY)]


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
