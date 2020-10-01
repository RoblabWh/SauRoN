class CollidorLine:
    def __init__(self,x1, y1, x2, y2, xn = 0, yn = 0):
        self.a = (x1, y1)
        self.b = (x2, y2)
        if xn == 0 and yn == 0:
            xDif = x2-x1
            yDif = y2-y1
            self.n = (-yDif, xDif)
        else:
            self.n = (xn, yn)

    def getN(self):
        return self.n