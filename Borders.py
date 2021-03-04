from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt

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

    def getStart(self):
        return self.a

    def getEnd(self):
        return self.b

    def paint(self, painter, scaleFactor):
        painter.setPen(QPen(Qt.black))
        painter.drawLine(self.a[0] * scaleFactor, self.a[1] * scaleFactor, self.b[0] * scaleFactor, self.b[1] * scaleFactor)