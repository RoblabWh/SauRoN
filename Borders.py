from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt
import numpy as np

class ColliderLine:
    def __init__(self,x1, y1, x2, y2, xn = 0, yn = 0):
        self.a = (x1, y1)
        self.b = (x2, y2)
        if xn == 0 and yn == 0:
            xDif = x2-x1
            yDif = y2-y1
            self.n = (-yDif, xDif)
        else:
            self.n = (xn, yn)

    def updatePos(self, pos1, pos2):
        self.a = pos1
        self.b = pos2


    def getN(self):
        return self.n

    def getStart(self):
        return self.a

    def getEnd(self):
        return self.b

    def paint(self, painter, scaleFactor):
        painter.setPen(QPen(Qt.black, 3))
        painter.drawLine(self.a[0] * scaleFactor, self.a[1] * scaleFactor, self.b[0] * scaleFactor, self.b[1] * scaleFactor)

import math

class SquareWall:

    def __init__(self, xPos, yPos, width, height, rotation, degree = False):
        halfW = width/2
        halfH = height/2

        # (xa,ya)     ya     (xb,ya)
        #     o----------------o
        # xa  |       pos      |  xb
        #     o----------------o
        # (xa,yb)     yb     (xb,yb)

        xa = - halfW
        xb = + halfW
        ya = - halfH
        yb = + halfH


        if(degree):
            rotation = rotation * (math.pi/180)

        #Rot matrix R:
        # [[rot00, rot01]    =   [[cos(r), -sin(r)]
        #  [rot10, rot11]]        [sin(r),  cos(r)]
        rot00 = math.cos(rotation)
        rot01 = -1* math.sin(rotation)
        rot10 = math.sin(rotation)
        rot11 = rot00

        #Rotation p' = R * p
        x0 = rot00 * xa + rot01 *ya +xPos
        y0 = rot10 * xa + rot11 *ya +yPos

        x1 = rot00 * xb + rot01 * ya +xPos
        y1 = rot10 * xb + rot11 * ya +yPos

        x2 = rot00 * xb + rot01 * yb +xPos
        y2 = rot10 * xb + rot11 * yb +yPos

        x3 = rot00 * xa + rot01 * yb +xPos
        y3 = rot10 * xa + rot11 * yb +yPos


        c1 = ColliderLine(x0, y0, x1, y1)
        c2 = ColliderLine(x1, y1, x2, y2)
        c3 = ColliderLine(x2, y2, x3, y3)
        c4 = ColliderLine(x3, y3, x0, y0)
        self.borders = [c1, c2, c3, c4]


    def getBorders(self):
        return self.borders