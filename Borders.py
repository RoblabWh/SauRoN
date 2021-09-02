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
            length = math.sqrt(xDif**2 + yDif**2)
            self.n = (-yDif/length, xDif/length)
            self.normalOrigin = (x1 + xDif/2, y1 + yDif/2)
        else:
            self.n = (xn, yn)

    def updatePos(self, pos1, pos2):
        self.a = pos1
        self.b = pos2

        xDif = pos2[0] - pos1[0]
        yDif = pos2[1] - pos1[1]
        length = math.sqrt(xDif ** 2 + yDif ** 2)
        self.n = (-yDif / length, xDif / length)
        self.normalOrigin = (pos1[0] + xDif / 2, pos1[1] + yDif / 2)

    def getN(self):
        return self.n

    def getStart(self):
        return self.a

    def getEnd(self):
        return self.b

    def paint(self, painter, scaleFactor):
        painter.setPen(QPen(Qt.black, 3))
        painter.drawLine(self.a[0] * scaleFactor, self.a[1] * scaleFactor, self.b[0] * scaleFactor, self.b[1] * scaleFactor)
        #Flächen-Normalen der Wände
        # painter.setPen(QPen(Qt.magenta))
        # painter.drawLine(self.normalOrigin[0]*scaleFactor, self.normalOrigin[1]*scaleFactor,
        #                  (self.normalOrigin[0]+ (self.n[0]*0.2)) *scaleFactor, (self.normalOrigin[1]+ (self.n[1]*0.2))*scaleFactor)

import math

class SquareWall:

    def __init__(self, xPos, yPos, width, height, rotation = 0, degree = False):
        self.xPos = xPos
        self.yPos = yPos
        self.halfW = width/2
        self.halfH = height/2

        # (xa,ya)     ya     (xb,ya)
        #     o----------------o
        # xa  |       pos      |  xb
        #     o----------------o
        # (xa,yb)     yb     (xb,yb)

        xa = - self.halfW
        xb = + self.halfW
        ya = - self.halfH
        yb = + self.halfH


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

        self.corners=[(x0,y0), (x1,y1), (x2,y2), (x3,y3)]


        c1 = ColliderLine(x1, y1, x0, y0)
        c2 = ColliderLine(x2, y2, x1, y1)
        c3 = ColliderLine(x3, y3, x2, y2)
        c4 = ColliderLine(x0, y0, x3, y3)
        self.borders = [c1, c2, c3, c4]

    def rotate(self, rot00, rot10, rot01, rot11):
        #-0.5736 0.8191 -0.8191 -0.5736
        # Rot matrix R:
        # [[rot00, rot01]    =   [[cos(r), -sin(r)]
        #  [rot10, rot11]]        [sin(r),  cos(r)]

        xPos = self.xPos
        yPos = self.yPos


        for i, corner in enumerate(self.corners):
            x, y = corner
            x -= xPos
            y -= yPos
            x1 = rot00 * x + rot01 * y + xPos
            y1 = rot10 * x + rot11 * y + yPos
            self.corners[i]= (x1,y1)

        for i, border in enumerate(self.borders):
            border.updatePos(self.corners[(i+1)%4], self.corners[i])



    def getBorders(self):
        return self.borders

class CircleWall:
    def __init__(self, cx, cy, r):
        self.posX = cx
        self.posY = cy
        self.radius = r


    def paint(self, painter, scaleFactor):
        self.scaleFactor = scaleFactor
        painter.setPen(QPen(Qt.black, 3))
        painter.drawEllipse((self.posX-self.radius) * self.scaleFactor, (self.posY-self.radius) * self.scaleFactor, self.radius*2 * self.scaleFactor, self.radius*2 * self.scaleFactor)

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getRadius(self):
        return self.radius