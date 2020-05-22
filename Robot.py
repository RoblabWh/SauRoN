import math
import keyboard


class Robot:

    def __init__(self, startPosX, startPosY, startDirection):
        self.width  = 0.5   # m
        self.length = 0.5   # m

        self.maxLinearVelocity      = 10   # m/s
        self.minLinearVelocity      =  0   # m/s
        self.maxLinearAcceleration  =  1   # m/s^2
        self.minLinearAcceleration  = -5   # m/s^2
        self.maxAngularVelocity     =  4   # rad/s
        self.minAngularVelocity     = -4   # rad/s
        self.maxAngularAcceleration =  2   # rad/s^2
        self.minAngularAcceleration = -2   # rad/s^2

        self.posX = startPosX
        self.posY = startPosY
        self.direction = startDirection

        self.linearVelocity = 0
        self.angularVelocity = 0

        self.targetLinearVelocity = 0
        self.targetAngularVelocity = 0

        self.goal = 0       # goal mit 0 initialisiert, muss zuerst zur PickUpStation, goal = 1 -> Delivery

    def setGoal(self, newGoal):
        self.goal = newGoal

    def getGoal(self):
        return self.goal

    def setTargetVelocity(self, newTargetLinearVelocity, newTargetAngularVelocity):
        self.targetLinearVelocity = newTargetLinearVelocity
        self.targetAngularVelocity = newTargetAngularVelocity

    def update(self, dt):
        self.compute_next_velocity(dt)
        self.posX += math.cos(self.direction) * self.linearVelocity * dt
        self.posY += math.sin(self.direction) * self.linearVelocity * dt
        self.direction += self.angularVelocity * dt

        # Tastatursteuerung des Roboters zu Testzwecken

#        if keyboard.is_pressed('right'):
#            self.posX += 0.05
#        if keyboard.is_pressed('left'):
#            self.posX -= 0.05
#        if keyboard.is_pressed('up'):
#            self.posY -= 0.05
#        if keyboard.is_pressed('down'):
#            self.posY += 0.05


    def compute_next_velocity(self, dt):

        # beschleunigen
        if self.linearVelocity < self.targetLinearVelocity:
            if self.linearVelocity > self.maxLinearVelocity:
                self.linearVelocity = self.maxLinearVelocity
            else:
                self.linearVelocity += self.maxLinearAcceleration * dt  # v(t) = v(t-1) + a * dt

        # bremsen
        elif self.linearVelocity > self.targetLinearVelocity:
            if self.linearVelocity < self.minLinearVelocity:
                self.linearVelocity = self.minLinearVelocity
            else:
                self.linearVelocity += self.minLinearAcceleration * dt

        # nach links drehen
        if self.angularVelocity < self.targetAngularVelocity:
            if self.angularVelocity > self.maxAngularVelocity:
                self.angularVelocity = self.maxAngularVelocity
            else:
                self.angularVelocity += self.maxAngularAcceleration * dt

        # nach rechts drehen
        elif self.angularVelocity > self.targetAngularVelocity:
            if self.angularVelocity < self.minAngularVelocity:
                self.angularVelocity = self.minAngularVelocity
            else:
                self.angularVelocity += self.minAngularAcceleration * dt

    def getPose(self):
        return self.posX, self.posY, self.direction

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getPosXY(self):
        return [self.posX, self.getPosY()]

    def setPose(self, PosX, PosY):
        self.posX = PosX
        self.posY = PosY

    def getDirection(self):
        return self.direction

    def getVelocity(self):
        return self.linearVelocity, self.angularVelocity

    def getLinearVelocity(self):
        return self.linearVelocity

    def setLinearVelocity(self, value):
        self.linearVelocity = value

    def getAngularVelocity(self):
        return self.angularVelocity

    def getWidth(self):
        return self.width

    def getLength(self):
        return self.length
