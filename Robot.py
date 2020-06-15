import math


# import keyboard


class Robot:

    def __init__(self, position, startDirection, station):
        self.startposX, self.startposY = position
        self.startDirection = startDirection
        self.goalX, self.goalY = station.getPosX(), station.getPosY()
        self.state = []
        #[posX, posY, direction, linearVelocity, angularVelocity, targetLinearVelocity, targetAngularVelocity, goalX, goalY]

        self.time_steps = 4
        # Robot Hardware Params
        self.width = 0.5  # m
        self.length = 0.5  # m

        self.maxLinearVelocity = 10  # m/s
        self.minLinearVelocity = 0  # m/s
        self.maxLinearAcceleration = 1  # m/s^2
        self.minLinearAcceleration = -5  # m/s^2
        self.maxAngularVelocity = 4  # rad/s
        self.minAngularVelocity = -4  # rad/s
        self.maxAngularAcceleration = 2  # rad/s^2
        self.minAngularAcceleration = -2  # rad/s^2

        self.reset()

    def reset(self):
        posX = self.startposX
        posY = self.startposY
        direction = self.startDirection
        linearVelocity = 0
        angularVelocity = 0
        targetLinearVelocity = 0
        targetAngularVelocity = 0
        goalX = self.goalX
        goalY = self.goalY

        frame = [posX, posY, direction, linearVelocity, angularVelocity, targetLinearVelocity,
                 targetAngularVelocity, goalX, goalY]

        for i in range(self.time_steps):
            self.push_frame(frame)

    #TESTEN!
    def push_frame(self, frame):
        if len(self.state) >= self.time_steps:
            self.state.pop(0)
            self.state.append(frame)
        else:
            self.state.append(frame)

    def update(self, dt, vel, goal):
        goalX, goalY = goal
        tarLinVel, tarAngVel = vel
        linVel, angVel = self.compute_next_velocity(dt, self.getLinearVelocity(), self.getAngularVelocity(), tarLinVel, tarAngVel)
        posX, posY = self.getPosX(), self.getPosY()
        direction = self.getDirection()
        posX += math.cos(self.getDirection()) * linVel * dt
        posY += math.sin(self.getDirection()) * linVel * dt
        direction += angVel * dt

        frame = [posX, posY, direction, linVel, angVel, tarLinVel, tarAngVel, goalX, goalY]
        self.push_frame(frame)

    def compute_next_velocity(self, dt, linVel, angVel, tarLinVel, tarAngVel):
        # beschleunigen
        if linVel < tarLinVel:
            if linVel > self.maxLinearVelocity:
                linVel = self.maxLinearVelocity
            else:
                linVel += self.maxLinearAcceleration * dt  # v(t) = v(t-1) + a * dt

        # bremsen
        elif linVel > tarLinVel:
            if linVel < self.minLinearVelocity:
                linVel = self.minLinearVelocity
            else:
                linVel += self.minLinearAcceleration * dt

        # nach links drehen
        if angVel < tarAngVel:
            if angVel > self.maxAngularVelocity:
                angVel = self.maxAngularVelocity
            else:
                angVel += self.maxAngularAcceleration * dt

        # nach rechts drehen
        elif angVel > tarAngVel:
            if angVel < self.minAngularVelocity:
                angVel = self.minAngularVelocity
            else:
                angVel += self.minAngularAcceleration * dt

        return linVel, angVel

    def collideWithStation(self, station):
        if self.getPosX() <= station.getPosX() + station.getWidth() and \
                self.getPosX() + self.width >= station.getPosX() and \
                self.getPosY() + self.length >= station.getPosY() and \
                self.getPosY() <= station.getPosY() + station.getLength():
            return True
        return False

    def hasGoal(self, station):
        if self.getGoalX() == station.getPosX() and self.getGoalY() == station.getPosY():
            return True
        return False

    def getPosX(self):
        return self.state[3][0]

    def getPosY(self):
        return self.state[3][1]

    def getDirection(self):
        return self.state[3][2]

    def getLinearVelocity(self):
        return self.state[3][3]

    def getGoalX(self):
        return self.state[3][7]

    def getGoalY(self):
        return self.state[3][8]

    def getVelocity(self):
        return self.state[3][3], self.state[3][4]

    def getLinearVelocity(self):
        return self.state[3][3]

    def getAngularVelocity(self):
        return self.state[3][4]

    def getTargetLinearVelocity(self):
        return self.state[3][5]

    def getTargetAngularVelocity(self):
        return self.state[3][6]
