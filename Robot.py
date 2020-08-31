import math


# import keyboard
from pynput.keyboard import Key, Listener


class Robot:

    def __init__(self, position, startDirection, station, args):
        self.startposX, self.startposY = position
        self.startDirection = startDirection
        self.goalX, self.goalY = station.getPosX(), station.getPosY()
        self.state = []
        self.state_raw = []
        # [posX, posY, direction, linearVelocity, angularVelocity, targetLinearVelocity, targetAngularVelocity, goalX, goalY]

        self.time_steps = 8
        # Robot Hardware Params
        self.width = 50  # cm
        self.length = 50  # cm
        self.radius = self.width / 2

        self.maxLinearVelocity = 10  # m/s
        self.minLinearVelocity = 0  # m/s
        self.maxLinearAcceleration = 5  # m/s^2
        self.minLinearAcceleration = -5  # m/s^2
        self.maxAngularVelocity = 1  # rad/s
        self.minAngularVelocity = -1  # rad/s
        self.maxAngularAcceleration = 0.02  # rad/s^2
        self.minAngularAcceleration = -0.02 # rad/s^2

        self.XYnorm = [1000, 600]
        self.directionnom = [0, 2 * math.pi]

        self.manuell = args.manually

        if self.manuell:
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            self.linTast = 0
            self.angTast = 0

        self.reset()

    def reset(self):
        posX = self.startposX
        posY = self.startposY
        direction = self.startDirection
        linVel = 0
        angVel = 0
        targetLinearVelocity = 0
        targetAngularVelocity = 0
        goalX = self.goalX
        goalY = self.goalY

        # frame = [posX, posY, direction, linearVelocity, angularVelocity, targetLinearVelocity,
        #          targetAngularVelocity, goalX, goalY]
        frame = [posX, posY, direction, linVel, angVel, goalX, goalY]

        for _ in range(self.time_steps):
            self.push_frame(frame)

    def denormdata(self, data, limits):
        return (data * (limits[1] - limits[0])) + limits[0]

    def normalize(self, frame):
        posX = frame[0] / self.XYnorm[0]
        posY = frame[1] / self.XYnorm[1]
        direction = (frame[2] - self.directionnom[0]) / (self.directionnom[1] - self.directionnom[0])
        linVel = (frame[3] - self.minLinearVelocity) / (self.maxLinearVelocity - self.minLinearVelocity)
        angVel = (frame[4] - self.minAngularVelocity) / (self.maxAngularVelocity - self.minAngularVelocity)
        goalX = frame[5] / self.XYnorm[0]
        goalY = frame[6] / self.XYnorm[1]

        frame = [posX, posY, direction, linVel, angVel, goalX, goalY]

        return frame

    # TESTEN!
    def push_frame(self, frame):
        frame_norm = self.normalize(frame)
        if len(self.state) >= self.time_steps:
            self.state.pop(0)
            self.state.append(frame_norm)
            self.state_raw.pop(0)
            self.state_raw.append(frame)
        else:
            self.state.append(frame_norm)
            self.state_raw.append(frame)

    def update(self, dt, vel, goal):

        ##### OLD ########
        # direction = self.getDirection()
        # posX += math.cos(self.getDirection()) * linVel * dt
        # posY += math.sin(self.getDirection()) * linVel * dt
        # direction += angVel * dt

        # posX += math.cos(direction) * linVel * dt
        # posY += math.sin(-direction) * linVel * dt
        # direction = self.getDirection() + angVel * dt
        ##################

        posX, posY = self.getPosX(), self.getPosY()
        goalX, goalY = goal
        tarLinVel, tarAngVel = vel

        if not self.manuell:
            linVel, angVel = self.compute_next_velocity(dt, self.getLinearVelocity(), self.getAngularVelocity(),
                                                        tarLinVel, tarAngVel)
        else:
            linVel = self.linTast
            angVel = self.angTast

        direction = (self.getDirection() + (angVel * dt) + 2 * math.pi) % (2 * math.pi)
        posX += math.cos(direction) * linVel * dt
        posY += math.sin(direction) * linVel * dt

        # frame = [posX, posY, direction, linVel, angVel, tarLinVel, tarAngVel, goalX, goalY]
        frame = [posX, posY, direction, linVel, angVel, goalX, goalY]
        self.push_frame(frame)

    def compute_next_velocity(self, dt, linVel, angVel, tarLinVel, tarAngVel):
        # beschleunigen
        if linVel < tarLinVel:
            linVel += self.maxLinearAcceleration * dt  # v(t) = v(t-1) + a * dt
            if linVel > self.maxLinearVelocity:
                linVel = self.maxLinearVelocity

        # bremsen
        elif linVel > tarLinVel:
            linVel += self.minLinearAcceleration * dt
            if linVel < self.minLinearVelocity:
                linVel = self.minLinearVelocity

        # nach links drehen
        if angVel < tarAngVel:
            angVel += self.maxAngularAcceleration * dt
            if angVel > self.maxAngularVelocity:
                angVel = self.maxAngularVelocity

        # nach rechts drehen
        elif angVel > tarAngVel:
            angVel += self.minAngularAcceleration * dt
            if angVel < self.minAngularVelocity:
                angVel = self.minAngularVelocity

        return linVel, angVel

    def collideWithStation(self, station):
        if self.getPosX() <= station.getPosX() + station.getWidth() and \
                self.getPosX() + self.width >= station.getPosX() and \
                self.getPosY() + self.length >= station.getPosY() and \
                self.getPosY() <= station.getPosY() + station.getLength():
            return True
        return False

    def isInCircleOfGoal(self, r):
        return math.sqrt((self.getPosX() - self.getGoalX()) ** 2 +
                  (self.getPosY() - self.getGoalY()) ** 2) < r

    def hasGoal(self, station):
        if self.getGoalX() == station.getPosX() and self.getGoalY() == station.getPosY():
            return True
        return False

    # def getPosX(self):
    #     return self.denormdata(self.state[self.time_steps - 1][0], [0, self.XYnorm[0]])
    #
    # def getPosY(self):
    #     return self.denormdata(self.state[self.time_steps - 1][1], [0, self.XYnorm[1]])
    #
    # def getDirection(self):
    #     return self.denormdata(self.state[self.time_steps - 1][2], self.directionnom)
    #
    # def getLinearVelocity(self):
    #     return self.denormdata(self.state[self.time_steps - 1][3], [self.minLinearVelocity, self.maxLinearVelocity])
    #
    # def getAngularVelocity(self):
    #     return self.denormdata(self.state[self.time_steps - 1][4], [self.minAngularVelocity, self.maxAngularVelocity])
    #
    # def getGoalX(self):
    #     return self.denormdata(self.state[self.time_steps - 1][5], [0, self.XYnorm[0]])
    #
    # def getGoalY(self):
    #     return self.denormdata(self.state[self.time_steps - 1][6], [0, self.XYnorm[1]])

    def getPosX(self):
        return self.state_raw[self.time_steps - 1][0]

    def getPosY(self):
        return self.state_raw[self.time_steps - 1][1]

    def getDirection(self):
        return self.state_raw[self.time_steps - 1][2]

    def getLinearVelocity(self):
        return self.state_raw[self.time_steps - 1][3]

    def getAngularVelocity(self):
        return self.state_raw[self.time_steps - 1][4]

    def getGoalX(self):
        return self.state_raw[self.time_steps - 1][5]

    def getGoalY(self):
        return self.denormdata(self.state[self.time_steps - 1][6], [0, self.XYnorm[1]])

    def getVelocity(self):
        return self.getLinearVelocity(), self.getAngularVelocity()

    def on_press(self, key):

        if key.char == 'w':
            self.linTast = 0.5
        if key.char == 'a':
            self.angTast = -0.02
        if key.char == 's':
            self.linTast = 0
        if key.char == 'd':
            self.angTast = 0.02
        if key.char == 'c':
            self.angTast = 0

    def on_release(self, key):
        self.linTast = 0
        self.angTast = 0


