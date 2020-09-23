import math


# import keyboard
from pynput.keyboard import Key, Listener


class Robot:

    def __init__(self, position, startDirection, station, args, timeframes):
        """
        :param position: tuple (float,float) -
            defines the robots starting position
        :param startDirection: float -
            defines the robots starting orientation
            one evolution is equal to 2*pi
            a value of 0 causes the robot to look right
            by increasing the value the robots start orientation will change clockwise
        :param station:Station.Station -
            defines the target of the robot
        :param args:
            args defined in main
        :param timeframes: int -
            the amount of frames saved as a history to train the neural net
        """
        self.startposX, self.startposY = position
        self.startDirection = startDirection
        self.goalX, self.goalY = station.getPosX(), station.getPosY()
        self.state = []
        self.state_raw = []
        # [posX, posY, direction, linearVelocity, angularVelocity, goalX, goalY, targetLinearVelocity, targetAngularVelocity]

        self.time_steps = timeframes #4
        # Robot Hardware Params
        self.width = 50  # cm
        self.length = 50  # cm
        self.radius = self.width / 2

        self.maxLinearVelocity = 1000  # cm/s entspricht 10m/s
        self.minLinearVelocity = 0  # m/s
        self.maxLinearAcceleration = 500  # cm/s^2 entspricht 5m/s^2
        self.minLinearAcceleration = -500  # cm/s^2 entspricht 5m/s^2
        self.maxAngularVelocity = 1  # rad/s
        self.minAngularVelocity = -1  # rad/s
        self.maxAngularAcceleration = 0.5  # rad/s^2
        self.minAngularAcceleration = -0.5 # rad/s^2

        self.XYnorm = [args.arena_width, args.arena_length]
        self.directionnom = [0, 2 * math.pi]

        self.manuell = args.manually

        if self.manuell:
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            self.linTast = 0
            self.angTast = 0

        self.reset()

    def reset(self):
        """
        resets the robots position (to his starting position), his orientation (to his starting orientation)
        and sets all velocities back to zero.
        In addition the state gets cleared.
        This method is typically called at the beginning of a training epoch.
        """
        posX = self.startposX
        posY = self.startposY
        direction = self.startDirection
        linVel = 0
        angVel = 0
        tarLinVel = 0
        tarAngVel = 0
        goalX = self.goalX
        goalY = self.goalY

        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]
        frame = [posX, posY, direction, linVel, angVel, goalX, goalY]

        for _ in range(self.time_steps):
            self.push_frame(frame)

    def denormdata(self, data, limits):
        """
        denormalizes a given input (from a range of 0 to 1 back into its original form)

        :param data: float -
            normalized value
        :param limits: list -
            list of two floats representing the min and max limit
        :return: float -
            denormalized data value
        """
        return (data * (limits[1] - limits[0])) + limits[0]

    # Hier nochmal Debuggen. Werte vllt Runden??
    def normalize(self, frame):
        """
        normalizes all values of a frame

        :param frame: list -
            [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]
        :return: list -
            normalized frame with values only between 0 an 1
        """
        posX = frame[0] / self.XYnorm[0]
        posY = frame[1] / self.XYnorm[1]
        direction = (frame[2] - self.directionnom[0]) / (self.directionnom[1] - self.directionnom[0])
        linVel = (frame[3] - self.minLinearVelocity) / (self.maxLinearVelocity - self.minLinearVelocity)
        angVel = (frame[4] - self.minAngularVelocity) / (self.maxAngularVelocity - self.minAngularVelocity)
        goalX = frame[5] / self.XYnorm[0]
        goalY = frame[6] / self.XYnorm[1]
        # tarLinVel = (frame[7] - self.minLinearVelocity) / (self.maxLinearVelocity - self.minLinearVelocity)
        # tarAngVel = (frame[8] - self.minAngularVelocity) / (self.maxAngularVelocity - self.minAngularVelocity)

        frame = [posX, posY, direction, linVel, angVel, goalX, goalY]
        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]

        return frame

    def push_frame(self, frame):
        """
        adds the given frame to the end of the objects state list.
        If the state has reached its maximum length (defined by timeframes in the constructor),
        it will pop its first element of the list before adding the given frame.

        :param frame: list -
            frame that should be added to the state
        """
        frame_norm = self.normalize(frame)
        if len(self.state) >= self.time_steps:
            self.state.pop(0)
            self.state.append(frame_norm)
            self.state_raw.pop(0)
            self.state_raw.append(frame)
        else:
            self.state.append(frame_norm)
            self.state_raw.append(frame)

    def update(self, dt, tarLinVel, tarAngVel, goal):
        """
        updates the robots position, direction and velocities.
        In addition to that a new frame is created

        :param dt: float -
            passed (simulated) time since last call
        :param tarLinVel: int/ float -
            target linear velocity
        :param tarAngVel: int/ float -
            target angular velocity
        :param goal: tuple -
            position of the current goal
        """
        posX, posY = self.getPosX(), self.getPosY()
        goalX, goalY = goal

        if not self.manuell:
            linVel, angVel = self.compute_next_velocity(dt, self.getLinearVelocity(), self.getAngularVelocity(),
                                                        tarLinVel, tarAngVel)
        else:
            linVel = self.linTast
            angVel = self.angTast

        direction = (self.getDirection() + (angVel * dt) + 2 * math.pi) % (2 * math.pi)
        posX += math.cos(direction) * linVel * dt
        posY += math.sin(direction) * linVel * dt

        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]
        frame = [posX, posY, direction, linVel, angVel, goalX, goalY]
        self.push_frame(frame)

    def compute_next_velocity(self, dt, linVel, angVel, tarLinVel, tarAngVel):
        """
        :param dt: float -
            passed (simulated) time since last call
        :param linVel: float -
            current linear velocity
        :param angVel: float -
            current angular velocity
        :param tarLinVel: float/ int -
            target linear velocity
        :param tarAngVel: float/ int -
            target angular velocity

        :return: tuple
            (float - linear velocity, float - angular velocity)
        """

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
        """
        :param station: Station.station -
            Target station object of the robot
        :return: Boolean
        """
        if self.getPosX() <= station.getPosX() + station.getWidth() and \
                self.getPosX() + self.width >= station.getPosX() and \
                self.getPosY() + self.length >= station.getPosY() and \
                self.getPosY() <= station.getPosY() + station.getLength():
            return True
        return False

    def isInCircleOfGoal(self, r):
        """
        Checks whether the robot is closer to its target than defined by the parameter r
        :param r: int -
            radius of an area (circle) around the goal
        :return: Boolean
        """
        return math.sqrt((self.getPosX() - self.getGoalX()) ** 2 +
                  (self.getPosY() - self.getGoalY()) ** 2) < r

    def hasGoal(self, station):
        """
        Checks whether the robot has reached its target
        :param station: Station.station -
            Target station object of the robot
        :return: Boolean
        """
        if self.getGoalX() == station.getPosX() and self.getGoalY() == station.getPosY():
            return True
        return False

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


