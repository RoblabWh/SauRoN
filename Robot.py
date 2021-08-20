import math, random, time

#from tensorflow.python.framework.test_ops import old

from Borders import ColliderLine


# import keyboard
from pynput.keyboard import Key, Listener


class Robot:
    """
    Defines a Robot that can move inside of simulation
    """

    def __init__(self, position, startOrientation, station, args, walls, allStations):
        """
        :param position: tuple (float,float) -
            defines the robots starting position
        :param startOrientation: float -
            defines the robots starting orientation
            one evolution is equal to 2*pi
            a value of 0 causes the robot to look right
            by increasing the value the robots start orientation will change clockwise
        :param station:Station.Station -
            defines the target of the robot
        :param args:
            args defined in main
        :param walls: list of Borders.ColliderLines -
            should at least include the borders of the arena
        :param allStations: list of Station.Stations
            target stations of the other robots can be used as a collider
        """
        self.startposX, self.startposY = position
        self.startDirectionX = math.cos(startOrientation)
        self.startDirectionY = math.sin(startOrientation)
        self.startOrientation = startOrientation
        self.goalX, self.goalY = station.getPosX(), station.getPosY()

        # Variables regarding the state
        self.time_steps = args.time_frames #4
        self.state = []
        self.state_raw = []
        self.stateLidar = []
        self.netOutput = (0,0)
        self.distances = []
        self.radarHits = []
        self.collisionDistances = []
        self.angularDeviation = 0

        # Robot Hardware Params
        self.width = 0.35  # m
        self.length = 0.35  # m
        self.radius = self.width / 2

        self.maxLinearVelocity = 0.7  # m/s
        self.minLinearVelocity = -0.7  # m/s
        self.maxLinearAcceleration = 1.5  # m/s^2
        self.minLinearAcceleration = -1.5  # m/s^2
        self.maxAngularVelocity = 1.5 * math.pi  # rad/s
        self.minAngularVelocity = -1.5 * math.pi  # rad/s
        self.maxAngularAcceleration = 1.5 * math.pi   #rad/s^2
        self.minAngularAcceleration = -1.5 * math.pi  #rad/s^2

        if args.load_christian:
            self.width = 0.35  # m
            self.length = 0.35  # m
            self.radius = self.width / 2
            self.minLinearVelocity = 0
            self.maxLinearVelocity = 0.6
            self.maxAngularVelocity = 1.5#1.5#1.5 #* math.pi
            self.minAngularVelocity = -1.5#-1.5#-1.5 #* math.pi
            self.maxAngularAcceleration = 1.5 #* math.pi  # rad/s^2
            self.minAngularAcceleration = -1.5 #* math.pi  # rad/s^2
            #v = 0.6


        self.maxLinearVelocityFact = 1/self.maxLinearVelocity
        self.maxAngularVelocityFact = 1/self.maxAngularVelocity


        self.XYnorm = [args.arena_width, args.arena_length]
        self.XYnormFact = [1/args.arena_width, 1/args.arena_length]
        self.directionnom = [-1, 1]#2 * math.pi]


        if args.load_christian: self.maxDistFact = 1/20 #FÜR CHRISTIANS Netz
        else: self.maxDistFact = 1/math.sqrt(self.XYnorm[0] ** 2 + self.XYnorm[1] ** 2)

        self.manuell = args.manually
        self.args = args
        self.station = station
        self.walls = walls

        self.hasPieSlice = args.has_pie_slice
        self.pieSlicePoints = []
        self.pieSliceWalls = []
        if self.hasPieSlice:
            self.pieSliceWalls = [ColliderLine(0,0,1,1), ColliderLine(0,0,1,1), ColliderLine(0,0,1,1), ColliderLine(0,0,1,1), ColliderLine(0,0,1,1)]
        self.posSensor = []
        self.robotsPieSliceWalls = []
        self.offsetSensorDist = 0.08
        normFac = 1 / self.radius
        self.offsetAngle = 2/3 * np.arccos((np.sqrt(2-(self.offsetSensorDist* normFac)**2)-(self.offsetSensorDist* normFac)) / 2)


        #only use with rectangular targets
        self.collidorStationsWalls = []
        # for pickUp in allStations:
        #     if not station is pickUp:
        #         self.collidorStationsWalls = self.collidorStationsWalls + pickUp.borders
        self.collidorStationsCircles = []
        for pickUp in allStations:
            if not station is pickUp:
                self.collidorStationsCircles.append((pickUp.getPosX(), pickUp.getPosY(), pickUp.getRadius()))

        if self.manuell:
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            self.linTast = 0
            self.angTast = 0





    def reset(self, allStations, pos = None, orientation = None, walls = None):
        """
        resets the robots position (to his starting position), his orientation (to his starting orientation)
        and sets all velocities back to zero.
        In addition the state gets cleared.
        This method is typically called at the beginning of a training epoch.


        It can also provide information about a new level (different walls, starting position...)

        :param allStations: list of Station.Stations
            target stations of the other robots can be used as a collider
        :param pos:tuple (float,float) -
            defines the robots starting position
            can be used to set a new start position in case of a level change
        :param orientation: float -
            defines the robots starting orientation
            can be used to set a new start orientation in case of a level change
        :param walls: list of Borders.ColliderLines -
            should at least include the borders of the arena
            can be used to set walls in case of a level change
        """

        if(pos != None):
            self.startposX, self.startposY = pos
        posX = self.startposX
        posY = self.startposY

        if(orientation != None):
            self.startDirectionX = math.cos(orientation)
            self.startDirectionY = math.sin(orientation)
            self.startOrientation = orientation
        else:
            orientation = self.startOrientation
        directionX = self.startDirectionX
        directionY = self.startDirectionY


        if walls != None:
            self.walls = walls

        self.collidorStationsCircles = []
        for station in allStations:
            if not station is self.station:
                self.collidorStationsCircles.append((station.getPosX(), station.getPosY(), station.getRadius()))


        self.active = True
        linVel = 0
        angVel = 0
        tarLinVel = 0
        tarAngVel = 0
        self.goalX = self.station.getPosX()
        self.goalY = self.station.getPosY()
        goalDist = math.sqrt((posX - self.goalX) ** 2 + (posY - self.goalY) ** 2)

        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]
        frame = [posX, posY, directionX, directionY, linVel, angVel, self.goalX, self.goalY, goalDist, orientation]


        for _ in range(self.time_steps):
            self.push_frame(frame)


        self.stateLidar = []


        self.distances = []
        self.collisionDistances = []
        self.radarHits = []
        self.netOutput = (0, 0)

        self.bestDistToGoal = goalDist

        if self.hasPieSlice:
            self.posSensor = [posX + self.offsetSensorDist * directionX, posY + self.offsetSensorDist * directionY]
            self.calculatePieSlice((directionX,directionY))
        else:
            self.posSensor = [posX, posY]

    def resetLidar(self, robots):
        if self.args.mode == 'sonar':
            if self.hasPieSlice:
                self.robotsPieSliceWalls = []
                for robot in robots:
                    if robot is not self:
                        self.robotsPieSliceWalls += robot.getPieSliceWalls()

            for _ in range(self.time_steps):
                self.lidarReading(robots, self.args.steps, self.args.steps)


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


    def normalize(self, frame):
        """
        normalizes all values of a frame

        :param frame: list -
            [posX, posY, directionX, directionY, linVel, angVel, goalX, goalY, dist]
        :return: list -
            normalized frame with values only between 0 an 1
        """
        posX = frame[0] * self.XYnormFact[0]
        posY = frame[1] * self.XYnormFact[1]
        directionX = frame[2]
        directionY = frame[3]
        linVel = (frame[4] - ((self.minLinearVelocity + self.maxLinearVelocity) * 0.5)) /  ((self.maxLinearVelocity - self.minLinearVelocity) * 0.5)
        #self.maxLinearVelocityFact
        angVel = (frame[5] - ((self.minAngularVelocity + self.maxAngularVelocity) * 0.5)) /  ((self.maxAngularVelocity - self.minAngularVelocity) * 0.5)
        #self.maxAngularVelocityFact
        goalX = frame[6] * self.XYnormFact[0]
        goalY = frame[7] * self.XYnormFact[1]
        dist = frame[8] * self.maxDistFact

        frame = [posX, posY, directionX, directionY, linVel, angVel, goalX, goalY, dist]

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


    def update(self, dt, tarLinVel, tarAngVel):
        """
        updates the robots position, direction and velocities.
        In addition to that a new frame is created

        :param dt: float -
            passed (simulated) time since last call
        :param tarLinVel: int/ float -
            target linear velocity
        :param tarAngVel: int/ float -
            target angular velocity
        """
        posX, posY = self.getPosX(), self.getPosY()
        goalX, goalY = self.goalX, self.goalY

        if not self.manuell:
            linVel, angVel = self.computeNextVelocityContinuous(dt, self.getLinearVelocity(), self.getAngularVelocity(),
                                                                tarLinVel, tarAngVel)
        else:
            linVel = self.linTast
            angVel = self.angTast

        oldDir = self.getDirectionAngle()
        directionVector = self.directionVectorFromAngle(oldDir)

        deltaPosX = directionVector[0] * linVel * dt  # math.cos(direction) * linVel * dt
        deltaPosY = directionVector[1] * linVel * dt  # math.sin(direction) * linVel * dt
        posX += deltaPosX
        posY += deltaPosY


        direction = (self.getDirectionAngle() + (angVel * dt) + 2 * math.pi) % (2 * math.pi)
        directionVector = self.directionVectorFromAngle(direction)
        deltaDir = direction - oldDir



        goalDist = math.sqrt((posX-goalX)**2+(posY-goalY)**2)


        frame = [posX, posY, directionVector[0], directionVector[1], linVel, angVel, goalX, goalY, goalDist, direction]
        self.push_frame(frame)
        if self.hasPieSlice:
            self.updatePieSlice(deltaDir, (deltaPosX, deltaPosY))
        else:
            self.posSensor = [posX, posY]

    def calculatePieSlice(self, dirV):
        offsetSensorDist = self.offsetSensorDist
        posX, posY = self.getPosX(), self.getPosY()
        dirVx, dirVy = dirV
        rearX, rearY = -self.radius * dirVx, -self.radius * dirVy
        angle = self.offsetAngle
        offsets = [angle * 1.5, angle * .5, angle * -.5, angle * -1.5]

        p0x, p0y = posX + dirVx * offsetSensorDist, posY + dirVy * offsetSensorDist
        points = [(p0x, p0y)]

        o = np.atleast_2d([posX, posY])
        for offsetsAngle in offsets:
            R = np.array([[np.cos(offsetsAngle), -np.sin(offsetsAngle)],
                          [np.sin(offsetsAngle), np.cos(offsetsAngle)]])
            p = np.atleast_2d(np.asarray([rearX, rearY]))
            points += [np.squeeze((R @ (p.T) + o.T ).T).tolist()]

        self.posSensor = points[0]
        self.pieSlicePoints = points

        walls = self.pieSliceWalls
        for i, w in enumerate(walls):
            w.updatePos(points[i], points[(i+1)%len(walls)])



    def updatePieSlice(self, deltaAngle, deltaPos):
        points = self.pieSlicePoints
        posX, posY = self.getPosX(), self.getPosY()
        deltaX, deltaY = deltaPos

        oldPos = np.atleast_2d([posX-deltaX, posY-deltaY])
        newPos = np.atleast_2d([posX, posY])
        R = np.array([[np.cos(deltaAngle), -np.sin(deltaAngle)],
                      [np.sin(deltaAngle), np.cos(deltaAngle)]])
        p = np.atleast_2d(np.asarray(points))
        points = np.squeeze((R @ (p.T - oldPos.T) + newPos.T).T).tolist()



        walls = self.pieSliceWalls
        for i, w in enumerate(walls):
            w.updatePos(points[i], points[(i + 1) % len(walls)])

        self.posSensor = points[0]
        self.pieSlicePoints = points

    def lidarReading(self, robots, stepsLeft, steps):
        """
        Creates a state with a virtual 2D laser scan

        The state consists of n frames (a frame is data from a certain timeframe) with each containing the following data:
        [normalised laser distances of 2D laser scan, angular Deviation between the robots forward axis and its target represented by vector of the length 1,
         normalised distance to the robots target, [normalised linear velocity, normalised angular velocity], current timestep]

        The state is used to train the neural net

        :param robots: list of Robot.Robot objects -
            the positions of the other robots are needed for the laser scan
        :param stepsLeft: remaining steps of current epoch
        :param steps: number of steps in one epoch
        """

        dir = (self.getDirectionAngle() - (self.args.field_of_view / 2)) % (2 * math.pi)

        colliderLines = self.walls + self.collidorStationsWalls + self.robotsPieSliceWalls
        collidorCirclePosWithoutRobots = []
        collidorCirclePosOnlyRobots = []
        collidorCircleAllForTerminations = []

        for robotA in robots:
            if robotA is not self:
                collidorCirclePosOnlyRobots.append((robotA.getPosX(), robotA.getPosY(), robotA.getRadius()))

        if self.args.collide_other_targets:
            collidorCirclePosWithoutRobots = self.collidorStationsCircles



        colLinesStartPoints = np.swapaxes(np.array([cl.getStart() for cl in colliderLines]), 0, 1)  # [[x,x,x,x],[y,y,y,y]]
        colLinesEndPoints = np.swapaxes(np.array([cl.getEnd() for cl in colliderLines]), 0, 1)
        normals = np.swapaxes(np.array([cl.getN() for cl in colliderLines]), 0, 1)

        collidorCircleAllForTerminations = collidorCirclePosWithoutRobots + collidorCirclePosOnlyRobots

        if self.hasPieSlice:
            position = self.posSensor
            usedCircleCollider = collidorCirclePosWithoutRobots
        else:
            position = [self.getPosX(), self.getPosY()]
            usedCircleCollider = collidorCircleAllForTerminations


        circleX = [r[0] for r in usedCircleCollider]
        circleY = [r[1] for r in usedCircleCollider]
        circleR = [r[2] for r in usedCircleCollider]

        circlesPositions = np.array([circleX, circleY])

        rayCol = FastCollisionRay(position, self.args.number_of_rays, dir, self.radius, self.args.field_of_view)
        distances, radarHits = (rayCol.lineRayIntersectionPoint(colLinesStartPoints, colLinesEndPoints, normals, circlesPositions, circleR, self.offsetSensorDist))


        circleX = [r[0] for r in collidorCircleAllForTerminations]
        circleY = [r[1] for r in collidorCircleAllForTerminations]
        circleR = [r[2] for r in collidorCircleAllForTerminations]
        circlesPositionsAll = np.array([circleX, circleY])
        self.collisionDistances = rayCol.shortestDistanceToCollidors([self.getPosX(), self.getPosY()], colliderLines, circlesPositionsAll, circleR)



        self.radarHits = (radarHits)
        self.distances = (distances)


        # frame_lidar = []
        target = (self.station.posX, self.station.posY)
        # target = (self.station.posX + (self.station.width / 2), self.station.posY + (self.station.length/2))
        distance = math.sqrt((self.getPosX() - target[0]) ** 2 + (self.getPosY() - target[1]) ** 2)

        oriRobotV = (self.getDirectionX(), self.getDirectionY())
        oriTargetV = ((self.getPosX() - target[0]),(self.getPosY() - target[1]))
        skalarProd = oriRobotV[0]*oriTargetV[0]+oriRobotV[1]*oriTargetV[1]
        oriTargetVLength = distance
        oriRobotVLength = 1
        ratio = skalarProd/(oriTargetVLength*oriRobotVLength)

        if ratio>1 or ratio<-1:
            print("oriRobotV:", oriRobotV, "| oriTargetV:", oriTargetV, "| skalarProd:", skalarProd, "| oriTargetVLength:", oriTargetVLength, "|ratio: ", ratio)
            if ratio<-1:
                ratio = -1
            else:
                ratio = 1

        angularDeviation = math.acos(ratio)

        c = (self.getPosX()+oriRobotV[0], self.getPosY()+oriRobotV[1])
        angularDeviation = angularDeviation - math.pi
        if ((target[0] - self.getPosX()) * (c[1] - self.getPosY()) - (target[1] - self.getPosY()) * (c[0] - self.getPosX())) < 0:
            angularDeviation = angularDeviation*-1

        self.angularDeviation = angularDeviation
        # print(angularDeviation)

        anglDeviationV = self.directionVectorFromAngle(angularDeviation)

        orientation = [anglDeviationV[0], anglDeviationV[1]]
        self.debugAngle = orientation


        distancesNorm = self.distances* self.maxDistFact
        if self.args.load_christian:
            distancesNorm = np.where(distancesNorm > 1, 1, distancesNorm)  # FÜR CHRISTIANS SIM
        distancesNorm = distancesNorm.tolist()

        currentTimestep = (steps - stepsLeft)/steps #TODO setps im Konstruktor übergeben und einenFaktor draus machen, muss nicht bei jedem Aufruf mit übergeben werden


        frame_lidar = [distancesNorm, orientation, [(distance * self.maxDistFact)], [self.getLinearVelocityNorm(), self.getAngularVelocityNorm()], currentTimestep]

        if len(self.stateLidar) >= self.time_steps:
            self.stateLidar.pop(0)
            self.stateLidar.append(frame_lidar)
        else:
            self.stateLidar.append(frame_lidar)



    def compute_next_velocity(self, dt, linVel, angVel, tarLinVel, tarAngVel):
        """
        DEPRECATED

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

    def computeNextVelocityContinuous(self, dt, linVel, angVel, tarLinVel, tarAngVel):
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
        self.netOutput = (tarAngVel, tarLinVel)

        tarLinVel = max(-1, min(tarLinVel, 1))
        tarAngVel = max(-1, min(tarAngVel, 1))

        # tarAngVel = tarAngVel * ((self.maxAngularVelocity - self.minAngularVelocity)* 0.5) + (self.maxAngularVelocity + self.minAngularVelocity) * 0.5
        tarAngVel = tarAngVel * ((self.maxAngularVelocity - self.minAngularVelocity)* 0.5) + ((self.minAngularVelocity + self.maxAngularVelocity) * 0.5)
        tarLinVel = tarLinVel * ((self.maxLinearVelocity - self.minLinearVelocity)* 0.5) + ((self.minLinearVelocity + self.maxLinearVelocity) * 0.5)






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

        #return tarLinVel, tarAngVel
        return linVel, angVel


    def directionVectorFromAngle(self, direction):
        """
        calculates a vector of length 1 based on the direction in radians
        :param direction: float direction in radians
        :return: (float, float) direction as a vector
        """
        angX = math.cos(direction)
        angY = math.sin(direction)
        return(angX,angY)

    def collideWithTargetStationCircular(self):
        """

        :return: Boolean
        """
        station = self.station
        distance2StationCenter = math.sqrt((station.getPosX() - self.getPosX())**2 + (station.getPosY() - self.getPosY())**2) + self.radius
        return (distance2StationCenter < station.radius)

    def collideWithTargetStation(self):
        """

        :return: Boolean
        """
        station = self.station

        #Check if close to target
        if self.getPosX()+self.radius > station.getPosX():
            if self.getPosX()-self.radius < station.getPosX()+station.getWidth():
                if self.getPosY() + self.radius > station.getPosY():
                    if self.getPosY() - self.radius < station.getPosY() + station.getLength():
                        #check for corners
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

    def getLastPosX(self):
        return self.state_raw[self.time_steps - 2][0]

    def getLastPosY(self):
        return self.state_raw[self.time_steps - 2][1]

    def getDirectionX(self):
        return self.state_raw[self.time_steps - 1][2]

    def getDirectionY(self):
        return self.state_raw[self.time_steps - 1][3]

    def getLastDirectionX(self):
        return self.state_raw[self.time_steps - 2][2]

    def getLastDirectionY(self):
        return self.state_raw[self.time_steps - 2][3]

    def getLinearVelocity(self):
        return self.state_raw[self.time_steps - 1][4]

    def getAngularVelocity(self):
        return self.state_raw[self.time_steps - 1][5]

    def getLinearVelocityNorm(self):
        return self.state[self.time_steps - 1][4]

    def getAngularVelocityNorm(self):
        return self.state[self.time_steps - 1][5]

    def getGoalX(self):
        return self.state_raw[self.time_steps - 1][6]

    def getGoalY(self):
        # return self.denormdata(self.state[self.time_steps - 1][7], [0, self.XYnorm[1]])
        return self.state_raw[self.time_steps - 1][7]

    def getVelocity(self):
        return self.getLinearVelocity(), self.getAngularVelocity()

    def getRadius(self):
        return self.width * .5

    def getPieSliceWalls(self):

        return self.pieSliceWalls

    def isActive(self):
        return self.active

    def deactivate(self):
        self.active = False

    def getDirectionAngle(self, last=False):
        """
        :param last:
        :return: Current forward Dir in Range of 0 to 2Pi
        """
        if not last:
            return self.state_raw[self.time_steps - 1][9]
        return self.state_raw[self.time_steps - 2][9]


    def on_press(self, key):
        if key.char == 'w':
            self.linTast = 1.5
        if key.char == 'a':
            self.angTast = -1.6
        if key.char == 's':
            self.linTast = -1.5
        if key.char == 'd':
            self.angTast = 1.6
        if key.char == 'c':
            self.angTast = 0

    def on_release(self, key):
        self.linTast = 0
        self.angTast = 0






import numpy as np
class FastCollisionRay:
    """
    A class for simulating light rays around the robot by using numpy calculations to find possible intersections
    with other line segments or circles.
    """

    def __init__(self, rayOrigin, numberOfRays, startAngle, radius, fov):
        """
        Creates multiple laser rays around the robot for a fast calculation of intersections
        (and distances to those intersections)

        :param rayOrigin: (x,y) position of the robot
        :param rayCount: number of rays
        :param startAngle: float - usually the orientation of the robot added by some amount to prevent edge problems
        during the feature detection in the convolutional layers of the neural net.
        :param radius: the radius of the robot itself
        """
        self.rayOrigin = rayOrigin

        stepSize = fov / numberOfRays
        steps = np.array([startAngle+i*stepSize for i in range(numberOfRays)])
        self.rayDirX = np.array([math.cos(step) for step in steps])
        self.rayDirY = np.array([math.sin(step) for step in steps])
        self.ownRadius = radius

    def lineRayIntersectionPoint(self, points1, points2, normals, pointsRobots, radius, sensorOffset):
        """

        :param points1: List of starting points of collision lines - points[[x1,x2,x3...xn],[y1,y2,y3...yn]]
        :param points2: List of ending points of collision lines - points[[x1,x2,x3...xn],[y1,y2,y3...yn]]
        :param pointsRobots: List of all robots positions which can be a collider as an np.array[[x0, x2, x3..xn], [y0, y2, y3..yn]]
        :return:
        """

        x1 = self.rayOrigin[0] #originX
        y1 = self.rayOrigin[1] #originY
        x2V = np.swapaxes(np.array([self.rayDirX for _ in range(len(points1[0]))]),0,1) #directionX
        y2V = np.swapaxes(np.array([self.rayDirY for _ in range(len(points1[0]))]),0,1) #directionY
        x2 = x2V+x1
        y2 = y2V+y1

        nX = np.array([normals[0] for _ in range(len(self.rayDirX))]) #normalsXArray
        nY = np.array([normals[1] for _ in range(len(self.rayDirX))]) #normalsXArray

        skalarProd = nX * x2V + nY * y2V

        x3 = np.array([points1[0] for _ in range(len(self.rayDirX))]) #lineStartXArray
        y3 = np.array([points1[1] for _ in range(len(self.rayDirX))]) #lineStartYArray
        x4 = np.array([points2[0] for _ in range(len(self.rayDirX))]) #lineEndXArray
        y4 = np.array([points2[1] for _ in range(len(self.rayDirX))]) #lineEndYArray


        #t1=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        #t2=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))


        # Ersten den denominator berechnen, da er für t1 und t2 gleich ist.
        # 1 / ... um später für beiden die Multiplikation zu verwenden. Division ist die teuerste mathematische Operation ;)
        denominator = np.where(skalarProd<0, 1.0 / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)), -1)

        # t1=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))*denominator #Faktor des Laserstrahls vom Roboter bis zum Schnittpunkt
        # t2=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))*denominator #Faktor vom Startpunkt des Geradenabschnitts bis zum Schnittpunkt
        t1=np.where(skalarProd<0, ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))*denominator,-1) #Faktor des Laserstrahls vom Roboter bis zum Schnittpunkt
        t2=np.where(skalarProd<0,((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))*denominator, -1) #Faktor vom Startpunkt des Geradenabschnitts bis zum Schnittpunkt

        # Liegt der Schnitt bei einem t2<0 oder t2>1 ist der Schnitt nicht auf dem Geradenabschnitt
        # Liegt der Schnitt bei einem t1<0 ist der Schnitt in negativer Richtung des Lichtstrahls
        # der kleinste t1 Faktor pro Strahl wird ausgewählt (das ist der Schnittpunkt, der am nächsten liegt)
        t1 = np.where((t2<0) | (t2>1), -1, t1)
        t1 = np.where(t1>=0, t1, 2048)
        t1NearestHit = np.amin(t1, axis=1)

        collisionPoints = np.array([x1+t1NearestHit* x2V[:,0], y1+t1NearestHit* y2V[:,0]]) #[:,0] returns the first column # Aufbau nach [x0,x1…x2], [y0,y1…yn]]


        # prüfen, für jedes Segement zwischen Robot-Origin und Collision Point prüfen, ob ein anderer Roboter dazwischen ist.
        # Dafür benötigt: Mittelpunkte aller Roboter und Linie (also Start und Ziel)


        if len(pointsRobots[0]) > 0:
            qX = np.array([pointsRobots[0] for _ in range(len(collisionPoints[0]))])
            qY = np.array([pointsRobots[1] for _ in range(len(collisionPoints[0]))])
            radii = np.array([radius for _ in range(len(collisionPoints[0]))])
            qX = np.swapaxes(qX,0,1)
            qY = np.swapaxes(qY,0,1)
            radii = np.swapaxes(radii,0,1)
            colX = np.array([collisionPoints[0] for _ in range(len(pointsRobots[0]))])
            colY = np.array([collisionPoints[1] for _ in range(len(pointsRobots[0]))])


            vX = colX - x1
            vY = colY - y1
            # normalize vector to a length of 1, so that the t parameters of the line-line intersection can be compared with the t's from cirle-line intersection
            vLengthFact = 1/np.sqrt(vX**2 + vY**2) #again division costs too much for us poor students so we like to use multiplications with a factor
            vX= vX * vLengthFact
            vY= vY * vLengthFact


            # a,b und c als Array zum Berechnen der Diskriminanten
            a = vX * vX + vY * vY # array voll skalarere Werte
            b = 2 * (vX * (x1 - qX) + vY * (y1 - qY))
            c = (x1**2 + y1**2) + (qX**2 + qY**2) - (2* (x1 * qX + y1*qY)) - radii**2

            disc = b**2 - 4 * a * c
            denominator = 1/ (2 * a)
            tc1 = np.where((disc>0), ((-b + np.sqrt(disc)) * denominator), -1) #check if discriminat is negative --> no collision
            tc2 = np.where((disc>0), ((-b - np.sqrt(disc)) * denominator), -1)
            tc1 = np.where((tc1>=0), tc1, 2048)
            tc2 = np.where((tc2>=0), tc2, 2048)

            smallestTOfCircle = np.where((tc1<tc2), tc1, tc2)
            smallestTOfCircle = np.amin(smallestTOfCircle, axis=0)

            t1NearestHit = np.where(((smallestTOfCircle<2048) & (smallestTOfCircle<t1NearestHit)), smallestTOfCircle, t1NearestHit)

        collisionPoints = np.array([x1+t1NearestHit* x2V[:,0], y1+t1NearestHit* y2V[:,0]]) #[:,0] returns the first column # Aufbau nach [x0,x1…x2], [y0,y1…yn]]

        # t1NearestHit = t1NearestHit - (self.ownRadius-sensorOffset) #AUSKOMMENTIEREN FÜR CHRISTIANS NETZ


        collisionPoints = np.swapaxes(collisionPoints, 0, 1)#für Rückgabe in x,y-Paaren
        return [t1NearestHit, collisionPoints]

    def shortestDistanceToCollidors(self, pos, lineSegments, circles, radii):
        x1 = pos[0]  # originX
        y1 = pos[1]  # originY

        x2 = np.array([lineSegments[i].getStart()[0] for i in range(len(lineSegments))])  # lineStartXArray
        y2 = np.array([lineSegments[i].getStart()[1] for i in range(len(lineSegments))])  # lineStartYArray
        x3 = np.array([lineSegments[i].getEnd()[0] for i in range(len(lineSegments))])  # lineEndXArray
        y3 = np.array([lineSegments[i].getEnd()[1] for i in range(len(lineSegments))])  # lineEndYArray

        sqrDist = (x2 - x3) ** 2 + (y2 - y3) ** 2
        t = np.where((sqrDist == 0), 0, ((x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)) / sqrDist)
        t = np.clip(t, 0, 1)

        dist = (x1 - (x2 + t * (x3 - x2))) ** 2 + (y1 - (y2 + t * (y3 - y2))) ** 2
        dist = np.sqrt(dist)

        if len(circles[0]) > 0:
            x4 = np.array([circles[0][i] for i in range(len(circles[0]))])  # circleXArray
            y4 = np.array([circles[1][i] for i in range(len(circles[1]))])  # circleYArray
            r = np.array([radii[i] for i in range(len(radii))])        # circleradiusArray

            distCircles = np.sqrt((x1-x4)**2 + (y1-y4)**2) - r
            dist = np.concatenate((dist, distCircles))

        return dist

