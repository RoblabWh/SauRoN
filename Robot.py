import math, random, time


# import keyboard
from pynput.keyboard import Key, Listener


class Robot:

    def __init__(self, position, startDirection, station, args, timeframes, walls, allStations):
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
        self.startDirectionX = math.cos((startDirection))
        self.startDirectionY = math.sin((startDirection))

        # print(self.startDirectionX, self.startDirectionY, startDirection)
        self.goalX, self.goalY = station.getPosX(), station.getPosY()
        self.state = []
        self.state_raw = []
        self.stateSonar = []
        self.netOutput = (0,0)
        self.distances = []
        self.radarHits = []
        self.angularDeviation = 0
        # [posX, posY, directionX, directionY, linearVelocity, angularVelocity, goalX, goalY, targetLinearVelocity, targetAngularVelocity]

        self.time_steps = timeframes #4
        # Robot Hardware Params
        self.width = 0.5  # m
        self.length = 0.5  # m
        self.radius = self.width / 2

        self.maxLinearVelocity = 0.7  # 10m/s
        self.minLinearVelocity = -0.7  # m/s
        self.maxLinearAcceleration = 1.5  # 5m/s^2
        self.minLinearAcceleration = -1.5  # 5m/s^2
        self.maxAngularVelocity = 1 * math.pi  # rad/s
        self.minAngularVelocity = -1 * math.pi  # rad/s
        self.maxAngularAcceleration = 0.5  # rad/s^2
        self.minAngularAcceleration = -0.5 # rad/s^2

        self.XYnorm = [args.arena_width, args.arena_length]
        self.directionnom = [-1, 1]#2 * math.pi]

        self.manuell = args.manually
        self.args = args

        self.station = station
        self.walls = walls

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
        """
        self.active = True
        if(pos != None):
            self.startposX, self.startposY = pos
        if(orientation != None):
            self.startDirectionX = math.cos(orientation)
            self.startDirectionY = math.sin(orientation)
        posX = self.startposX
        posY = self.startposY
        # randDirection = random.uniform(0, 2*math.pi)
        directionX = self.startDirectionX
        directionY = self.startDirectionY
        linVel = 0
        angVel = 0
        tarLinVel = 0
        tarAngVel = 0
        goalX = self.station.getPosX() #random.randrange(100, self.XYnorm[0]-100)#self.goalX
        goalY = self.station.getPosY() #random.randrange(100, self.XYnorm[1]-100)#self.goalY
        self.station.reposition(goalX,goalY)

        self.collidorStationsCircles = []
        for station in allStations:
            if not station is self.station:
                self.collidorStationsCircles.append((station.getPosX(), station.getPosY(), station.getRadius()))

        goalDist = math.sqrt((posX - goalX) ** 2 + (posY - goalY) ** 2)

        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]
        frame = [posX, posY, directionX, directionY, linVel, angVel, goalX, goalY, goalDist]

        for _ in range(self.time_steps):
            self.push_frame(frame)

        self.stateSonar = []

        self.distances = []
        self.radarHits = []
        self.netOutput = (0, 0)
        if walls != None:
            self.walls = walls

        self.bestDistToGoal = goalDist

    def resetSonar(self, robots):
        if self.args.mode == 'sonar':
            for _ in range(self.time_steps):
                self.sonarReading(robots, self.args.steps,self.args.steps)


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
        directionX = (frame[2] - self.directionnom[0]) / (self.directionnom[1] - self.directionnom[0])
        directionY = (frame[3] - self.directionnom[0]) / (self.directionnom[1] - self.directionnom[0])
        linVel = frame[4]/ self.maxLinearVelocity #(frame[4] - self.minLinearVelocity) / (self.maxLinearVelocity - self.minLinearVelocity)
        angVel = frame[5]/self.maxAngularVelocity #(frame[5] - self.minAngularVelocity) / (self.maxAngularVelocity - self.minAngularVelocity)
        goalX = frame[6] / self.XYnorm[0]
        goalY = frame[7] / self.XYnorm[1]
        dist = frame[8] / math.sqrt(self.XYnorm[0]**2+self.XYnorm[0]**2)
        # tarLinVel = (frame[7] - self.minLinearVelocity) / (self.maxLinearVelocity - self.minLinearVelocity)
        # tarAngVel = (frame[8] - self.minAngularVelocity) / (self.maxAngularVelocity - self.minAngularVelocity)

        frame = [posX, posY, directionX, directionY, linVel, angVel, goalX, goalY, dist]
        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY]
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
            linVel, angVel = self.compute_next_velocity_continuous(dt, self.getLinearVelocity(), self.getAngularVelocity(),
                                                        tarLinVel, tarAngVel)
        else:
            linVel = self.linTast
            angVel = self.angTast

        goalDist = math.sqrt((posX-goalX)**2+(posY-goalY)**2)

        direction = (self.getDirectionAngle() + (angVel * dt) + 2 * math.pi) % (2 * math.pi)
        posX += math.cos(direction) * linVel * dt
        posY += math.sin(direction) * linVel * dt
        directionVector = self.directionVectorFromAngle(direction)
        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY, tarLinVel, tarAngVel]
        # frame = [posX, posY, direction, linVel, angVel, goalX, goalY]
        frame = [posX, posY, directionVector[0], directionVector[1], linVel, angVel, goalX, goalY, goalDist]
        self.push_frame(frame)


    def sonarReading(self, robots, stepsLeft, steps):
        #TODO bei mehreren Stationen nicht die eigene als hindernis, nur andere
        #TODO Kollision mit Robotern /Geraden- Kreis Kollision
        colliders = self.walls + self.collidorStationsWalls

        circleCollisionPos = []
        for robotA in robots:
            if robotA is not  self:
                circleCollisionPos.append((robotA.getPosX(), robotA.getPosY(), robotA.getRadius()))
        #TODO runde ziele mit hinzufügen


        circleCollisionPos = circleCollisionPos + self.collidorStationsCircles

        self.lookAround(self.args.angle_steps, colliders, circleCollisionPos)

        # frame_sonar = []
        target = (self.station.posX, self.station.posY)
        # target = (self.station.posX + (self.station.width / 2), self.station.posY + (self.station.length/2))
        distance = math.sqrt((self.getPosX() - target[0]) ** 2 + (self.getPosY() - target[1]) ** 2)
        maxDist = math.sqrt(self.XYnorm[0] ** 2 + self.XYnorm[0] ** 2)

        # frame_sonar.append((distance / maxDist))

        #robot_orientation = self.getDirectionAngle()
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

        # frame_sonar.append(self.netOutput[0])
        # frame_sonar.append(self.netOutput[1])
        # frame_sonar.append(self.getLinearVelocityNorm())
        # frame_sonar.append(self.getAngularVelocityNorm())

        distancesNorm = self.distances/maxDist
        distancesNorm = distancesNorm.tolist()
        # for i in range(len(self.distances)):
        #     distancesNorm.append(self.distances[i] / maxDist)

        currentTimestep = (steps - stepsLeft)/steps


        frame_sonar = [distancesNorm, orientation, [(distance / maxDist)], [self.getLinearVelocityNorm(), self.getAngularVelocityNorm()], currentTimestep]

        if len(self.stateSonar) >= self.time_steps:
            self.stateSonar.pop(0)
            self.stateSonar.append(frame_sonar)
        else:
            self.stateSonar.append(frame_sonar)


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

    def compute_next_velocity_continuous(self, dt, linVel, angVel, tarLinVel, tarAngVel):
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

        tarAngVel = tarAngVel * self.maxAngularVelocity
        tarLinVel = tarLinVel * self.maxLinearVelocity



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

        # return tarLinVel, tarAngVel
        return linVel, angVel

    def directionVectorFromAngle(self, direction):
        angX = math.cos(direction)
        angY = math.sin(direction)
        return(angX,angY)

    def collideWithTargetStationCircular(self):
        """
        :param station: Station.station -
            Target station object of the robot
        :return: Boolean
        """
        station = self.station
        distance2StationCenter = math.sqrt((station.getPosX() - self.getPosX())**2 + (station.getPosY() - self.getPosY())**2) + self.radius
        return (distance2StationCenter < station.radius)

    def collideWithTargetStation(self):
        """
        :param station: Station.station -
            Target station object of the robot
        :return: Boolean
        """
        station = self.station

        # if self.getPosX() > station.getPosX() and self.getPosX() < station.getPosX()+station.getWidth():
        #     if self.getPosY()+self.radius > station.getPosY() and self.getPosY()-self.radius < station.getPosY()+station.getLength():
        #         return True
        # elif self.getPosY() > station.getPosY() and self.getPosY() < station.getPosY()+station.getLength():
        #     if self.getPosX()+self.radius > station.getPosX() and self.getPosX()-self.radius < station.getPosX()+station.getWidth():
        #         return True

        #Check if close to target
        if self.getPosX()+self.radius > station.getPosX():
            if self.getPosX()-self.radius < station.getPosX()+station.getWidth():
                if self.getPosY() + self.radius > station.getPosY():
                    if self.getPosY() - self.radius < station.getPosY() + station.getLength():
                        #check for corners
                        return True
        #                 if self.getPosX() < station.getPosX():
        #                     if self.getPosY() <
        #
        #
        #
        # if self.getPosX()-self.radius <= station.getPosX() + station.getWidth() and \
        #         self.getPosX()+self.radius + self.width >= station.getPosX() and \
        #         self.getPosY() + self.radius >= station.getPosY() and \
        #         self.getPosY() - self.radius <= station.getPosY() + station.getLength():
        #     #self.radarHits=[]
        #     return True
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
        return self.denormdata(self.state[self.time_steps - 1][7], [0, self.XYnorm[1]])

    def getVelocity(self):
        return self.getLinearVelocity(), self.getAngularVelocity()

    def getRadius(self):
        return self.width/2

    def isActive(self):
        return self.active

    def deactivate(self):
        self.active = False

    def getDirectionAngle(self, last=False):
        if not last:
            angX = self.getDirectionX()
            angY = self.getDirectionY()
        else:
            angX = self.getLastDirectionX()
            angY = self.getLastDirectionY()


        direction = 0
        if angX == 0:
            if (angY > 0):
                direction = 0.5 * math.pi
            elif angY < -1:
                direction = 1.5 * math.pi
            else:
                print("error! wrong input vector length | angX: ", angX, " | angY: " + angY)
        elif angY == 1:
            direction = 0.5 * math.pi
        elif angY == -1:
            direction = 1.5 * math.pi
        else:
            direction = math.atan(angY / angX)
            if angX < 0:
                direction = direction + math.pi
            elif angY < 0:
                direction = direction + 2 * math.pi

        direction = (direction + (2*math.pi)) % (2*math.pi)
        return direction

    def on_press(self, key):

        if key.char == 'w':
            self.linTast = 0.5
        if key.char == 'a':
            self.angTast = -0.005
        if key.char == 's':
            self.linTast = 0
        if key.char == 'd':
            self.angTast = 0.005
        if key.char == 'c':
            self.angTast = 0

    def on_release(self, key):
        self.linTast = 0
        self.angTast = 0

    def lookAround(self, alpha, collisionLines, roboterList = []):#TODO cas 7.8s und lookAraound 22.768a --> Optimierungsbedarf

        piFactor = (math.pi/180)
        twoPi = 2*math.pi
        radarHits = []
        distances = []
        dir = (self.getDirectionAngle() + (.5*math.pi)) % (2*math.pi)   #offsets the first Collision Line by 90 degrees to avoid edge errors during a convolution in neural net
        posX = self.getPosX()
        posY = self.getPosY()

        colLinesStartPoints= np.swapaxes(np.array([cl.getStart() for cl in collisionLines]),0,1) #[[x,x,x,x],[y,y,y,y]]
        colLinesEndPoints = np.swapaxes(np.array([cl.getEnd() for cl in collisionLines]),0,1)
        circleX = [r[0] for r in roboterList]
        circleY = [r[1] for r in roboterList]
        circleR = [r[2] for r in roboterList]


        rayCol = FastCollisionRay2([self.getPosX(), self.getPosY()], int(360/alpha), dir)
        rayHit = (rayCol.lineRayIntersectionPoint(colLinesStartPoints, colLinesEndPoints, np.array([circleX, circleY]), circleR))
        distances = (rayHit[0])
        radarHits = (rayHit[1])

        #for every line
            # for line in collisionLines:
            #
            #     lineN = line.getN()
            #     skalarProd = lineN[0] * rayV[0] + lineN[1] * rayV[1]
            #
            #     if skalarProd < 0:
            #         intersect = ray.cast(line)
            #
            #         if intersect is not None:
            #             intersections.append(intersect)
            #
            # nmbOfIntersections = len(intersections)
            # if  nmbOfIntersections > 1:
            #
            #     shortest = 0
            #     for i in range(1, nmbOfIntersections):
            #         if intersections[i][0] < intersections[shortest][0]:
            #             shortest = i
            #     radarHits.append(intersections[shortest][1])
            #     distances.append(intersections[shortest][0])
            #
            # elif nmbOfIntersections == 1:
            #     radarHits.append(intersections[0][1])
            #     distances.append(intersections[0][0])
            # else:
            #     radarHits.append([posX, posY])
            #     distances.append(0)

            # for circle in roboterList:
            #     intersectCircle = ray.castOnCircle((circle[0], circle[1]), circle[2], (posX, posY), radarHits[
            #         len(radarHits)-1])
            #     # print(intersectCircle)
            #     circleDists = []
            #     for i in range(0, len(intersectCircle)):
            #         circleDists.append( (posX-intersectCircle[i][0])**2 + (posY-intersectCircle[i][1])**2)
            #
            #     shortestIndex = -1
            #     shortestDist = distances[len(distances)-1] **2
            #     for i in range(0, len(circleDists)):
            #         if shortestDist > circleDists[i]:
            #             shortestDist = circleDists[i]
            #             shortestIndex = i
            #
            #     if shortestIndex is not -1:
            #         radarHits.pop(len(radarHits)-1)
            #         distances.pop(len(distances)-1)
            #         radarHits.append(intersectCircle[shortestIndex])
            #         distances.append(math.sqrt(shortestDist))
            #


        self.radarHits = radarHits
        self.distances = distances




import numpy as np
class FastCollisionRay2:
    def __init__(self, rayOrigin, rayCount, startAngle):
        self.rayOrigin = rayOrigin #np.array([[rayOrigin[0]],[rayOrigin[1]]], dtype=np.float)
        # für die dauer von raycount mache
        stepSize = 2*math.pi/rayCount
        steps = np.array([startAngle+i*stepSize for i in range(rayCount)])
        self.rayDirX = np.array([math.cos(step) for step in steps])
        self.rayDirY = np.array([math.sin(step) for step in steps])


    def lineRayIntersectionPoint(self, points1, points2, pointsRobots, radius):
        """

        :param points1: Liste der Startpunkte von Kollisionslinien points[[x1,x2,x3...xn],[y1,y2,y3...yn]]
        :param points2: Liste der Endpunkte von Kollisionslinien points[[x1,x2,x3...xn],[y1,y2,y3...yn]]
        :param pointsRobots: Liste derRoboterpositionen mit denen kollidiert werden kann als np.array[[x0, x2, x3..xn], [y0, y2, y3..yn]]
        :return:
        """

        x1 = self.rayOrigin[0] #originX
        y1 = self.rayOrigin[1] #originY
        x2V = np.swapaxes(np.array([self.rayDirX for _ in range(len(points1[0]))]),0,1) #directionX
        y2V = np.swapaxes(np.array([self.rayDirY for _ in range(len(points1[0]))]),0,1) #directionY
        x2 = x2V+x1
        y2 = y2V+y1

        x3 = np.array([points1[0] for _ in range(len(self.rayDirX))]) #lineStartXArray
        y3 = np.array([points1[1] for _ in range(len(self.rayDirX))]) #lineStartYArray
        x4 = np.array([points2[0] for _ in range(len(self.rayDirX))]) #lineEndXArray
        y4 = np.array([points2[1] for _ in range(len(self.rayDirX))]) #lineEndYArray


        #t1=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        #t2=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

        # Kleiner vorschlag von mir :)
        # Ersten den denominator berechnen, da er für t1 und t2 gleich ist.
        # 1 / ... um später für beiden die Multiplikation zu verwenden. Division ist die teuerste mathematische Operation ;)
        denominator = 1 / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

        # das koennte nochmal einen kleinen boost geben
        t1=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))*denominator
        t2=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))*denominator

        t1 = np.where((t2<0) | (t2>1), -1, t1)
        t1 = np.where(t1>=0, t1, 2048)
        # t1NearestHit = np.min(t1[np.greater_equal(t1, 0)])
        t1NearestHit = np.amin(t1, axis=1)

        collisionPoints = np.array([x1+t1NearestHit* x2V[:,0], y1+t1NearestHit* y2V[:,0]]) #[:,0] returns the first column # Aufbau nach [x0,x1…x2], [y0,y1…yn]]


        #TODO prüfen, für jedes Segement zwischen Robot-Origin und Collision Point prüfen, ob ein anderer Roboter dazwischen ist.
        # Dafür benötigt: Mittelpunkte aller Roboter und Linie (also start und Ziel)

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
        vLength = np.sqrt(vX**2 + vY**2)
        vX= vX/vLength
        vY= vY/vLength


        # a,b und c als Array zum Berechnen der Diskriminanten
        a = vX * vX + vY * vY # array voll skalarere Werte
        b = 2 * (vX * (x1 - qX) + vY * (y1 - qY))
        c = (x1**2 + y1**2) + (qX**2 + qY**2) - (2* (x1 * qX + y1*qY)) - radii**2

        disc = b**2 - 4 * a * c
        tc1 = np.where((disc>0), ((-b + np.sqrt(disc)) / (2 * a)), -1) #check if discriminat is negative --> no collision
        tc2 = np.where((disc>0), ((-b - np.sqrt(disc)) / (2 * a)), -1)
        tc1 = np.where((tc1>=0), tc1, 2048)
        tc2 = np.where((tc2>=0), tc2, 2048)

        smallestTOfCircle = np.where((tc1<tc2), tc1, tc2)
        smallestTOfCircle = np.amin(smallestTOfCircle, axis=0)

        t1NearestHit = np.where(((smallestTOfCircle<2048) & (smallestTOfCircle<t1NearestHit)), smallestTOfCircle, t1NearestHit)

        collisionPoints = np.array([x1+t1NearestHit* x2V[:,0], y1+t1NearestHit* y2V[:,0]]) #[:,0] returns the first column # Aufbau nach [x0,x1…x2], [y0,y1…yn]]
        #TODO der neu t wert ist nihct normiert und daher muss er auf seinen eigenen vektor berchnet werden und nihct auf den normierten ursprungsvektor



        collisionPoints = np.swapaxes(collisionPoints, 0, 1)#für Rückgabe in x,y-Paaren
        return [t1NearestHit, collisionPoints]




class Ray:
    def __init__(self, x, y, angle):
        self.pos = [x, y]
        self.dir = [math.cos(angle), math.sin(angle)]

    def getVector(self):
        x = (self.dir[0])
        y = (self.dir[1])
        return (x,y)

    def cast(self, line):
        # start point
        x1 = line.a[0]
        y1 = line.a[1]
        # end point
        x2 = line.b[0]
        y2 = line.b[1]

        # position of the ray
        x3 = self.pos[0]
        y3 = self.pos[1]
        x4 = self.pos[0] + self.dir[0]
        y4 = self.pos[1] + self.dir[1]

        # denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        # numerator
        num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        if den == 0:
            return None

        # formulars
        t = num / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if t > 0 and t < 1 and u > 0:
            # Px, Py
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            pot = [x, y]
            # return pot
            return [u,pot]



    def castOnCircle(self, circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
        """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

        :param circle_center: The (x, y) location of the circle center
        :param circle_radius: The radius of the circle
        :param pt1: The (x, y) location of the first point of the segment
        :param pt2: The (x, y) location of the second point of the segment
        :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
        :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
        :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

        Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
        """

        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center # p1 = roboter pos | p2 = aktueller nächster kollisionspunkt | circle_center trivial
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy) #Vektor con Kreismitte zu P1 und Kreismitte zu P2
        dx, dy = (x2 - x1), (y2 - y1) #Vektor von P1 zu P2
        dr = (dx ** 2 + dy ** 2)**.5 #Betrag
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # No intersection between circle and line
            return []
        else:  # There may be 0, 1, or 2 intersections with the segment
            if dr != 0:
                intersections = [
                    (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
                     cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
                    for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
                if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
                    fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
                    intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
                if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
                    return [intersections[0]]
                else:
                    return intersections
            else:
                return []


