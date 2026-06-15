import warnings

from Environment.Components.Border import ColliderLine

import math
#from pynput.keyboard import Listener
import copy
import numpy as np
from utils import scan1DTo2D, CircularBuffer
import torch

import os
import time
from PIL import Image


class Robot:
    """
    Defines a Robot that can move inside of simulation
    """

    def __init__(self, idx, position, startOrientation, station, args, walls, allStations, circleWalls):
        """
        :param position: tuple (float,float) -
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
        self.args = args
        
        self.idx = idx

        self.startposX, self.startposY = position
        self.buffersize = 80
        self.last_positions = CircularBuffer(self.buffersize)
        self.last_positions.add(self.startposX, self.startposY)
        self.startDirectionX = math.cos(startOrientation)
        self.startDirectionY = math.sin(startOrientation)
        self.startOrientation = startOrientation
        self.goalX, self.goalY = station.getPosX(), station.getPosY()
        self.station = station

        # Variables regarding the state
        self.time_frames = args.time_frames #4
        self.state_raw = []
        self.robot_states = []
        self.netOutput = (0,0)
        self.distances = []
        self.lidarHits = []
        self.lidarHitMask = []
        self.collisionDistances = []
        self.collisionDistancesRobots = []
        self.angularDeviation = 0
        self.fieldOfView = args.field_of_view / 180 * np.pi

        # Robot Hardware Params
        # self.width = 0.35  # m
        # self.length = 0.35  # m
        # self.radius = self.width / 2
        #
        # self.maxLinearVelocity = 0.6  # m/s
        # self.minLinearVelocity = 0  # m/s
        # self.maxLinearAcceleration = 1.5  # m/s^2
        # self.minLinearAcceleration = -1.5  # m/s^2
        # self.maxAngularVelocity = 1.5  # rad/s
        # self.minAngularVelocity = -1.5 # rad/s
        # self.maxAngularAcceleration = 1.5 * math.pi   #rad/s^2
        # self.minAngularAcceleration = -1.5 * math.pi  #rad/s^2
        self.width = 0.35  # m
        self.length = 0.35  # m
        self.radius = self.width / 2

        self.maxLinearVelocity = 0.175  # m/s
        self.minLinearVelocity = 0  # m/s
        self.maxLinearAcceleration = 0.05  # m/s^2
        self.minLinearAcceleration = -0.05  # m/s^2
        self.maxAngularVelocity = 1  # rad/s
        self.minAngularVelocity = -1 # rad/s
        self.maxAngularAcceleration = 0.05 * math.pi   #rad/s^2
        self.minAngularAcceleration = -0.05 * math.pi  #rad/s^2

        #Factors for normalization
        #self.maxLinearVelocityFact = 1/self.maxLinearVelocity
        #self.maxAngularVelocityFact = 1/self.maxAngularVelocity
        # Inverse-depth encoding x = d0 / (d0 + d): bounded in (0, 1] with no max
        # sensor range; d0 sets where the resolution concentrates (x = 0.5 at d = d0)
        self.lidarInvDepthScale = getattr(args, 'lidar_d0', 2.0)
        self.goalDistScale = getattr(args, 'goal_d0', 10.0)

        #Pie Slice (chassy for better lidar detection as used with real robots)
        self.hasPieSlice = args.has_pie_slice
        self.pieSlicePoints = []
        self.pieSliceWalls = []
        if self.hasPieSlice:
            self.pieSliceWalls = [ColliderLine(0,0,1,1), ColliderLine(0,0,1,1), ColliderLine(0,0,1,1), ColliderLine(0,0,1,1), ColliderLine(0,0,1,1)]
        self.posSensor = []
        self.robotsPieSliceWalls = []
        self.offsetSensorDist = 0.08
        normFacPieSlice = 1 / self.radius
        self.offsetAnglePieSilce = 2/3 * np.arccos((np.sqrt(2-(self.offsetSensorDist* normFacPieSlice)**2)-(self.offsetSensorDist* normFacPieSlice)) / 2)

        #reward variables
        goalDist = math.sqrt((self.goalX - self.startposX) ** 2 + (self.goalY - self.startposY) ** 2)
        self.initialGoalDist = goalDist
        self.bestGoalDist = goalDist
        # which metric bestGoalDist is measured in -- geodesic and Euclidean
        # distances must never be compared against each other in the ratchet
        self.bestGoalDistIsGeodesic = False
        self.fallbackWarned = False
        # geodesic distance field for reward shaping, assigned by Simulation.reset
        # after stations are randomized (None = Euclidean reward fallback)
        self.flowField = None
        self.stepsAlive = 0
        self.maxSteps = args.steps

        self.walls = walls
        self.circleWalls = circleWalls
        #only use with rectangular targets
        self.collidorStationsWalls = []
        # for pickUp in allStations:
        #     if not station is pickUp:
        #         self.collidorStationsWalls = self.collidorStationsWalls + pickUp.borders
        self.collidorStationsCircles = []

        for pickUp in allStations:
            if not station is pickUp:
                self.collidorStationsCircles.append((pickUp.getPosX(), pickUp.getPosY(), pickUp.getRadius()))

        #direction = (self.getDirectionAngle() + 2 * math.pi) % (2 * math.pi)
        #self.rayCol = FastCollisionRay([self.startposX, self.startposY], self.args.number_of_rays, direction, self.radius, self.fieldOfView)

        self.manuell = args.manually
        if self.manuell:
            print("Robot is in manual mode. Use the arrow keys to control the robot.")
            warnings.warn("Manual mode is not implemented yet. The robot will not move in manual mode.", UserWarning)
            # self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            # self.listener.start()
            # self.linTast = 0
            # self.angTast = 0

    def reset(self, allStations, pos = None, orientation = None, walls = None, goalStation = None):
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
        self.last_positions = CircularBuffer(self.buffersize)
        self.last_positions.add(posX, posY)

        if(orientation != None):
            self.startDirectionX = math.cos(orientation)
            self.startDirectionY = math.sin(orientation)
            self.startOrientation = orientation
        else:
            orientation = self.startOrientation
        directionX = self.startDirectionX
        directionY = self.startDirectionY

        if (goalStation != None):
            self.station = goalStation
        self.goalX = self.station.getPosX()
        self.goalY = self.station.getPosY()

        if walls != None:
            self.walls = walls

        self.collidorStationsCircles = []
        for station in allStations:
            if not station is self.station:
                self.collidorStationsCircles.append((station.getPosX(), station.getPosY(), station.getRadius()))

        self.active = True
        linVel = 0
        angVel = 0
        goalDist = math.sqrt((posX - self.goalX) ** 2 + (posY - self.goalY) ** 2)

        frame = [posX, posY, directionX, directionY, linVel, angVel, self.goalX, self.goalY, goalDist, orientation]

        for _ in range(self.time_frames):
            self.push_frame(frame)

        self.robot_states = []

        self.stepsAlive = 0
        self.distances = []
        self.collisionDistances = []
        self.collisionDistancesRobots = []
        self.lidarHits = []
        self.lidarHitMask = []
        self.netOutput = (0, 0)

        # reward variables
        self.initialGoalDist = goalDist
        self.bestGoalDist = goalDist
        self.bestGoalDistIsGeodesic = False
        self.fallbackWarned = False
        # geodesic distance field for reward shaping, assigned by Simulation.reset
        # after stations are randomized (None = Euclidean reward fallback)
        self.flowField = None
        direction = (self.getDirectionAngle() + 2 * math.pi) % (2 * math.pi)
        self.rayCol = FastCollisionRay([posX, posY], self.args.number_of_rays, direction, self.radius, self.fieldOfView)

        # the wall colliders are static within an episode -- marshal them into numpy
        # arrays once instead of per robot per step in lidarReading
        self._staticColliderLines = self.walls + self.collidorStationsWalls
        self._colStart = np.array([cl.getStart() for cl in self._staticColliderLines], dtype=np.float64).T.reshape(2, -1)
        self._colEnd = np.array([cl.getEnd() for cl in self._staticColliderLines], dtype=np.float64).T.reshape(2, -1)
        segLenSq = (self._colStart[0] - self._colEnd[0]) ** 2 + (self._colStart[1] - self._colEnd[1]) ** 2
        self._segLenSqInv = 1.0 / np.maximum(segLenSq, 1e-12)

        # fill distance for rays that hit nothing: the level's bounding-box diagonal,
        # so lidar hit points stay within the level for visualization while the
        # inverse-depth encoding of a no-hit reading is still ~0
        allCoords = np.concatenate([self._colStart, self._colEnd], axis=1)
        if allCoords.size > 0:
            extents = allCoords.max(axis=1) - allCoords.min(axis=1)
            self.sensorFillRange = float(np.linalg.norm(extents))
        else:
            self.sensorFillRange = 100.0

        if self.hasPieSlice:
            self.posSensor = [posX + self.offsetSensorDist * directionX, posY + self.offsetSensorDist * directionY]
            self.calculatePieSlice((directionX,directionY))
        else:
            self.posSensor = [posX, posY]

    def resetLidar(self, robots):
        if self.hasPieSlice:
            self.robotsPieSliceWalls = []
            for robot in robots:
                if robot is not self:
                    self.robotsPieSliceWalls += robot.getPieSliceWalls()

        for _ in range(self.time_frames):
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

    def push_frame(self, frame):
        """
        adds the given frame to the end of the objects state list.
        If the state has reached its maximum length (defined by timeframes in the constructor),
        it will pop its first element of the list before adding the given frame.

        :param frame: list -
            frame that should be added to the state
        """
        if len(self.state_raw) >= self.time_frames:
            self.state_raw.pop(0)
            self.state_raw.append(frame)
        else:
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
            # linVel, angVel = self.computeNextVelocityContinuous(dt, self.getLinearVelocity(), self.getAngularVelocity(),
            #                                                     tarLinVel, tarAngVel)
            linVel, angVel = self.computeNextVelocity(tarLinVel, tarAngVel)
        else:
            linVel = self.linTast
            angVel = self.angTast

        #print("linVel: ", linVel, "angVel: ", angVel)
        oldDir = self.getDirectionAngle()
        #directionVector = self.directionVectorFromAngle(oldDir)

        direction = (self.getDirectionAngle() + (angVel * dt) + 2 * math.pi) % (2 * math.pi)
        directionVector = self.directionVectorFromAngle(direction)

        deltaPosX = directionVector[0] * linVel * dt  # math.cos(direction) * linVel * dt
        deltaPosY = directionVector[1] * linVel * dt  # math.sin(direction) * linVel * dt
        posX += deltaPosX
        posY += deltaPosY

        self.last_positions.add(posX, posY)

        deltaDir = direction - oldDir

        goalDist = math.sqrt((posX-goalX)**2+(posY-goalY)**2)

        self.stepsAlive += 1

        # raw velocities (m/s, rad/s): the norm getters divide by the physical
        # velocity limits, so storing dt-scaled values would compress the
        # observation channel and couple it to --sim_time_step
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
        angle = self.offsetAnglePieSilce
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

    def lidarReading(self, robots, steps_left, steps):
        """
        Creates a state with a virtual 2D laser scan

        The state consists of n frames (a frame is data from a certain timeframe) with each containing the following data:
        [normalised laser distances of 2D laser scan, angular Deviation between the robots forward axis and its target represented by vector of the length 1,
         normalised distance to the robots target, [normalised linear velocity, normalised angular velocity], current timestep]

        The state is used to train the neural net

        :param robots: list of Robot.Robot objects -
            the positions of the other robots are needed for the laser scan
        :param steps_left: remaining steps of current epoch
        :param steps: number of steps in one epoch
        """

        dir = (self.getDirectionAngle() - (self.fieldOfView / 2)) % (2 * math.pi)

        # static wall arrays are cached in reset(); only the pie-slice walls of other
        # robots change per step (and only exist when has_pie_slice is enabled)
        if self.robotsPieSliceWalls:
            colliderLines = self._staticColliderLines + self.robotsPieSliceWalls
            colLinesStartPoints = np.array([cl.getStart() for cl in colliderLines], dtype=np.float64).T.reshape(2, -1)
            colLinesEndPoints = np.array([cl.getEnd() for cl in colliderLines], dtype=np.float64).T.reshape(2, -1)
            segLenSq = (colLinesStartPoints[0] - colLinesEndPoints[0]) ** 2 + (colLinesStartPoints[1] - colLinesEndPoints[1]) ** 2
            segLenSqInv = 1.0 / np.maximum(segLenSq, 1e-12)
        else:
            colLinesStartPoints = self._colStart
            colLinesEndPoints = self._colEnd
            segLenSqInv = self._segLenSqInv

        collidorCirclePosWithoutRobots = [(wall.getPosX(), wall.getPosY(), wall.getRadius()) for wall in self.circleWalls]
        collidorCirclePosOnlyRobots = []

        for robotA in robots:
            if robotA is not self:
                collidorCirclePosOnlyRobots.append((robotA.getPosX(), robotA.getPosY(), robotA.getRadius()))

        if self.args.collide_other_targets:
            collidorCirclePosWithoutRobots += self.collidorStationsCircles

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

        self.rayCol.new_scan(position, dir)
        distances, lidar_hits = (self.rayCol.lineRayIntersectionPoint(colLinesStartPoints, colLinesEndPoints, circlesPositions, circleR, self.sensorFillRange))

        circleX = [r[0] for r in collidorCircleAllForTerminations]
        circleY = [r[1] for r in collidorCircleAllForTerminations]
        circleR = [r[2] for r in collidorCircleAllForTerminations]
        circlesPositionsAll = np.array([circleX, circleY])
        self.collisionDistances, self.collisionDistancesRobots = self.rayCol.shortestDistanceToCollidors([self.getPosX(), self.getPosY()], colLinesStartPoints, colLinesEndPoints, segLenSqInv, circlesPositionsAll, circleR)

        self.lidarHits = lidar_hits
        # rays that hit nothing report exactly sensorFillRange; their synthetic
        # endpoints must not be rendered as real obstacle points
        self.lidarHitMask = distances < self.sensorFillRange
        self.distances = [distances]

        # calculate distance to goal
        distance_to_goal = np.linalg.norm(
            np.array([self.getPosX(), self.getPosY()]) - np.array([self.station.posX, self.station.posY]))
        distance_to_goal_norm = distance_to_goal / (distance_to_goal + self.goalDistScale)

        # calculate the angle between the robot orientation and the target
        orientation_vec = np.array([self.getDirectionX(), self.getDirectionY()])
        target_vec = np.array([self.station.posX - self.getPosX(), self.station.posY - self.getPosY()])
        cross = orientation_vec[0] * target_vec[1] - orientation_vec[1] * target_vec[0]
        angle = np.arctan2(abs(cross), np.dot(orientation_vec, target_vec))

        # determine the sign of angle
        if -cross > 0:
            angle = -angle

        self.angularDeviation = angle

        anglDeviationV = self.directionVectorFromAngle(self.angularDeviation)

        orientation = [anglDeviationV[0], anglDeviationV[1]]
        self.debugAngle = orientation

        noise = np.random.uniform(low=-0.04, high=0.04, size=distances.shape)
        distances = distances + noise
        # inverse-depth encoding: close obstacles -> 1, free space fades smoothly to 0;
        # clamping at 0 keeps noisy near-zero readings inside (0, 1]
        laser = self.lidarInvDepthScale / (self.lidarInvDepthScale + np.maximum(distances, 0.0))

        robot_state = [laser,                                                       # 2D Laser Scan
                       np.asarray(orientation),                                     # Orientation to Goal
                       np.expand_dims(np.asarray(distance_to_goal_norm), axis=0),   # Distance to Goal
                       np.array([self.getLinearVelocityNorm(),                      # Linear Velocity
                                 self.getAngularVelocityNorm()]),                   # Angular Velocity
                       ]

        if len(self.robot_states) >= self.time_frames:
            self.robot_states.pop(0)
            self.robot_states.append(robot_state)
        else:
            self.robot_states.append(robot_state)

    def get_robot_states(self, reverse=True):
        """
        Returns the Robot States.
        :param reverse: (reverse = false : current state in last place and the oldest at Index 0)
        :return: list of robot states
        """
        # shallow copy is enough: every frame is a freshly built list of arrays
        # (see lidarReading) and consumers never mutate them; the previous deepcopy
        # of 4x1081 floats per robot per step was a simulation hotspot
        robot_states = list(self.robot_states)
        if reverse:
             robot_states.reverse()
        return robot_states


    def discretize(self, value, bin_size):
        '''
        Discretizes a value to a given bin size
        :param value: float - value to be discretized
        :param bin_size: float - size of the bin
        '''
        value = np.clip(value, -1, 1)
        bin_value = round(value / bin_size) * bin_size
        return bin_value

    def computeNextVelocity(self, tarLinVel, tarAngVel):
        """
        :param tarLinVel: float/ int - target linear velocity
        :param tarAngVel: float/ int - target angular velocity
        :return: tuple (float - linear velocity, float - angular velocity)
        """

        self.netOutput = (tarLinVel, tarAngVel)

        assert tarLinVel >= -1 and tarLinVel <= 1, "velocity received from neural net is out of bounds. Fix your code!"
        assert tarAngVel >= -1 and tarAngVel <= 1, "velocity received from neural net is out of bounds. Fix your code!"

        # Map the Velocities to a range of 40 values between -1 and 1
        norm_linvel = self.minLinearVelocity + ((tarLinVel + 1) / 2) * (self.maxLinearVelocity - self.minLinearVelocity)
        norm_angvel = self.minAngularVelocity + ((tarAngVel + 1) / 2) * (self.maxAngularVelocity - self.minAngularVelocity)
        lin_vel = self.discretize(norm_linvel, 0.005)
        ang_vel = self.discretize(norm_angvel, 0.0555)

        return np.around(lin_vel, decimals=3), np.around(ang_vel, decimals=3)

    def computeNextVelocityContinuous(self, dt, linVel, angVel, tarLinVel, tarAngVel):
        """
        :param dt: float - passed (simulated) time since last call
        :param linVel: float - current linear velocity
        :param angVel: float - current angular velocity
        :param tarLinVel: float/ int - target linear velocity
        :param tarAngVel: float/ int - target angular velocity
        :return: tuple (float - linear velocity, float - angular velocity)
        """

        self.netOutput = (tarLinVel, tarAngVel)

        # Check if tarLinVel and tarAngVel are within bounds
        if tarLinVel < -1 or tarLinVel > 1 or tarAngVel < -1 or tarAngVel > 1:
            raise Exception("velocity received from neural net is out of bounds. Fix your code!")

        # Map the net output range of -1 to 1 onto the velocity ranges of the robot
        tarAngVel = tarAngVel * ((self.maxAngularVelocity - self.minAngularVelocity) * 0.5) + (
                    (self.minAngularVelocity + self.maxAngularVelocity) * 0.5)
        tarLinVel = tarLinVel * ((self.maxLinearVelocity - self.minLinearVelocity) * 0.5) + (
                    (self.minLinearVelocity + self.maxLinearVelocity) * 0.5)

        # Check boundaries
        tarLinVel = max(self.minLinearVelocity, min(tarLinVel, self.maxLinearVelocity))
        tarAngVel = max(self.minAngularVelocity, min(tarAngVel, self.maxAngularVelocity))

        return np.around(tarLinVel, decimals=3), np.around(tarAngVel, decimals=3)

        # # beschleunigen
        # if linVel < tarLinVel:
        #     linVel += self.maxLinearAcceleration * dt  # v(t) = v(t-1) + a * dt
        #     if linVel > self.maxLinearVelocity:
        #         linVel = self.maxLinearVelocity
        #
        # # bremsen
        # elif linVel > tarLinVel:
        #     linVel += self.minLinearAcceleration * dt
        #     if linVel < self.minLinearVelocity:
        #         linVel = self.minLinearVelocity
        #
        # # nach links drehen
        # if angVel < tarAngVel:
        #     angVel += self.maxAngularAcceleration * dt
        #     if angVel > self.maxAngularVelocity:
        #         angVel = self.maxAngularVelocity
        #
        # # nach rechts drehen
        # elif angVel > tarAngVel:
        #     angVel += self.minAngularAcceleration * dt
        #     if angVel < self.minAngularVelocity:
        #         angVel = self.minAngularVelocity
        #
        # return linVel, angVel
        # maybe ändern

    def directionVectorFromAngle(self, direction):
        """
        calculates a vector of length 1 based on the direction in radians
        :param direction: float direction in radians
        :return: (float, float) direction as a vector
        """
        angX = math.cos(direction)
        angY = math.sin(direction)
        return [angX, angY]

    def collideWithTargetStationCircular(self):
        """
        :return: Boolean
        """
        station = self.station
        distance2StationCenter = math.sqrt((station.getPosX() - self.getPosX())**2 +
                                           (station.getPosY() - self.getPosY())**2) + self.radius
        return (distance2StationCenter < (station.radius * 1.2))

    def collideWithTargetStationRectengular(self):
        """
        :return: Boolean
        """
        station = self.station

        #Check if close to target
        if self.getPosX()+self.radius > station.getPosX():
            if self.getPosX()-self.radius < station.getPosX()+station.getWidth():
                if self.getPosY() + self.radius > station.getPosY():
                    if self.getPosY() - self.radius < station.getPosY() + station.getLength():
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

    def getPosX(self):
        return self.state_raw[self.time_frames - 1][0]

    def getPosY(self):
        return self.state_raw[self.time_frames - 1][1]

    def getLastPosX(self):
        return self.state_raw[self.time_frames - 2][0]

    def getLastPosY(self):
        return self.state_raw[self.time_frames - 2][1]

    def getDirectionX(self):
        return self.state_raw[self.time_frames - 1][2]

    def getDirectionY(self):
        return self.state_raw[self.time_frames - 1][3]

    def getLastDirectionX(self):
        return self.state_raw[self.time_frames - 2][2]

    def getLastDirectionY(self):
        return self.state_raw[self.time_frames - 2][3]

    def getLinearVelocity(self):
        return np.around(self.state_raw[self.time_frames - 1][4], decimals=5)

    def getAngularVelocity(self):
        return np.around(self.state_raw[self.time_frames - 1][5], decimals=5)

    def getLinearVelocityNorm(self):
        return np.around((self.getLinearVelocity() - ((self.minLinearVelocity + self.maxLinearVelocity) * 0.5)) / (
                    (self.maxLinearVelocity - self.minLinearVelocity) * 0.5), decimals=5)

    def getAngularVelocityNorm(self):
        return np.around((self.getAngularVelocity() - ((self.minAngularVelocity + self.maxAngularVelocity) * 0.5)) / (
                (self.maxAngularVelocity - self.minAngularVelocity) * 0.5), decimals=5)

    def getGoalX(self):
        return self.state_raw[self.time_frames - 1][6]

    def getGoalY(self):
        return self.state_raw[self.time_frames - 1][7]

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
            return self.state_raw[self.time_frames - 1][9]
        return self.state_raw[self.time_frames - 2][9]

    def setGoal(self, goal):
        goalX, goalY = goal
        self.goalX = goalX
        self.goalY = goalY

    def on_press(self, key):
        if key.char == 'w':
            self.linTast = 1.49999
        if key.char == 'a':
            self.angTast = -0.5999
        if key.char == 's':
            self.linTast = 0
        if key.char == 'd':
            self.angTast = 0.5 #0.5999
        if key.char == 'c':
            self.angTast = 0

    def on_release(self, key):
        if key.char == 'w' or key.char == 's':
            self.linTast = 0
        if key.char == 'a' or key.char == 'd':
            self.angTast = 0


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
        :param numberOfRays: number of rays
        :param startAngle: float - the angle of the first ray (orientation - fov/2)
        :param radius: the radius of the robot itself
        :param fov: field of view in radians, covered inclusively from the first to the last ray
        """
        self.rayOrigin = rayOrigin
        self.numberOfRays = numberOfRays
        # inclusive spacing: 1081 rays over 270 degrees -> exactly 0.25 degrees per step
        self.stepSize = fov / max(numberOfRays - 1, 1)
        self._raySteps = self.stepSize * np.arange(numberOfRays)
        self.new_scan(rayOrigin, startAngle)

    def new_scan(self, rayOrigin, startAngle):
        self.rayOrigin = rayOrigin
        steps = startAngle + self._raySteps
        self.rayDirX = np.cos(steps)
        self.rayDirY = np.sin(steps)

    def lineRayIntersectionPoint(self, points1, points2, pointsRobots, radius, maxRange):
        """
        Intersects all rays with all wall segments and robot circles. Walls are solid from
        both sides (the previous normal-based back-face culling made walls one-way
        transparent depending on the drawing direction in the level file).

        :param points1: starting points of collision lines - [[x1..xn], [y1..yn]]
        :param points2: ending points of collision lines - [[x1..xn], [y1..yn]]
        :param pointsRobots: positions of all robot colliders - np.array [[x0..xn], [y0..yn]]
        :param radius: radii of the robot colliders
        :param maxRange: sensor range; rays that hit nothing report this distance
        :return: [distances per ray, collision points per ray]
        """

        ox = self.rayOrigin[0]
        oy = self.rayOrigin[1]
        dx = self.rayDirX  # (R,) unit directions
        dy = self.rayDirY

        sx = points1[0] - points2[0]  # (W,) segment start - end
        sy = points1[1] - points2[1]
        rx = ox - points1[0]  # (W,) origin - segment start
        ry = oy - points1[1]

        # line-line intersection: ray = origin + t1 * dir, segment = start + t2 * (end - start)
        # det only depends on ray dir and segment dir; the t1 numerator only on origin and segment
        det = np.outer(dy, sx) - np.outer(dx, sy)        # (R, W)
        t1_num = rx * sy - ry * sx                       # (W,)
        t2_num = np.outer(dx, ry) - np.outer(dy, rx)     # (R, W)

        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / det
            t1 = t1_num * inv
            t2 = t2_num * inv

        # small tolerance on the segment ends so rays cannot slip through the seam
        # where two walls share a corner
        eps = 1e-9
        valid = (t2 >= -eps) & (t2 <= 1 + eps) & (t1 >= 0) & np.isfinite(t1)
        t1 = np.where(valid, t1, maxRange)
        t1NearestHit = np.amin(t1, axis=1) if t1.shape[1] > 0 else np.full(len(dx), float(maxRange))
        t1NearestHit = np.minimum(t1NearestHit, maxRange)

        # circle colliders (other robots): ray dirs are unit vectors, so the quadratic
        # has a = 1 and the roots are directly comparable with the segment t's
        if len(pointsRobots[0]) > 0:
            qx = ox - np.asarray(pointsRobots[0])[:, None]   # (C, 1)
            qy = oy - np.asarray(pointsRobots[1])[:, None]
            r = np.asarray(radius, dtype=np.float64)[:, None]

            b = 2.0 * (dx[None, :] * qx + dy[None, :] * qy)  # (C, R)
            c = qx * qx + qy * qy - r * r                    # (C, 1)
            disc = b * b - 4.0 * c

            with np.errstate(invalid='ignore'):
                sqrt_disc = np.sqrt(disc)
            tc1 = (-b + sqrt_disc) * 0.5
            tc2 = (-b - sqrt_disc) * 0.5
            tc1 = np.where((disc > 0) & (tc1 >= 0), tc1, maxRange)
            tc2 = np.where((disc > 0) & (tc2 >= 0), tc2, maxRange)
            nearestCircleHit = np.amin(np.minimum(tc1, tc2), axis=0)

            t1NearestHit = np.minimum(t1NearestHit, nearestCircleHit)

        collisionPoints = np.stack([ox + t1NearestHit * dx, oy + t1NearestHit * dy], axis=1)  # (R, 2)

        return [t1NearestHit, collisionPoints]

    def shortestDistanceToCollidors(self, pos, segStart, segEnd, segLenSqInv, circles, radii):
        """
        Shortest distance from pos to each wall segment and circle collider.

        :param segStart: segment start points - [[x1..xn], [y1..yn]]
        :param segEnd: segment end points - [[x1..xn], [y1..yn]]
        :param segLenSqInv: precomputed 1 / squared segment lengths
        """
        x1, y1 = pos  # originX and originY
        x2, y2 = segStart
        x3, y3 = segEnd

        t = np.clip(((x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)) * segLenSqInv, 0, 1)
        dist = np.sqrt((x1 - (x2 + t * (x3 - x2))) ** 2 + (y1 - (y2 + t * (y3 - y2))) ** 2)

        if len(circles[0]) > 0:
            x4, y4 = np.array(circles) # circleXArray and circleYArray
            distCircles = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2) - radii
            dist = np.concatenate((dist, distCircles))
        else:
            distCircles = []

        return dist, distCircles

