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
        # Maximum distance in laserscan is 20 meters
        self.maxDistFact = 1/10
        #self.maxDistSim = 22

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
        self.netOutput = (0, 0)

        # reward variables
        self.initialGoalDist = goalDist
        direction = (self.getDirectionAngle() + 2 * math.pi) % (2 * math.pi)
        self.rayCol = FastCollisionRay([posX, posY], self.args.number_of_rays, direction, self.radius, self.fieldOfView)

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

        # TODO Question: Should the linVel and angVel be saved with or without dt ?
        frame = [posX, posY, directionVector[0], directionVector[1], linVel * dt, angVel * dt, goalX, goalY, goalDist, direction]

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

        colliderLines = self.walls + self.collidorStationsWalls + self.robotsPieSliceWalls
        collidorCirclePosWithoutRobots = [(wall.getPosX(), wall.getPosY(), wall.getRadius()) for wall in self.circleWalls]
        collidorCirclePosOnlyRobots = []

        for robotA in robots:
            if robotA is not self:
                collidorCirclePosOnlyRobots.append((robotA.getPosX(), robotA.getPosY(), robotA.getRadius()))

        if self.args.collide_other_targets:
            collidorCirclePosWithoutRobots += self.collidorStationsCircles

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

        self.rayCol.new_scan(position, dir)
        distances, lidar_hits = (self.rayCol.lineRayIntersectionPoint(colLinesStartPoints, colLinesEndPoints, normals, circlesPositions, circleR, self.offsetSensorDist))

        circleX = [r[0] for r in collidorCircleAllForTerminations]
        circleY = [r[1] for r in collidorCircleAllForTerminations]
        circleR = [r[2] for r in collidorCircleAllForTerminations]
        circlesPositionsAll = np.array([circleX, circleY])
        self.collisionDistances, self.collisionDistancesRobots = self.rayCol.shortestDistanceToCollidors([self.getPosX(), self.getPosY()], colliderLines, circlesPositionsAll, circleR)

        self.lidarHits = lidar_hits
        self.distances = [distances]

        # calculate distance to goal
        distance_to_goal = np.linalg.norm(
            np.array([self.getPosX(), self.getPosY()]) - np.array([self.station.posX, self.station.posY]))
        distance_to_goal_norm = distance_to_goal * self.maxDistFact

        # calculate the angle between the robot orientation and the target
        orientation_vec = np.array([self.getDirectionX(), self.getDirectionY()])
        target_vec = np.array([self.station.posX - self.getPosX(), self.station.posY - self.getPosY()])
        angle = np.arctan2(np.linalg.norm(np.cross(orientation_vec, target_vec)), np.dot(orientation_vec, target_vec))

        # determine the sign of angle
        if np.cross(target_vec, orientation_vec) > 0:
            angle = -angle

        self.angularDeviation = angle

        anglDeviationV = self.directionVectorFromAngle(self.angularDeviation)

        orientation = [anglDeviationV[0], anglDeviationV[1]]
        self.debugAngle = orientation

        noise = np.random.uniform(low=-0.04, high=0.04, size=distances.shape)
        distances = distances + noise
        laser = distances * self.maxDistFact
        # laser = np.where(laser > 1, 1, laser)

        current_timestep = (steps - steps_left) / steps

        robot_state = [laser,                                                       # 2D Laser Scan
                       np.asarray(orientation),                                     # Orientation to Goal
                       np.expand_dims(np.asarray(distance_to_goal_norm), axis=0),   # Distance to Goal
                       np.array([self.getLinearVelocityNorm(),                      # Linear Velocity
                                 self.getAngularVelocityNorm()]),                   # Angular Velocity
                       current_timestep                                             # Current Timestep
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
        # tmp_state = copy.deepcopy(self.stateLidar)
        # zipstate = list(zip(*tmp_state))
        # states = []
        # for state in zipstate:
        #     states.append(np.array(state))
        # return states
        robot_states = copy.deepcopy(self.robot_states)
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
        :param rayCount: number of rays
        :param startAngle: float - usually the orientation of the robot added by some amount to prevent edge problems
        during the feature detection in the convolutional layers of the neural net.
        :param radius: the radius of the robot itself
        """
        self.rayOrigin = rayOrigin
        self.numberOfRays = numberOfRays

        self.stepSize = fov / numberOfRays
        self.rayOrigin = rayOrigin
        steps = np.arange(startAngle, (startAngle + self.stepSize * self.numberOfRays) - (self.stepSize / 2), self.stepSize)
        self.rayDirX = np.cos(steps)
        self.rayDirY = np.sin(steps)
        # self.rayOrigin = rayOrigin
        # steps = np.array([startAngle+ i * self.stepSize for i in range(self.numberOfRays)])
        # self.rayDirX = np.array([math.cos(step) for step in steps])
        # self.rayDirY = np.array([math.sin(step) for step in steps])
        # self.ownRadius = radius

    def new_scan(self, rayOrigin, startAngle):
        self.rayOrigin = rayOrigin
        steps = np.arange(startAngle, (startAngle + self.stepSize * self.numberOfRays) - (self.stepSize / 2), self.stepSize)
        self.rayDirX = np.cos(steps)
        self.rayDirY = np.sin(steps)
        # self.rayOrigin = rayOrigin
        # steps = np.array([startAngle+ i * self.stepSize for i in range(self.numberOfRays)])
        # self.rayDirX = np.array([math.cos(step) for step in steps])
        # self.rayDirY = np.array([math.sin(step) for step in steps])

    def lineRayIntersectionPoint(self, points1, points2, normals, pointsRobots, radius, sensorOffset):
        """

        :param points1: List of starting points of collision lines - points[[x1,x2,x3...xn],[y1,y2,y3...yn]]
        :param points2: List of ending points of collision lines - points[[x1,x2,x3...xn],[y1,y2,y3...yn]]
        :param pointsRobots: List of all robots positions which can be a collider as an np.array[[x0, x2, x3..xn], [y0, y2, y3..yn]]
        :return:
        """

        x1 = self.rayOrigin[0] #originX
        y1 = self.rayOrigin[1] #originY
        x2Vswap = np.tile(self.rayDirX, (len(points1[0]), 1))
        x2V = np.swapaxes(x2Vswap, 0, 1)  # directionX
        y2Vswap = np.tile(self.rayDirY, (len(points1[0]), 1))
        y2V = np.swapaxes(y2Vswap, 0, 1) #directionY
        x2 = x2V+x1
        y2 = y2V+y1

        nX = np.tile(normals[0], (len(self.rayDirX), 1))
        nY = np.tile(normals[1], (len(self.rayDirX), 1))

        skalarProd = nX * x2V + nY * y2V

        x3 = np.tile(points1[0], (len(self.rayDirX), 1)) #lineStartXArray
        y3 = np.tile(points1[1], (len(self.rayDirX), 1)) #lineStartYArray
        x4 = np.tile(points2[0], (len(self.rayDirX), 1)) #lineEndXArray
        y4 = np.tile(points2[1], (len(self.rayDirX), 1)) #lineEndYArray

        # Ersten den denominator berechnen, da er für t1 und t2 gleich ist.
        denominator = np.where(skalarProd<0, 1.0 / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)), -1)

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
            qX = np.tile(pointsRobots[0], (len(collisionPoints[0]), 1))
            qY = np.tile(pointsRobots[1], (len(collisionPoints[0]), 1))

            radii = np.tile(radius, (len(collisionPoints[0]), 1))
            qX = np.swapaxes(qX,0,1)
            qY = np.swapaxes(qY,0,1)
            radii = np.swapaxes(radii,0,1)

            colX = np.tile(collisionPoints[0], (len(pointsRobots[0]), 1))
            colY = np.tile(collisionPoints[1], (len(pointsRobots[0]), 1))

            vX = colX - x1
            vY = colY - y1
            # normalize vector to a length of 1, so that the t parameters of the line-line intersection can be compared with the t's from cirle-line intersection
            vLengthFact = 1/np.sqrt(vX**2 + vY**2) #again division costs too much for us poor students so we like to use multiplications with a factor
            vX= vX * vLengthFact
            vY= vY * vLengthFact

            # a,b und c als Array zum Berechnen der Diskriminanten
            a = vX * vX + vY * vY # array voll skalarere Werte
            b = 2 * (vX * (x1 - qX) + vY * (y1 - qY))
            c = (x1**2 + y1**2) + (qX**2 + qY**2) - (2 * (x1 * qX + y1*qY)) - radii**2

            disc = b**2 - 4 * a * c
            denominator = 1 / (2 * a)

            # check if discriminat is negative ==> no collision
            indices = np.where(disc > 0)

            tc1 = np.full(disc.shape, -1.0)
            tc1[indices] = (-b[indices] + np.sqrt(disc[indices])) * denominator[indices]

            tc2 = np.full(disc.shape, -1.0)
            tc2[indices] = (-b[indices] - np.sqrt(disc[indices])) * denominator[indices]

            # throws RunTimewarnings
            #tc1 = np.where((disc > 0), ((-b + np.sqrt(disc)) * denominator), -1)
            #tc2 = np.where((disc > 0), ((-b - np.sqrt(disc)) * denominator), -1)

            tc1 = np.where((tc1 >= 0), tc1, 2048)
            tc2 = np.where((tc2 >= 0), tc2, 2048)

            smallestTOfCircle = np.where((tc1<tc2), tc1, tc2)
            smallestTOfCircle = np.amin(smallestTOfCircle, axis=0)

            t1NearestHit = np.where(((smallestTOfCircle<2048) & (smallestTOfCircle<t1NearestHit)), smallestTOfCircle, t1NearestHit)

        collisionPoints = np.array([x1+t1NearestHit* x2V[:,0], y1+t1NearestHit* y2V[:,0]]) #[:,0] returns the first column # Aufbau nach [x0,x1…x2], [y0,y1…yn]]

        # if sensorOffset<0: t1NearestHit = t1NearestHit - (self.ownRadius-sensorOffset) #CHRISTIANS FRAGEN
        collisionPoints = np.swapaxes(collisionPoints, 0, 1) # für Rückgabe in x,y-Paaren

        return [t1NearestHit, collisionPoints]

    def shortestDistanceToCollidors(self, pos, lineSegments, circles, radii):
        x1, y1 = pos  # originX and originY

        x2, y2 = np.array([segment.getStart() for segment in lineSegments]).T  # lineStartXArray and lineStartYArray
        x3, y3 = np.array([segment.getEnd() for segment in lineSegments]).T  # lineEndXArray and lineEndYArray

        t = np.clip(((x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)) / ((x2 - x3) ** 2 + (y2 - y3) ** 2), 0, 1)
        dist = np.sqrt((x1 - (x2 + t * (x3 - x2))) ** 2 + (y1 - (y2 + t * (y3 - y2))) ** 2)

        if len(circles[0]) > 0:
            x4, y4 = np.array(circles) # circleXArray and circleYArray
            distCircles = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2) - radii
            dist = np.concatenate((dist, distCircles))
        else:
            distCircles = []

        return dist, distCircles

