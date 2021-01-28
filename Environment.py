import math
import SimulationWithoutUI
import time
import numpy as np
from old.PlotterWindow import PlotterWindow
import time

class Environment:
    def __init__(self, steps, args, timeframes, id):
        self.args = args
        self.steps = steps
        self.steps_left = steps
        #self.simulation = Simulation.Simulation(app, args, timeframes)
        self.simulation = SimulationWithoutUI.Simulation(args, timeframes)
        self.total_reward = 0.0
        self.done = False
        self.shape = np.asarray([0]).shape
        self.id = id     # Multiprocessing Wiedererkennung zum Zuordnen der Rückgabewerte


    def get_observation(self, i):
        #TODO den richtigen Roboter aus der Liste wählen mit parameter i --> getRobot(i)
        if self.args.mode == 'global':
            return np.asarray(self.simulation.robots[i].state)  # Pos, Geschwindigkeit, Zielposition
        elif self.args.mode == 'sonar':
            return np.asarray(self.simulation.robots[i].stateSonar)  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel

    @staticmethod
    def get_actions():
        return 2#[0, 1, 2, 3]  # Links, Rechts, Oben, Unten

   # @staticmethod
    def get_velocity(self, action):
        if(action == None):
            return (None, None)
        actualLinVel = self.simulation.robot.getLinearVelocity()
        actualAngVel = self.simulation.robot.getAngularVelocity()

        tarLinVel = 0 #actualLinVel
        tarAngVel = 0 #actualAngVel

        # Links drehen mit vorheriger Linear Velocity
        if action == 0:
            tarAngVel = self.simulation.robot.minAngularVelocity

        # Rechts drehen mit vorheriger Linear Velocity
        if action == 1:
            tarAngVel = self.simulation.robot.maxAngularVelocity

        # Geradeaus fahren mit vorheriger Linear Velocity
        # if action == 2:
        #     tarAngVel = 0

        # Beschleunigen auf maximale Linear Velocity, drehen mit vorheriger Angular Velocity
        if action == 2:
            tarLinVel = self.simulation.robot.maxLinearVelocity

        # stehen bleiben, Angular Velocity wie vorher
        if action == 3:
            tarLinVel = self.simulation.robot.minLinearVelocity

        return tarLinVel, tarAngVel

    def is_done(self):
        robotsDone = True
        for robot in self.simulation.robots:
            if robot.isActive():
                robotsDone = False
                break
        return self.steps_left <= 0 or robotsDone

    def step(self, actions):
        # print(actions)
        time1 = time.time()
        self.steps_left -= 1

        ######## Update der Simulation #######
        robotsTarVels = []
        for action in actions:
            tarLinVel, tarAngVel = action #self.get_velocity(action)
            robotsTarVels.append((tarLinVel, tarAngVel))

        robotsTermination = self.simulation.update(robotsTarVels, self.steps_left)
        robotsDataCurrentFrame = []
        for i, termination in enumerate(robotsTermination):
            if termination != (None, None, None):
                robotsDataCurrentFrame.append(self.extractRobotData(i, robotsTermination[i]))
            else:
                robotsDataCurrentFrame.append((None, None, None))
        time2 = time.time()
        # print(self.steps_left, self.id, time1, time2)
        #print("Env", time2-time1)
        #print(self.id, self.steps_left)
        return (robotsDataCurrentFrame, self.id)



    def extractRobotData(self, i, terminantions):

        robot = self.simulation.robots[i]
        collision, reachedPickup, runOutOfTime = terminantions

        ############ State Robot i ############
        next_state = self.get_observation(i)  # TODO state von Robot i bekommen
        next_state = np.expand_dims(next_state, axis=0)


        ############ Euklidsche Distanz und Orientierung ##############

        radius = robot.radius
        goal_pose_old_x = robot.getGoalX()
        goal_pose_old_y = robot.getGoalY()
        robot_pose_old_x = robot.getLastPosX()
        robot_pose_old_y = robot.getLastPosY()
        robot_orientation_old = robot.getDirectionAngle(last=True)
        #TODO Goal des Robots nutzen
        orientation_goal_old = math.atan2(
            (goal_pose_old_y + (self.simulation.getGoalLength() / 2)) - (robot_pose_old_y),
            (goal_pose_old_x + (self.simulation.getGoalWidth() / 2)) - (robot_pose_old_x))
        if orientation_goal_old < 0:
            orientation_goal_old += (2 * math.pi)

        # einzeln Abstand berechnen
        robot_pose_current_x = robot.getPosX()
        robot_pose_current_y = robot.getPosY()
        robot_orientation_new = robot.getDirectionAngle()
        orientation_goal_new = math.atan2(
            (goal_pose_old_y + (self.simulation.getGoalLength() / 2)) - (robot_pose_current_y),
            (goal_pose_old_x + (self.simulation.getGoalWidth() / 2)) - (robot_pose_current_x))
        if orientation_goal_new < 0:
            orientation_goal_new += (2 * math.pi)
        # TODO Goal des Robots nutzen
        distance_old = math.sqrt((robot_pose_old_x - (goal_pose_old_x + (self.simulation.getGoalWidth() / 2))) ** 2 +
                                 (robot_pose_old_y - (goal_pose_old_y + (self.simulation.getGoalLength() / 2))) ** 2)
        distance_new = math.sqrt(
            (robot_pose_current_x - (goal_pose_old_x + (self.simulation.getGoalWidth() / 2))) ** 2 +
            (robot_pose_current_y - (goal_pose_old_y + (self.simulation.getGoalLength() / 2))) ** 2)
        delta_dist = distance_old - distance_new

        ########### REWARD CALCULATION ################
        # reward = self.createReward01(robot, delta_dist, robot_orientation_new,orientation_goal_new, collision, reachedPickup)
        # reward = self.createReward02(robot, delta_dist, robot_orientation_old, orientation_goal_old, robot_orientation_new,
        #                              orientation_goal_new, collision, reachedPickup)
        reward = self.createReward03(robot, delta_dist,distance_new, robot_orientation_old, orientation_goal_old, robot_orientation_new,
                                     orientation_goal_new, collision, reachedPickup)
        return (next_state, reward / 10, not robot.isActive(), reachedPickup)


    def createReward01(self, robot, delta_dist, robot_orientation, orientation_goal_new, outOfArea, reachedPickup):
        reward = delta_dist / 5
        # print("Delta Dist: " + str(delta_dist))

        if delta_dist > 0.0:
            reward += reward  # * 0.01
        if delta_dist == 0.0:
            reward += -0.5
        if delta_dist < 0.0:
            reward += reward  # * 0.5 # * 0.001

        anglDeviation = math.fabs(robot_orientation - orientation_goal_new)
        reward += (anglDeviation * -1 + math.pi/3) * 2
        if anglDeviation < 0.2 and delta_dist > 0:
            reward = reward * 2

        # print("angDev: ", anglDeviation, "  Reward-Anteil: ",  (anglDeviation * -1 + 0.35) * 2)

        # if math.fabs(robot_orientation - orientation_goal_new) < 0.001:  # 0.05 0.3
        #     #if(distance_old - distance_new) > 0:
        #     reward += 1.0
        #
        # if math.fabs(robot_orientation - orientation_goal_new) < 0.05:  # 0.5
        #     #if(distance_old - distance_new) > 0:
        #     reward += 0.01

        # else:
        #     reward += -0.1
        # if distance_old > distance_new:
        #     reward += 2
        # if distance_old < distance_new:
        #     reward = -1
        # if distance_old == distance_new:
        #     reward += -1
        if robot.isInCircleOfGoal(500):
            reward += 0.001
        if robot.isInCircleOfGoal(200):
            reward += 0.002
        if robot.isInCircleOfGoal(100):
            reward += 0.003
        if outOfArea:
            reward += -2.0

        if reachedPickup:
            reward += 2.0



        # reward -= (self.steps - self.steps_left) / self.steps
        # if self.steps_left <= 0:
        #    reward += -1.0

        # reward = factor * distance        # evtl. reward gewichten
        return reward


    def createReward02(self, robot, delta_dist, robot_orientation_old,orientation_goal_old, robot_orientation_new, orientation_goal_new, outOfArea, reachedPickup):

        reward = delta_dist

        anglDeviation_old = math.fabs(robot_orientation_old - orientation_goal_old)
        if anglDeviation_old > math.pi:
            subtractor = 2 * (anglDeviation_old - math.pi)
            anglDeviation_old = anglDeviation_old-subtractor

        anglDeviation_new = math.fabs(robot_orientation_new - orientation_goal_new)
        if anglDeviation_new > math.pi:
            subtractor = 2* (anglDeviation_new - math.pi)
            anglDeviation_new = anglDeviation_new - subtractor

        if anglDeviation_new < anglDeviation_old:
            reward += ((math.pi - anglDeviation_new)) /math.pi /6
        else:
            reward -= ((anglDeviation_new)) / math.pi /6

        if(anglDeviation_new < 1):
            reward += ((math.pi - anglDeviation_new)) /math.pi /6

        reward = reward/2

        # print(reward, anglDeviation_new * 180/math.pi, (math.pi-anglDeviation_new)/math.pi /4)
        # reward+= ((anglDeviation_old - anglDeviation_new)/math.pi)/8
        # print(reward)
        # print(anglDeviation_old*180/math.pi,anglDeviation_new*180/math.pi, ((anglDeviation_old - anglDeviation_new)*180/math.pi) )

        if outOfArea:
            reward += -3.0

        if reachedPickup:
            reward += 3.0


        return reward

    def createReward03(self, robot, delta_dist,distance_new, robot_orientation_old, orientation_goal_old, robot_orientation_new,
                       orientation_goal_new, collision, reachedPickup):
        reward = 0
        deltaDist = delta_dist #bei max Vel von 0.7 und einem 0.1 Timestep ist die max Dist 0.07m --> 7cm
        angularDeviation = (abs(robot.angularDeviation / math.pi) *-1) + 0.5


        reward = deltaDist + (angularDeviation/25)

        if collision:
            reward += -1.0

        if reachedPickup:
            reward += 1.0

        # timeInfluence = 0.05
        # bonusTime = 200
        #
        # timePenalty = 0
        # if(self.steps-self.steps_left > bonusTime):
        #     timePenalty = (1/self.steps) * (self.steps+bonusTime - self.steps_left) * timeInfluence


        # print('Orient. Diff: ', angularDeviation, '   deltaDist: ', deltaDist, '   reward: ', reward)


        return reward

    def reset(self):

        for robot in self.simulation.robots:
            robot.reset()
        for robot in self.simulation.robots:
            robot.resetSonar(self.simulation.robots)

        self.steps_left = self.steps
        self.total_reward = 0.0
        self.done = False
