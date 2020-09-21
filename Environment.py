import math
import Simulation
import time
import numpy as np
from old.PlotterWindow import PlotterWindow


class Environment:
    def __init__(self, app, steps, args, timeframes):
        self.steps = steps
        self.steps_left = steps
        self.simulation = Simulation.Simulation(app, args, timeframes)
        self.total_reward = 0.0
        self.done = False
        self.shape = np.asarray([0]).shape

        self.plotterWindow = PlotterWindow(app)

    def get_observation(self):
        return np.asarray(self.simulation.robot.state)  # Pos, Geschwindigkeit, Zielposition

    @staticmethod
    def get_actions():
        return [0, 1, 2, 3, 4]  # Links, Rechts, Oben, Unten

   # @staticmethod
    def get_velocity(self, action):
        actualLinVel = self.simulation.robot.getLinearVelocity()
        actualAngVel = self.simulation.robot.getAngularVelocity()

        tarLinVel = actualLinVel
        tarAngVel = actualAngVel

        # Links drehen mit vorheriger Linear Velocity
        if action == 0:
            tarAngVel = self.simulation.robot.minAngularVelocity

        # Rechts drehen mit vorheriger Linear Velocity
        if action == 1:
            tarAngVel = self.simulation.robot.maxAngularVelocity

        # Geradeaus fahren mit vorheriger Linear Velocity
        if action == 2:
            tarAngVel = 0

        # Beschleunigen auf maximale Linear Velocity, drehen mit vorheriger Angular Velocity
        if action == 3:
            tarLinVel = self.simulation.robot.maxLinearVelocity

        # stehen bleiben, Angular Velocity wie vorher
        if action == 4:
            tarLinVel = self.simulation.robot.minLinearVelocity

        return tarLinVel, tarAngVel

    def is_done(self):
        return self.steps_left <= 0 or self.done

    def step(self, action):

        self.steps_left -= 1

        tarLinVel, tarAngVel = self.get_velocity(action)

        ######## Update der Simulation #######
        radius = self.simulation.getRobot().radius
        goal_pose_old_x = self.simulation.robot.getGoalX()
        goal_pose_old_y = self.simulation.robot.getGoalY()
        robot_pose_old_x = self.simulation.getRobot().getPosX() + radius
        robot_pose_old_y = self.simulation.getRobot().getPosY() + radius

        outOfArea, reachedPickup, reachedDelivery = self.simulation.update(tarLinVel, tarAngVel)
        #######################################

        next_state = self.get_observation()
        next_state = np.expand_dims(next_state, axis=0)

        ############ Euklidsche Distanz und Orientierung ##############

        # einzeln Abstand berechnen
        robot_pose_current_x = self.simulation.getRobot().getPosX() + radius
        robot_pose_current_y = self.simulation.getRobot().getPosY() + radius

        robot_orientation = self.simulation.getRobot().getDirection()
        orientation_goal_new = math.atan2((goal_pose_old_y + (self.simulation.getGoalLength() / 2)) - (robot_pose_current_y + radius),
                                          (goal_pose_old_x + (self.simulation.getGoalWidth() / 2)) - (robot_pose_current_x + radius))
        if orientation_goal_new < 0:
            orientation_goal_new += (2 * math.pi)

        distance_old = math.sqrt((robot_pose_old_x - (goal_pose_old_x + (self.simulation.getGoalWidth() / 2))) ** 2 +
                                 (robot_pose_old_y - (goal_pose_old_y + (self.simulation.getGoalLength() / 2))) ** 2)

        distance_new = math.sqrt((robot_pose_current_x - (goal_pose_old_x + (self.simulation.getGoalWidth() / 2))) ** 2 +
                                  (robot_pose_current_y - (goal_pose_old_y + (self.simulation.getGoalLength() / 2))) ** 2)

        delta_dist = distance_old - distance_new


        # print("Robot Orientation: " + str(robot_orientation))
        # print("Goal Orientation: " + str(orientation_goal_new))
        # print("DeltaDirection: " + str(robot_orientation - orientation_goal_new))
        #####################################################################################################

        ########### REWARD CALCULATION ################


        reward = delta_dist / 10
        #print("Delta Dist: " + str(delta_dist))



        if delta_dist > 0.0:
            reward += reward  # * 0.01
        if delta_dist == 0.0:
            reward += -0.5
        if delta_dist < 0.0:
            reward += reward #* 0.5 # * 0.001

        anglDeviation = math.fabs(robot_orientation - orientation_goal_new)
        reward += (anglDeviation * -1 + 0.35) * 2
        if anglDeviation < 0.2 and delta_dist > 0:
            reward = reward * 2

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
        if self.simulation.getRobot().isInCircleOfGoal(500):
            reward += 0.01
        if self.simulation.getRobot().isInCircleOfGoal(200):
            reward += 0.02
        if self.simulation.getRobot().isInCircleOfGoal(100):
            reward += 0.03
        if outOfArea:
            reward += -30.0
            self.done = True
        if reachedPickup:
            reward += 30.0
            self.done = True
        if reachedDelivery:
            reward += 30.0
            self.done = True

        reward -= (self.steps - self.steps_left) / self.steps
        # if self.steps_left <= 0:
        #    reward += -1.0

        # reward = factor * distance        # evtl. reward gewichten

        ################
        #print("Reward: " + str(reward))
        self.plotterWindow.plot(reward, self.simulation.simTime)
        time.sleep(0.05)


        return next_state, reward, self.is_done()

    def reset(self):
        self.simulation.getRobot().reset()
        self.steps_left = self.steps
        self.total_reward = 0.0
        self.done = False
