import math
import Simulation
import time
import numpy as np
from old.PlotterWindow import PlotterWindow


class Environment:
    def __init__(self, app, steps, args, timeframes):
        self.args = args
        self.steps = steps
        self.steps_left = steps
        self.simulation = Simulation.Simulation(app, args, timeframes)
        self.total_reward = 0.0
        self.done = False
        self.shape = np.asarray([0]).shape

        self.plotterWindow = PlotterWindow(app)

    def get_observation(self):
        #TODO den richtigen Roboter aus der Liste wählen mit parameter i --> getRobot(i)
        if self.args.mode == 'global':
            return np.asarray(self.simulation.robot.state)  # Pos, Geschwindigkeit, Zielposition
        elif self.args.mode == 'sonar':
            return np.asarray(self.simulation.robot.stateSonar)  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel

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
    #TODO Param Liste von actions für alle Roboter (bei done Robotern = null)

        self.steps_left -= 1

        #TODO Liste von tarVel erzeugen und bei update übergeben
        tarLinVel, tarAngVel = self.get_velocity(action)
        outOfArea, reachedPickup, reachedDelivery = self.simulation.update(tarLinVel, tarAngVel)
        #TODO obere Rückgabe als liste durchiterieren um für jeden noch aktiven Roboter den reward zu berechnen

        #TODO Schleife aller roboter (getRobot braucht dann eventuell i als Parameter)
        ######## Update der Simulation #######
        radius = self.simulation.getRobot().radius
        goal_pose_old_x = self.simulation.robot.getGoalX()
        goal_pose_old_y = self.simulation.robot.getGoalY()
        robot_pose_old_x = self.simulation.getRobot().getPosX()
        robot_pose_old_y = self.simulation.getRobot().getPosY()
        robot_orientation_old = self.simulation.getRobot().getDirectionAngle()
        orientation_goal_old = math.atan2(
            (goal_pose_old_y + (self.simulation.getGoalLength() / 2)) - (robot_pose_old_y),
            (goal_pose_old_x + (self.simulation.getGoalWidth() / 2)) - (robot_pose_old_x))
        if orientation_goal_old < 0:
            orientation_goal_old += (2 * math.pi)

        #######################################

        next_state = self.get_observation()#TODO state von Robot i bekommen
        next_state = np.expand_dims(next_state, axis=0)


        ############ Euklidsche Distanz und Orientierung ##############

        # einzeln Abstand berechnen
        robot_pose_current_x = self.simulation.getRobot().getPosX()
        robot_pose_current_y = self.simulation.getRobot().getPosY()

        robot_orientation_new = self.simulation.getRobot().getDirectionAngle()
        orientation_goal_new = math.atan2((goal_pose_old_y + (self.simulation.getGoalLength() / 2)) - (robot_pose_current_y),
                                          (goal_pose_old_x + (self.simulation.getGoalWidth() / 2)) - (robot_pose_current_x))
        if orientation_goal_new < 0:
            orientation_goal_new += (2 * math.pi)

        distance_old = math.sqrt((robot_pose_old_x - (goal_pose_old_x + (self.simulation.getGoalWidth() / 2))) ** 2 +
                                 (robot_pose_old_y - (goal_pose_old_y + (self.simulation.getGoalLength() / 2))) ** 2)

        distance_new = math.sqrt((robot_pose_current_x - (goal_pose_old_x + (self.simulation.getGoalWidth() / 2))) ** 2 +
                                  (robot_pose_current_y - (goal_pose_old_y + (self.simulation.getGoalLength() / 2))) ** 2)

        delta_dist = distance_old - distance_new



        ########### REWARD CALCULATION ################
        # reward = self.createReward01(delta_dist, robot_orientation_new,orientation_goal_new, outOfArea, reachedPickup, reachedDelivery)
        reward = self.createReward02(delta_dist, robot_orientation_old,orientation_goal_old, robot_orientation_new, orientation_goal_new, outOfArea, reachedPickup, reachedDelivery)



        ################
        #print("Reward: " + str(reward))
        # self.plotterWindow.plot(reward, self.simulation.simTime)
        # time.sleep(0.02)

        #TODO Liste aller Roboter mit einem Tupel dieser Variablen übergeben (next_state, reward/10, self.is_done())
        return next_state, reward/10, self.is_done()

    def createReward01(self, delta_dist, robot_orientation, orientation_goal_new, outOfArea, reachedPickup, reachedDelivery):
        reward = delta_dist / 10
        # print("Delta Dist: " + str(delta_dist))

        if delta_dist > 0.0:
            reward += reward  # * 0.01
        if delta_dist == 0.0:
            reward += -0.5
        if delta_dist < 0.0:
            reward += reward  # * 0.5 # * 0.001

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

        # reward -= (self.steps - self.steps_left) / self.steps
        # if self.steps_left <= 0:
        #    reward += -1.0

        # reward = factor * distance        # evtl. reward gewichten
        return reward


    def createReward02(self, delta_dist, robot_orientation_old,orientation_goal_old, robot_orientation_new, orientation_goal_new, outOfArea, reachedPickup, reachedDelivery):

        reward = delta_dist /2

        anglDeviation_old = math.fabs(robot_orientation_old - orientation_goal_old)
        if anglDeviation_old > math.pi:
            subtractor = 2 * (anglDeviation_old - math.pi)
            anglDeviation_old = anglDeviation_old-subtractor

        anglDeviation_new = math.fabs(robot_orientation_new - orientation_goal_new)
        if anglDeviation_new > math.pi:
            subtractor = 2* (anglDeviation_new - math.pi)
            anglDeviation_new = anglDeviation_new - subtractor

        if anglDeviation_new < anglDeviation_old:
            reward += ((math.pi - anglDeviation_new)) /math.pi /4
        else:
            reward -= ((anglDeviation_new)) / math.pi /4




        # print(reward, anglDeviation_new * 180/math.pi, (math.pi-anglDeviation_new)/math.pi /4)
        # reward+= ((anglDeviation_old - anglDeviation_new)/math.pi)/8
        # print(reward)
        # print(anglDeviation_old*180/math.pi,anglDeviation_new*180/math.pi, ((anglDeviation_old - anglDeviation_new)*180/math.pi) )

        if outOfArea:
            reward += -3.0
            self.done = True
        if reachedPickup:
            reward += 3.0
            self.done = True

        return reward


    def reset(self):
        self.simulation.getRobot().reset()
        self.steps_left = self.steps
        self.total_reward = 0.0
        self.done = False
