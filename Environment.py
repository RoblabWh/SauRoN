import math
import Simulation
import numpy as np


class Environment:
    def __init__(self, app, steps, args, timeframes):
        self.steps = steps
        self.steps_left = steps
        self.simulation = Simulation.Simulation(app, args, timeframes)
        self.total_reward = 0.0
        self.done = False
        self.shape = np.asarray([0]).shape

    def get_observation(self):
        return np.asarray(self.simulation.robot.state)  # Pos, Geschwindigkeit, Zielposition

    @staticmethod
    def get_actions():
        return [0, 1, 2, 3]  # Links, Rechts, Oben, Unten

    @staticmethod
    def get_velocity(action):

        # Aktion = 0 = Links
        if action == 0: vel = (0, -1)

        # Aktion = 1 = Rechts
        if action == 1: vel = (0, 1)

        # Aktion = 2 = Vorne
        if action == 2: vel = (1, 0)

        # Aktion = 3 = Bremsen / Rueckwaertsfahren (wenn minLinearVelocity in Robot negativ ist,
        # dann kann er rueckwaerts fahren, ansonsten stoppt er bei 0)
        if action == 3: vel = (0, 0)  # stehen bleiben

        return vel

    def is_done(self):
        return self.steps_left <= 0 or self.done

    def step(self, action):

        self.steps_left -= 1

        vel = self.get_velocity(action)

        ######## Update der Simulation #######
        radius = self.simulation.getRobot().radius
        goal_pose_old_x = self.simulation.robot.getGoalX()
        goal_pose_old_y = self.simulation.robot.getGoalY()
        robot_pose_old_x = self.simulation.getRobot().getPosX() + radius
        robot_pose_old_y = self.simulation.getRobot().getPosY() + radius

        outOfArea, reachedPickup, reachedDelivery = self.simulation.update(vel)
        #######################################

        next_state = self.get_observation()
        next_state = np.expand_dims(next_state, axis=0)

        ############ Euklidsche Distanz und Orientierung ##############

        # einzeln Abstand berechnen
        robot_pose_current_x = self.simulation.getRobot().getPosX() + radius
        robot_pose_current_y = self.simulation.getRobot().getPosY() + radius

        robot_orientation = self.simulation.getRobot().getDirection()
        orientation_goal_new = math.atan2((goal_pose_old_y + self.simulation.getGoalLength()) - (robot_pose_current_y + radius),
                                          (goal_pose_old_x + self.simulation.getGoalWidth()) - (robot_pose_current_x + radius))
        orientation_goal_new += (2 * math.pi)

        distance_old = math.sqrt((robot_pose_old_x - goal_pose_old_x) ** 2 +
                                 (robot_pose_old_y - goal_pose_old_y) ** 2)

        distance_new = math.sqrt((robot_pose_current_x - goal_pose_old_x) ** 2 +
                                 (robot_pose_current_y - goal_pose_old_y) ** 2)

        delta_dist = distance_old - distance_new

        # print("Robot Orientation: " + str(robot_orientation))
        # print("Goal Orientation: " + str(orientation_goal_new))
        # print("DeltaDirection: " + str(robot_orientation - orientation_goal_new))
        #####################################################################################################

        ########### REWARD CALCULATION ################

        if delta_dist > 0.0:
            reward = delta_dist * 0.01
        else:
            reward = delta_dist * 0.001

        if math.fabs(robot_orientation - orientation_goal_new) < 0.3:  # 0.05
            if(distance_old - distance_new) > 0:
                reward += 0.1

        if math.fabs(robot_orientation - orientation_goal_new) < 0.5:  # 0.05
            if(distance_old - distance_new) > 0:
                reward += 0.01
        # else:
        #     reward += -0.1
        # if distance_old > distance_new:
        #     reward += 2
        # if distance_old < distance_new:
        #     reward = -1
        # if distance_old == distance_new:
        #     reward += -1
        if self.simulation.getRobot().isInCircleOfGoal(300):
            reward += 0.001
        if self.simulation.getRobot().isInCircleOfGoal(200):
            reward += 0.002
        if self.simulation.getRobot().isInCircleOfGoal(100):
            reward += 0.003
        if outOfArea:
            reward += -2.0
            self.done = True
        if reachedPickup:
            reward = 20.0
        if reachedDelivery:
            reward = 30.0
            self.done = True
        if self.steps_left <= 0:
            reward += -1.0

        # print ("Reward got for this action: " + str(reward))
        # reward = factor * distance        # evtl. reward gewichten

        ################
        # print(reward)
        return next_state, reward, self.is_done()

    def reset(self):
        self.simulation.getRobot().reset()
        self.steps_left = self.steps
        self.total_reward = 0.0
        self.done = False
