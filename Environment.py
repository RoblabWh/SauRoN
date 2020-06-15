import math
import Simulation
import numpy as np
import keras.backend as K


class Environment:
    def __init__(self, app, steps):
        self.steps = steps
        self.steps_left = steps
        self.simulation = Simulation.Simulation(app)
        self.total_reward = 0.0
        self.done = False

  #  def show(self):
  #      self.simulation.show()

    def get_observation(self):
        # xPos = self.simulation.getRobot().getPosX()     # Robot.Robot.getPosX(self.robot)
        # yPos = self.simulation.getRobot().getPosY()
        # linVel = self.simulation.getRobot().getLinearVelocity()
        # angVel = self.simulation.getRobot().getAngularVelocity()
        # xGoal = self.simulation.getGoalX()
        # yGoal = self.simulation.getGoalY()
        return np.asarray(self.simulation.robot.state)  # Pos, Geschwindigkeit, Zielposition

    def get_actions(self):
        return [0, 1, 2]         # Links, Rechts, Oben, Unten

    def is_done(self):
        return self.steps_left <= 0 or self.done

    def step(self, action):

        self.steps_left -= 1

        # einzeln Abstand berechnen
        goal_pose_old_x = self.simulation.robot.getGoalX()
        goal_pose_old_y = self.simulation.robot.getGoalY()
        robot_pose_old_x = self.simulation.getRobot().getPosX()
        robot_pose_old_y = self.simulation.getRobot().getPosY()

        # Aktion = 0 = Links
        if action == 0:
            vel = (1, -1)

        # Aktion = 1 = Rechts
        if action == 1:
            vel = (1, 1)

        # Aktion = 2 = Vorne
        if action == 2:
            vel = (1.5, 0)

        # Aktion = 3 = Bremsen / Rueckwaertsfahren (wenn minLinearVelocity in Robot negativ ist,
        # dann kann er rueckwaerts fahren, ansonsten stoppt er bei 0)
        if action == 3:
            vel = (0, 0)   # stehen bleiben


        # Update der Simulation
        outOfArea, reachedPickup, reachedDelivery = self.simulation.update(vel)

        next_state = self.get_observation()
        next_state = np.expand_dims(next_state, axis=0)

        robot_pose_current_x = self.simulation.getRobot().getPosX()
        robot_pose_current_y = self.simulation.getRobot().getPosY()

        # euklidsche Distanz
        distance_old = math.sqrt((robot_pose_old_x - goal_pose_old_x)**2 + (robot_pose_old_y - goal_pose_old_y)**2)
        distance_new = math.sqrt((robot_pose_current_x - goal_pose_old_x)**2 + (robot_pose_current_y - goal_pose_old_y)**2)

        if distance_old > distance_new:
            reward = 3
        if distance_old < distance_new:
            reward = -2
        if distance_old == distance_new:
            reward = -2
        if outOfArea:
            reward = -200
            self.done = True
        if reachedPickup:
            reward = 200
        if reachedDelivery:
            reward = 300
            self.done = True
        if self.steps_left <= 0:
            reward += -100
        # print ("Reward got for this action: " + str(reward))

        # reward = factor * distance        # evtl. reward gewichten
        # print(self.steps_left)
        return next_state, reward  # Output next_state, reward and done

    def reset(self):
        # self.simulation.getRobot().setPose(5, 5)
        # self.simulation.getRobot().setLinearVelocity(0)
        self.simulation.getRobot().reset()
        self.steps_left = self.steps
        self.total_reward = 0.0
        self.done = False
