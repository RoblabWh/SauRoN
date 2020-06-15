import math
import Simulation
import numpy as np

class Environment:
    def __init__(self, app):
        self.steps_left = 500
        self.simulation = Simulation.Simulation(app)



  #  def show(self):
  #      self.simulation.show()

    def get_observation(self):
        xPos = self.simulation.getRobot().getPosX()     # Robot.Robot.getPosX(self.robot)
        yPos = self.simulation.getRobot().getPosY()
        linVel = self.simulation.getRobot().getLinearVelocity()
        angVel = self.simulation.getRobot().getAngularVelocity()
        xGoal = self.simulation.getGoalX()
        yGoal = self.simulation.getGoalY()
        return np.array([xPos, yPos, linVel, angVel, xGoal, yGoal])  # Pos, Geschwindigkeit, Zielposition

    def get_actions(self):
        return [0, 1, 2]         # Links, Rechts, Oben, Unten

    def is_done(self):
        return self.steps_left <= 0

    def step(self, action):
        done = False
        if self.is_done():
            done = True
            # raise Exception("Game is over")
        self.steps_left -= 1
        linear, angular = self.simulation.getRobot().getVelocity()
        current_state = (self.simulation.getRobot().getPosX(), self.simulation.getRobot().getPosY(),
                         self.simulation.getRobot().getLinearVelocity(), self.simulation.getRobot().getAngularVelocity(),
                         self.simulation.getGoalX(), self.simulation.getGoalY())

        # Aktion = 0 = Links
        if action == 0:
            #self.simulation.getRobot().setTargetVelocity(targetLinVel, angular - 0.5)
            self.simulation.getRobot().setTargetVelocity(1, -1)

        # Aktion = 1 = Rechts
        if action == 1:
            #self.simulation.getRobot().setTargetVelocity(targetLinVel, angular + 0.5)
            self.simulation.getRobot().setTargetVelocity(1, 1)

        # Aktion = 2 = Vorne
        if action == 2:
            #self.simulation.getRobot().setTargetVelocity(targetLinVel + 0.5, 0)
            self.simulation.getRobot().setTargetVelocity(1.5, 0)

        # Aktion = 3 = Bremsen / Rueckwaertsfahren (wenn minLinearVelocity in Robot negativ ist,
        # dann kann er rueckwaerts fahren, ansonsten stoppt er bei 0)
        if action == 3:
            #self.simulation.getRobot().setTargetVelocity(targetLinVel - 0.5, 0)
           # self.simulation.getRobot().setTargetVelocity(0, 0)   # stehen bleiben
            pass



        # einzeln Abstand berechnen
        robot_pose_old_x = self.simulation.getRobot().getPosX()
        robot_pose_old_y = self.simulation.getRobot().getPosY()
        goal_pose_old_x = self.simulation.getGoalX()
        goal_pose_old_y = self.simulation.getGoalY()

        # Update der Simulation
        self.simulation.update()

        next_state = np.array([self.simulation.getRobot().getPosX(), self.simulation.getRobot().getPosY(),
                               self.simulation.getRobot().getLinearVelocity(),
                               self.simulation.getRobot().getAngularVelocity(),
                               self.simulation.getGoalX(), self.simulation.getGoalY()])

        robot_pose_current_x = self.simulation.getRobot().getPosX()
        robot_pose_current_y = self.simulation.getRobot().getPosY()


        # euklidsche Distanz
        distance_old = math.sqrt((robot_pose_old_x - goal_pose_old_x)**2 + (robot_pose_old_y - goal_pose_old_y)**2)
        distance_new = math.sqrt((robot_pose_current_x - goal_pose_old_x)**2 + (robot_pose_current_y - goal_pose_old_y)**2)

        if distance_old > distance_new:
            reward = 5
        if distance_old < distance_new:
            reward = -5
        if distance_old == distance_new:
            reward = 0
        print ("Reward got for this action: " + str(reward))

        # reward = factor * distance        # evtl. reward gewichten

        # print(self.steps_left)
        return next_state, reward, done  # Output next_state, reward and done

    def reset(self):
        self.simulation.getRobot().setPose(5, 5)
        self.simulation.getRobot().setLinearVelocity(0)
