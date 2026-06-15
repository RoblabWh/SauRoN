from Environment.Simulation import Simulation

import math
import numpy as np
from utils import is_staying_in_place


class Environment:
    """
    Defines the environment of the reinforcement learning algorithm
    """

    def __init__(self, args, reward_func=None, curriculum_func=None):
        """
        :param args: args defined in main
        :param reward_func: optional custom reward function (see createReward)
        :param curriculum_func: optional level-selection callback (see LevelManager)
        """

        self.args = args
        self.steps = args.steps
        self.steps_left = args.steps
        self.simulation = Simulation(args, curriculum_func=curriculum_func)
        self.episode = -1
        self.timeframs = args.time_frames
        self.total_reward = 0.0
        self.done = False
        self.shape = np.asarray([0]).shape
        self.piFact = 1 / math.pi
        if reward_func is None:
            self.reward_func = self.createReward
        else:
            self.reward_func = reward_func

    def get_observation(self, i):
        """
        Get observation from the i-th robot depending which mode is activated in the main

        - either the global state including the robots position, the orientation, linear and angular velocity
        and the goal position and the distance between the robot and the goal
        - or the local state including the data from the sonar, the linear and angular velocity,
        the orientation towards the goal and the distance to the goal

        :param i: i-th robot
        :return: returns either the global or the local (sonar) state
        """
        if self.args.mode == 'global':
            return np.asarray(self.simulation.robots[i].state)  # Pos, Geschwindigkeit, Zielposition
        #elif self.args.mode == 'sonar':
        else:
            reversed = True #determines the order of the state (reversed = false : current state in last place and the oldest at Index 0)
            #return np.asarray(self.simulation.robots[i].get_state_lidar(reversed))  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel
            return self.simulation.robots[i].get_robot_states(reversed)  # Sonardaten von x Frames, Winkel zum Ziel, Abstand zum Ziel

    @staticmethod
    def get_actions():
        """
        :return:

        In a discrete action space this method returns all discrete actions that can be taken in the world
        e.g. action number 0 represents turning left.

        In a continuous action space this method returns the number of actions for linear and angular velocity.
        """

        return 2#[0, 1, 2, 3]  # Links, Rechts, Oben, Unten

    def is_done(self):
        """
        Checks whether all robots are done (either crashed with a wall, crashed with another robot, reached their goal)
        or the steps have run out, so the simulation continues as long as at least one robot is still active.
        :return: returns True if every robot has finished or the steps have run out
        """
        robots_done = True
        for robot in self.simulation.robots:
            if robot.isActive():
                robots_done = False
                break

        return self.steps_left <= 0 or robots_done

    def update_level_manager(self, goals_reached):
        """
        Updates the levelmanager after the episode has ended
        :param goals_reached: list of booleans
        """
        self.simulation.levelManager.update(goals_reached)

    def step(self, actions, activations = None, proximity = None):
        """
        Executes a step in the environment and updates the simulation

        :param actions: list of all actions of every robot to take in this step
        :return: returns the new state of every robot in a list called robotsDataCurrentFrame
        """

        self.steps_left -= 1


        ######## Update der Simulation #######

        robotsTermination = self.simulation.update(actions, self.steps_left, activations, proximity)

        states = []
        rewards = []
        dones = []
        reachedPickups = []

        viz = self.simulation.viz
        viz_rewards = np.zeros(len(robotsTermination)) if viz is not None else None

        for i, termination in enumerate(robotsTermination):
            if termination != (None, None, None):
                state, reward, done, reachedPickup = self.extractRobotData(i, robotsTermination[i])
                states.append(state)
                rewards.append(reward)
                dones.append(int(done))
                reachedPickups.append(reachedPickup)
                if viz_rewards is not None:
                    viz_rewards[i] = sum(reward.values())
            else:
                # set robot to inactive if it has crashed with a wall or another robot
                pass

        if viz is not None:
            viz.log_rewards(viz_rewards)

        return states, rewards, dones, reachedPickups

    def extractRobotData(self, i, terminations):
        """
        Extracts state and calculates reward for the i-th robot
        :param i: i-th robot
        :param terminations: tuple - terminations of a single robot
            (Boolean collision with walls or other robots, Boolean reached PickUp, Boolean runOutOfTime)
        :return: tuple (list next state, float reward, Boolean is robot done, Boolean has reached goal)
        """
        robot = self.simulation.robots[i]
        collision, reachedPickup, runOutOfTime = terminations

        ############ State Robot i ############
        next_state = self.get_observation(i)

        ############ Distanz zum Ziel und Orientierung ##############

        goal_pos_x = robot.getGoalX()
        goal_pos_y = robot.getGoalY()
        robot_pos_old_x = robot.getLastPosX()
        robot_pos_old_y = robot.getLastPosY()

        robot_pos_current_x = robot.getPosX()
        robot_pos_current_y = robot.getPosY()

        # The reward shapes on geodesic (shortest-path) distance so that detours
        # around obstacles are paid correctly -- Euclidean shaping creates local
        # minima at every wall between robot and goal. This map knowledge is
        # privileged training-time information only; the observation's distance
        # to goal (Robot.lidarReading) deliberately stays Euclidean/map-free.
        distance_old = distance_new = None
        if robot.flowField is not None:
            distance_old = robot.flowField.distance(robot_pos_old_x, robot_pos_old_y)
            distance_new = robot.flowField.distance(robot_pos_current_x, robot_pos_current_y)
        used_geodesic = distance_old is not None and distance_new is not None
        if not used_geodesic:
            distance_old = math.sqrt((robot_pos_old_x - goal_pos_x) ** 2 +
                                     (robot_pos_old_y - goal_pos_y) ** 2)
            distance_new = math.sqrt(
                (robot_pos_current_x - goal_pos_x) ** 2 +
                (robot_pos_current_y - goal_pos_y) ** 2)

        # Geodesic and Euclidean distances are incommensurable: comparing the
        # ratcheted bestGoalDist across a metric switch pays a huge spurious
        # progress reward (geodesic best vs. straight-line current) and then
        # locks the ratchet below any reachable geodesic value for the rest of
        # the episode. Re-anchor in the new metric instead; this step pays no
        # progress reward.
        if used_geodesic != robot.bestGoalDistIsGeodesic:
            robot.bestGoalDist = distance_new
            robot.bestGoalDistIsGeodesic = used_geodesic
            if not used_geodesic and not robot.fallbackWarned:
                # a mid-episode fallback usually means the flow field has a hole:
                # a passage sealed by inflation (flow_field_cell too coarse) or a
                # position off-grid
                print(f"[flowfield] robot {robot.idx} fell back to Euclidean reward at "
                      f"({robot_pos_current_x:.2f}, {robot_pos_current_y:.2f}) in episode {self.episode}",
                      flush=True)
                robot.fallbackWarned = True

        ########### REWARD CALCULATION ################

        reward = self.reward_func(robot, distance_new, distance_old, reachedPickup, collision, runOutOfTime)

        # Only collision and arrival are true terminal states. Running out of time is a
        # truncation of the episode, not part of the task -- the transition is stored as
        # not-done so the value of the successor state gets bootstrapped.
        return [next_state, reward, collision or reachedPickup, reachedPickup]


    def createReward(self, robot, dist_new, dist_old, reachedPickup, collision, runOutOfTime):
        """
        Creates a (sparse) reward based on the euklidian distance, if the robot has reached his goal and if the robot
        collided with a wall or another robot.

        :param robot: robot
        :param dist_new: the new distance (after the action has been taken)
        :param dist_old: the old distance (before the action has been taken)
        :param reachedPickup: True if the robot reached his goal in this step
        :param collision: True if the robot collided with a wall or another robot
        :return: returns the result of the fitness function
        """

        reward = {}
        r_arrival = 10 # reward for reaching the goal
        r_collision = -10 # Robot crashed with a wall or another robot
        w_d = 2.5 # weight for the progress towards the goal (max speed ~0.04 m/step -> ~0.1/step)
        w_w = -0.1

        if reachedPickup:
            reward['arrival'] = r_arrival
        elif collision:
            reward['collision'] = r_collision
        else:
            # Progress reward: one-sided against the best (smallest) goal distance
            # achieved this episode. Backing out of dead ends and taking detours
            # around obstacles costs nothing (symmetric shaping punished every step
            # of a correct detour -> agents got stuck in alleys or died pushing
            # into walls towards the goal). Paying against the episode-best instead
            # of the previous step keeps loops unfarmable: re-covering old ground
            # earns nothing, only a new closest approach pays, and total shaping is
            # bounded by w_d * initial distance. Running out of time is intentionally
            # not punished: the time limit is a training artifact the agent cannot
            # observe, not part of the task.
            progress = robot.bestGoalDist - dist_new
            if progress > 0:
                reward['dist'] = w_d * progress
                robot.bestGoalDist = dist_new

            # Minimal distance to obstacle (pre-noise lidar reading, in meters)
            if np.min(robot.distances[0]) < 0.3:
                reward['wall'] = w_w

            # Protect engine Reward
            # currentLinVel = np.around(robot.state_raw[robot.time_steps - 1][4], decimals=5)
            # lastLinVel = np.around(robot.state_raw[robot.time_steps - 2][4], decimals=5)
            #
            # currentAngVel = np.around(robot.state_raw[robot.time_steps - 1][5], decimals=5)
            # lastAngVel = np.around(robot.state_raw[robot.time_steps - 2][5], decimals=5)

            # if (np.abs(currentLinVel - lastLinVel) < 0.15) and (np.abs(currentAngVel - lastAngVel) < 0.30):
            #     reward['motorReward'] = 0.1
            # else:
            #     reward['motorReward'] = -0.1
            #
            # # Agent stays in the same region for some time, indicating being stuck or driving in circles
            # if is_staying_in_place(robot.last_positions):
            #     reward['stucked'] = -10

            # Stop Reward
            # if abs(dist_old - dist_new) < 0.001:
            #     reward['stop'] = r_stop

            # Minimal distance to obstacle
            # if np.min(robot.get_state_lidar()[0][0]) < 0.012:
            #     reward['wall'] = w_w

            # Directional reward (look at the angle between the robot and the goal)
            # a1 = np.arctan2(robot.getGoalY() - robot.getPosY(), robot.getGoalX() - robot.getPosX())
            # a2 = np.arctan2(robot.getDirectionY(), robot.getDirectionX())
            # goalangle = np.abs(a1 - a2)
            # if goalangle < np.pi/4:
            #     alpha_norm = 1 - goalangle
            #     reward['directional'] = a_p * alpha_norm

            # Wiggle reward
            # if currentAngVel > 0.7:
            #     reward['wiggle'] = w_w * currentAngVel

            # PUBG Reward (only gets rewarded if it gets closer to the goal than previously)
            # if dist_new < robot.initialGoalDist:
            #     reward['pubg'] = w_d * (robot.initialGoalDist - dist_new)
            #     robot.initialGoalDist = dist_new

        return reward

    def reset(self):
        """
        Resets the simulation after each epoch
        :param level: int - defines the reset level
        """
        self.simulation.reset()
        self.steps_left = self.steps
        self.episode += 1
        self.simulation.episode = self.episode
        self.total_reward = 0.0
        self.done = False

        states = [self.get_observation(i) for i in range(self.simulation.getCurrentNumberOfRobots())]
        return states

    # returns number of robots in the simulation
    def getNumberOfRobots(self):
        return self.simulation.getCurrentNumberOfRobots()

    def getLevelManager(self):
        return self.simulation.levelManager

    def updateTrainingCounter(self, counter):
        self.simulation.updateTrainingCounter(counter)
