from Environment.SVGParser import SVGLevelParser
import numpy as np
import random

class LevelManager:
    def __init__(self, level_files):
        self.levels = level_files
        self.levels_dict = {item: 0 for item in self.levels}
        self.current_level_idx = 0
        self.current_level_id = self.levels[self.current_level_idx]

        # Current Level
        self.walls = []
        self.circleWalls = []
        self.robots = []
        self.stations = []
        self.robotOrientation = None
        self.robotPosition = None
        self.stations_pos = None
        self.arenaSize = []

    def get_level(self, idx):
        return self.levels[idx]

    def get_current_level(self):
        return self.levels[self.current_level_idx]

    def change_level(self):
        new_level = self.get_random_level()
        if new_level != self.current_level_id:
            self.current_level_idx = self.levels.index(new_level)
            self.current_level_id = new_level

    def get_random_level(self):
        return random.choice(self.levels)

    def get_level_name(self):
        return self.get_current_level().split('.', 1)[0]

    def load_level(self, args):
        self.change_level()
        selected_level = SVGLevelParser(self.get_current_level(), args)

        self.robots = selected_level.getRobots()
        if args.manually:
            self.robots = self.robots[0]

        self.stations = selected_level.getStations()
        self.stations_pos = selected_level.getStatsPos()
        self.walls = selected_level.getWalls()
        self.circleWalls = selected_level.getCircleWalls()
        self.robotOrientation = selected_level.getRobsOrient()
        self.robotPosition = selected_level.getRobsPos()
        self.arenaSize = selected_level.getArenaSize()

        for i, s in enumerate(self.stations):
            s.setPos(self.stations_pos[i])

        # Resetting each Station's color
        for i, station in enumerate(self.stations):
            station.setColor(i)

    def update(self, goals_reached):
        if np.any(goals_reached):
            self.levels_dict[self.get_current_level()] += 1

    def get_walls(self):
        return self.walls

    def get_robot_positions(self):
        return self.robotPosition

    def get_randomized_robot_positions(self):
        return random.sample(self.robotPosition, k=len(self.robotPosition))

    def randomize_stations(self):
        random.shuffle(self.stations)

        # Resetting each Station's color
        for i, station in enumerate(self.stations):
            station.setColor(i)

        return self.stations

    def __len__(self):
        return len(self.levels)