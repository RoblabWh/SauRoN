from Environment.SVGParser import SVGLevelParser
from Environment.LevelGenerator import generate_level, parse_gen_token
import numpy as np
import random

class LevelManager:
    def __init__(self, level_files, curriculum_func=None):
        self.levels = level_files
        # optional student-supplied curriculum: picks the next level from the
        # per-level success stats; None falls back to uniform random selection
        self.curriculum_func = curriculum_func
        self.levels_dict = {item: 0 for item in self.levels}
        self.current_level_idx = 0
        self.current_level_id = self.levels[self.current_level_idx]

        # fail fast on malformed gen: tokens before training starts
        for token in self.levels:
            if token.startswith('gen:'):
                parse_gen_token(token)

        # Current Level
        self.walls = []
        self.circleWalls = []
        self.robots = []
        self.stations = []
        self.robotOrientation = None
        self.robotPosition = None
        self.stations_pos = None
        self.arenaSize = []
        self.current_is_generated = False
        self.current_gen_seed = None
        self._gen_counter = 0

    def get_level(self, idx):
        return self.levels[idx]

    def get_current_level(self):
        return self.levels[self.current_level_idx]

    def change_level(self):
        if self.curriculum_func is not None:
            new_level = self.curriculum_func(
                list(self.levels), dict(self.levels_dict), self.current_level_id)
            # ignore an invalid pick (e.g. typo / removed level) and keep the
            # current one rather than crashing the training run
            if new_level not in self.levels:
                new_level = self.current_level_id
        else:
            new_level = self.get_random_level()
        if new_level != self.current_level_id:
            self.current_level_idx = self.levels.index(new_level)
            self.current_level_id = new_level

    def get_random_level(self):
        return random.choice(self.levels)

    def get_level_name(self):
        if self.current_is_generated:
            return f'{self.get_current_level()} · seed {self.current_gen_seed}'
        return self.get_current_level().split('.', 1)[0]

    def _generate_current(self, args):
        """Generate a fresh level for the current gen: token. Token overrides
        beat the global --gen_* defaults; an explicit seed= pins the level."""
        preset, overrides = parse_gen_token(self.get_current_level())
        arena = getattr(args, 'gen_arena', None) or (620.0, 280.0)
        params = {
            'width': arena[0], 'height': arena[1],
            'robots_min': getattr(args, 'gen_robots_min', 1),
            'robots_max': getattr(args, 'gen_robots_max', 2),
            'pool': getattr(args, 'gen_pool', False),
            'passage': getattr(args, 'gen_passage', 1.2),
            'density': getattr(args, 'gen_density', 50.0),
        }
        if 'seed' in overrides:
            seed = overrides.pop('seed')
        elif getattr(args, 'gen_seed', None) is not None:
            seed = args.gen_seed + self._gen_counter  # reproducible sequence
        else:
            seed = None  # fresh random level
        params.update(overrides)
        self._gen_counter += 1
        svg_text, used_seed = generate_level(preset, seed=seed, **params)
        self.current_gen_seed = used_seed
        return svg_text

    def load_level(self, args):
        self.change_level()
        token = self.get_current_level()
        self.current_is_generated = token.startswith('gen:')
        if self.current_is_generated:
            selected_level = SVGLevelParser(token, args, svg_text=self._generate_current(args))
        else:
            self.current_gen_seed = None
            selected_level = SVGLevelParser(token, args)

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