import colorsys
import os
import numpy as np


def stationColorRGB(i):
    """
    Color of the i-th station, same hue rotation the old Qt visualization used
    (QColor.fromHsv((i * 39) % 255, 255, 255)).
    """
    hue = ((i * 39) % 255) / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


# draw order: flow-field raster at the very back, lidar behind geometry, headings on top
DRAW_ORDER_FLOW = -10.0
DRAW_ORDER_LIDAR = 0.0
DRAW_ORDER_FLOW_PATH = 5.0
DRAW_ORDER_WALLS = 10.0
DRAW_ORDER_STATIONS = 20.0
DRAW_ORDER_ROBOTS = 30.0
DRAW_ORDER_HEADINGS = 40.0

BACKGROUND_COLOR = [18, 20, 26]
WALL_COLOR = [150, 152, 160]


class RerunVisualizer:
    """
    Streams the simulation into the rerun viewer (https://rerun.io).

    The viewer runs in its own process and renders on the GPU, so logging adds
    almost no overhead to the simulation loop. Pausing, scrubbing back through
    past episodes, toggling the lidar point clouds and zoom/pan all happen in
    the viewer without affecting training.
    """

    def __init__(self, args):
        # Batch log calls into 200ms chunks instead of the 8ms default. At sim rate
        # (~475 logged frames/s across several entities) the default floods the viewer
        # with tiny chunks whose index/metadata overhead counts against the memory
        # budget but cannot be reclaimed proportionally -- the GC then purges almost
        # the entire timeline in one go instead of trimming the oldest data smoothly
        # (rerun-io/rerun#5967, #5974). Bigger chunks also keep eviction time-ordered,
        # which prevents stale geometry (e.g. circle walls) from outliving its Clear.
        os.environ.setdefault('RERUN_FLUSH_TICK_SECS', '0.2')

        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr
        self.rrb = rrb
        self.step = 0
        # at training speed the sim runs far faster than any display; logging every
        # step would cost ~30% throughput for frames nobody can see. Test mode logs
        # every step (it is watched in real time), training every 4th by default.
        stride = getattr(args, 'viz_stride', None)
        if stride is None:
            stride = 1 if getattr(args, 'mode', 'train') == 'test' else 4
        self.stride = max(1, stride)
        self.lidar_stride = self.stride * max(1, getattr(args, 'viz_lidar_stride', 2))
        # angular decimation: the lidar clouds dominate the streamed data volume,
        # and every 3rd of 1081 rays is still visually dense for a 270 degree fan
        self.lidar_decimation = max(1, getattr(args, 'viz_lidar_decimation', 3))
        # debug overlay of the geodesic reward field; heavy (tens of thousands of
        # points per reset), so strictly opt-in
        self.viz_flow_field = bool(getattr(args, 'viz_flow_field', False))
        self._flow_entities = 0
        self._logged_this_step = False
        self._levelManager = None
        self._goalStations = None
        self._episode = 0
        # The viewer's memory GC evicts the oldest chunks per entity with no notion of
        # episodes. The level geometry is logged once per episode; once that event is
        # evicted, the rest of the episode would render without walls. Re-logging the
        # geometry every few logged frames keeps every retained time point a complete
        # scene at negligible cost (3 small batched calls per refresh).
        self.geometry_refresh = 5
        self._frames_since_geometry = 0

        # A long-lived kernel (Jupyter) keeps rerun's global SDK state between cell
        # runs. If the user closed the viewer, the gRPC sink from the previous run
        # is severed; rr.init() below flushes that prior recording (via
        # flush_and_cleanup_orphaned_recordings) and flushing into the dead
        # connection raises "gRPC connection severed". Drop the stale sink first so
        # that flush is a harmless no-op and a fresh viewer can be spawned. Safe on
        # the first run too (no active recording -> disconnect is a no-op).
        try:
            rr.disconnect()
        except Exception:
            pass

        rr.init('SauRoN')

        # The blueprint is sent exactly once per run, at startup: re-sending it later
        # would wipe the user's in-session manual tweaks (hidden entities, zoom,
        # layout). It must however be sent as the ACTIVE blueprint, not just the
        # default -- the viewer persists the active blueprint per app-id across
        # sessions and silently prefers it over any default, so blueprint changes in
        # this code would otherwise never reach a viewer that has seen 'SauRoN'
        # before. The camera is framed on the union of all level arenas so it fits
        # every level without ever being re-sent.
        blueprint = self._make_blueprint(arena_size=self._union_arena_size(args))

        save_path = getattr(args, 'rerun_save', None)
        if save_path:
            # the viewer picks its importer by extension and refuses anything but .rrd
            if not save_path.endswith('.rrd'):
                save_path += '.rrd'
                print(f'Note: rerun recordings need a .rrd extension, saving to {save_path} instead', flush=True)
            rr.save(save_path, default_blueprint=blueprint)
            print(f'Recording visualization to {save_path} (view with: rerun {save_path})', flush=True)
        else:
            memory_limit = str(getattr(args, 'rerun_memory_limit', '25%'))
            self._warn_tiny_memory_limit(memory_limit)
            rr.spawn(memory_limit=memory_limit, default_blueprint=blueprint)
            rr.send_blueprint(blueprint)

    @staticmethod
    def _union_arena_size(args):
        """Largest width/height over all configured level files (parsed once)."""
        try:
            from Environment.SVGParser import SVGLevelParser
            width = height = 0.0
            for level_file in getattr(args, 'level_files', []):
                try:
                    size = SVGLevelParser(level_file, args).getArenaSize()
                    width = max(width, float(size[0]))
                    height = max(height, float(size[1]))
                except Exception:
                    continue
            if width > 0 and height > 0:
                return width, height
        except Exception:
            pass
        return None

    def _make_blueprint(self, arena_size=None):
        rrb = self.rrb

        # Only the lidar gets a visible-time window: it is per-step data that must
        # expire instead of leaving a dead robot's last scan frozen on screen. The
        # window is slightly larger than the lidar's logging cadence so it always
        # contains the latest cloud while data is retained.
        #
        # The level geometry (walls, stations, circle walls) deliberately uses plain
        # latest-at instead: a cursor-relative window blanks the LIVE view whenever
        # the lookback happens to be empty -- GC eviction under memory pressure or
        # per-entity chunk ordering at flush boundaries strobe the walls in and out
        # (observed as constant flicker during training). Latest-at cannot blank at
        # the live head, since the newest geometry event is the last thing evicted.
        # Stale-geometry artifacts are handled without windows: geometry is re-logged
        # every few frames (geometry_refresh) and circle-free levels log an explicit
        # Clear. Trade-off: scrubbing deep into GC-evicted history may show a stale
        # level's geometry instead of nothing.
        def window(seq_back):
            return rrb.archetypes.VisibleTimeRanges(rrb.VisibleTimeRange(
                'step',
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-seq_back),
                end=rrb.TimeRangeBoundary.cursor_relative()))

        overrides = {
            'world/lidar': window(self.lidar_stride + self.stride),
            'world/flow_paths': window(self.lidar_stride + self.stride),
        }

        # camera framed on the arena instead of auto-fitting to whatever entities are
        # currently visible (which makes the view jump when robots finish)
        visual_bounds = None
        if arena_size is not None:
            width, height = float(arena_size[0]), float(arena_size[1])
            margin_x = max(0.5, 0.07 * width)
            margin_y = max(0.5, 0.07 * height)
            visual_bounds = rrb.VisualBounds2D(x_range=[-margin_x, width + margin_x],
                                               y_range=[-margin_y, height + margin_y])

        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin='world', name='Simulation',
                                  background=BACKGROUND_COLOR,
                                  visual_bounds=visual_bounds,
                                  overrides=overrides),
                rrb.Vertical(
                    # sliding window instead of the full retained history: the plot
                    # caches share the memory budget with the data store (#5904)
                    rrb.TimeSeriesView(origin='metrics', name='Rewards',
                                       time_ranges=rrb.VisibleTimeRange(
                                           'step',
                                           start=rrb.TimeRangeBoundary.cursor_relative(seq=-20000),
                                           end=rrb.TimeRangeBoundary.cursor_relative())),
                    rrb.TextDocumentView(origin='info', name='Level'),
                    row_shares=[4, 1],
                ),
                column_shares=[3, 2],
            )
        )

    @staticmethod
    def _warn_tiny_memory_limit(memory_limit):
        """The viewer's UI, caches and renderer share the counted memory budget with
        the data store (rerun-io/rerun#5974) and alone occupy tens of MB. Below that
        floor the GC is permanently over-limit and compacts continuously."""
        limit = memory_limit.strip().upper()
        if limit.endswith('%'):
            return
        units = {'KB': 1e3, 'KIB': 2**10, 'MB': 1e6, 'MIB': 2**20,
                 'GB': 1e9, 'GIB': 2**30, 'TB': 1e12, 'TIB': 2**40, 'B': 1}
        for suffix, factor in sorted(units.items(), key=lambda kv: -len(kv[0])):
            if limit.endswith(suffix):
                try:
                    num_bytes = float(limit[:-len(suffix)]) * factor
                except ValueError:
                    return
                if num_bytes < 1e9:
                    print(f'WARNING: --rerun_memory_limit {memory_limit} is below the viewer\'s '
                          f'own working memory (UI + caches, tens of MB share the same budget). '
                          f'The garbage collector will compact continuously. Use at least 2GB '
                          f'or a percentage.', flush=True)
                return

    def log_level(self, levelManager, robots=None):
        """Logs the geometry of the current level (walls, circle walls, stations)."""
        rr = self.rr
        self._levelManager = levelManager
        self._episode += 1
        # only stations that are actually some robot's goal are rendered;
        # levels may contain more stations than robots
        if robots is not None:
            self._goalStations = list(dict.fromkeys(robot.station for robot in robots))
        else:
            self._goalStations = None

        rr.set_time('step', sequence=self.step)
        self._log_geometry()

        rr.log('info/level', rr.TextDocument(
            f'## {levelManager.get_level_name()}\n\n'
            f'Episode {self._episode}' + (f' — {len(robots)} robots' if robots is not None else ''),
            media_type='text/markdown'))

        if robots is not None:
            # reward plot legend: one named series per robot, in its world color
            rr.log('metrics/reward', rr.SeriesLines(
                names=[f'robot {i}' for i in range(len(robots))],
                colors=[robot.station.color for robot in robots]), static=True)

    def _log_geometry(self):
        rr = self.rr
        levelManager = self._levelManager

        walls = levelManager.get_walls()
        strips = [[wall.getStart(), wall.getEnd()] for wall in walls]
        rr.log('world/walls', rr.LineStrips2D(strips, colors=WALL_COLOR, radii=0.025,
                                              draw_order=DRAW_ORDER_WALLS))

        circleWalls = levelManager.circleWalls
        if circleWalls:
            rr.log('world/circle_walls', rr.Points2D(
                [[c.getPosX(), c.getPosY()] for c in circleWalls],
                radii=[c.getRadius() for c in circleWalls],
                colors=WALL_COLOR,
                draw_order=DRAW_ORDER_WALLS))
        else:
            # empty component batches do not override older data on latest-at
            # queries -- without the explicit clear, circles from a previous
            # level would stay visible on circle-free levels
            rr.log('world/circle_walls', rr.Clear(recursive=False))

        stations = self._goalStations if self._goalStations is not None else levelManager.stations
        rr.log('world/stations', rr.Points2D(
            [[s.getPosX(), s.getPosY()] for s in stations],
            radii=[s.getRadius() for s in stations],
            colors=[s.color for s in stations],
            draw_order=DRAW_ORDER_STATIONS))

    def log_flow_field(self, robots):
        """Debug overlay of the geodesic reward fields: free cells colored by
        distance to the goal (yellow = near, blue = far), wall raster cells dark
        red and the robot-radius inflation band orange -- rendering raster and
        band separately makes rasterization vs. inflation bugs distinguishable.
        Logged once per reset (static per episode)."""
        if not self.viz_flow_field:
            return
        rr = self.rr
        rr.set_time('step', sequence=self.step)

        fields = list(dict.fromkeys(robot.flowField for robot in robots
                                    if getattr(robot, 'flowField', None) is not None))
        for i, field in enumerate(fields):
            free_xy, free_g, raster_xy, band_xy = field.debug_cells()
            g_max = max(float(free_g.max()), 1e-6) if len(free_g) else 1.0
            t = (free_g / g_max)[:, None]
            colors = ((1 - t) * np.array([[252, 220, 80, 110]]) +
                      t * np.array([[50, 70, 200, 110]])).astype(np.uint8)
            rr.log(f'world/flow_field/g{i}', rr.Points2D(
                free_xy, radii=field.cell * 0.5, colors=colors, draw_order=DRAW_ORDER_FLOW))
        if fields:
            field = fields[0]  # raster/inflation are identical across goals of a level
            _, _, raster_xy, band_xy = field.debug_cells()
            rr.log('world/flow_field/blocked', rr.Points2D(
                np.concatenate([raster_xy, band_xy]) if len(band_xy) else raster_xy,
                radii=field.cell * 0.5,
                colors=np.concatenate([np.tile([150, 45, 45, 200], (len(raster_xy), 1)),
                                       np.tile([200, 130, 60, 140], (len(band_xy), 1))]).astype(np.uint8),
                draw_order=DRAW_ORDER_FLOW))
        # levels can have fewer goals than the previous one: clear stale entities
        for i in range(len(fields), self._flow_entities):
            rr.log(f'world/flow_field/g{i}', rr.Clear(recursive=False))
        self._flow_entities = len(fields)

    def log_step(self, robots):
        """Logs robot poses, headings and lidar point clouds for one simulation step.

        Everything is logged as batched entities (one log call for all robots) --
        per-entity log calls are the dominant cost of streaming at sim rate.
        """
        rr = self.rr
        self.step += 1
        if self.step % self.stride:
            self._logged_this_step = False
            return
        self._logged_this_step = True
        rr.set_time('step', sequence=self.step)

        self._frames_since_geometry += 1
        if self._frames_since_geometry >= self.geometry_refresh and self._levelManager is not None:
            self._frames_since_geometry = 0
            self._log_geometry()

        positions, radii, colors = [], [], []
        origins, vectors = [], []
        lidar_points, lidar_colors = [], []
        log_lidar = (self.step % self.lidar_stride) == 0

        for robot in robots:
            color = robot.station.color
            x, y = robot.getPosX(), robot.getPosY()
            positions.append([x, y])
            radii.append(robot.radius)
            if not robot.isActive():
                # finished robots (crashed or arrived) stay visible as translucent
                # ghosts frozen at their terminal position, without heading/lidar
                colors.append((*color, 100))
                continue
            colors.append((*color, 255))
            origins.append([x, y])
            vectors.append([1.5 * robot.radius * robot.getDirectionX(),
                            1.5 * robot.radius * robot.getDirectionY()])
            if log_lidar and len(robot.lidarHits) > 0:
                # only rays that actually hit something; no-hit rays carry a
                # synthetic endpoint at the sensor fill range outside the level
                hits = np.asarray(robot.lidarHits)[robot.lidarHitMask][::self.lidar_decimation]
                if len(hits) > 0:
                    lidar_points.append(hits)
                    lidar_colors.append(np.full((len(hits), 4), (*color, 230), dtype=np.uint8))

        if positions:
            rr.log('world/robots', rr.Points2D(positions, radii=radii, colors=colors,
                                               draw_order=DRAW_ORDER_ROBOTS))
        else:
            rr.log('world/robots', rr.Clear(recursive=False))
        if origins:
            rr.log('world/headings', rr.Arrows2D(origins=origins, vectors=vectors,
                                                 colors=[255, 255, 255], radii=0.02,
                                                 draw_order=DRAW_ORDER_HEADINGS))
        else:
            rr.log('world/headings', rr.Clear(recursive=False))
        if log_lidar:
            if lidar_points:
                rr.log('world/lidar', rr.Points2D(np.concatenate(lidar_points), radii=0.018,
                                                  colors=np.concatenate(lidar_colors),
                                                  draw_order=DRAW_ORDER_LIDAR))
            else:
                rr.log('world/lidar', rr.Clear(recursive=False))
        if self.viz_flow_field and log_lidar:
            strips, strip_colors = [], []
            for robot in robots:
                if robot.isActive() and getattr(robot, 'flowField', None) is not None:
                    path = robot.flowField.path(robot.getPosX(), robot.getPosY())
                    if len(path) >= 2:
                        strips.append(path)
                        strip_colors.append(robot.station.color)
            if strips:
                rr.log('world/flow_paths', rr.LineStrips2D(strips, colors=strip_colors, radii=0.02,
                                                           draw_order=DRAW_ORDER_FLOW_PATH))
            else:
                rr.log('world/flow_paths', rr.Clear(recursive=False))

    def log_rewards(self, rewards):
        """Logs the summed step reward of every robot (one batched scalar series)."""
        if self._logged_this_step:
            self.rr.log('metrics/reward', self.rr.Scalars(rewards))
