import math
import numpy as np


def _shifted(a, dy, dx, fill):
    """Returns a copy of the 2D array `a` shifted by (dy, dx), padded with `fill`."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


class GeodesicField:
    """
    Geodesic distance field of a level for one goal: for every grid cell the
    length of the shortest obstacle-free path to the goal, with the obstacles
    inflated by the robot radius so distances reflect paths the robot's body
    can actually take. Used for reward shaping only -- the policy's
    observations stay map-free by design.
    """

    def __init__(self, g, raster, inflated, x0, y0, cell):
        self.g = g                  # (H, W) float64, meters, inf where unusable
        self.raster = raster        # (H, W) bool, cells touched by wall geometry
        self.inflated = inflated    # (H, W) bool, raster dilated by the robot radius
        self.x0 = x0
        self.y0 = y0
        self.cell = cell

    def _cellIndex(self, x, y):
        col = int((x - self.x0) / self.cell)
        row = int((y - self.y0) / self.cell)
        if 0 <= row < self.g.shape[0] and 0 <= col < self.g.shape[1]:
            return row, col
        return None

    def distance(self, x, y):
        """Geodesic distance to the goal in meters, or None if the position is
        off-grid or in a sealed region (caller falls back to Euclidean)."""
        idx = self._cellIndex(x, y)
        if idx is None:
            return None
        d = self.g[idx]
        return float(d) if np.isfinite(d) else None

    def path(self, x, y, max_steps=None):
        """Greedy descent on the field from (x, y) to the goal; returns the
        world-coordinate cell centers of the shortest path (for visualization).
        Descends over free cells only: inflated cells carry finite band-fill
        values for the reward, and at wall tips those inherit the far side's
        distance, so following them would cut the path through the wall corner.
        A start inside the band (wall-hugging robot) may step on band values
        until it escapes to a free cell."""
        idx = self._cellIndex(x, y)
        if idx is None or not np.isfinite(self.g[idx]):
            return np.empty((0, 2))
        h, w = self.g.shape
        if max_steps is None:
            max_steps = h + w + int(2 * (h + w))
        points = []
        row, col = idx
        for _ in range(max_steps):
            points.append((self.x0 + (col + 0.5) * self.cell, self.y0 + (row + 0.5) * self.cell))
            here = self.g[row, col]
            free_best, free_idx = here, None
            band_best, band_idx = here, None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    r, c = row + dy, col + dx
                    if (dy or dx) and 0 <= r < h and 0 <= c < w:
                        if not self.inflated[r, c]:
                            if self.g[r, c] < free_best:
                                free_best, free_idx = self.g[r, c], (r, c)
                        elif self.g[r, c] < band_best:
                            band_best, band_idx = self.g[r, c], (r, c)
            best_idx = free_idx
            if best_idx is None and self.inflated[row, col]:
                best_idx = band_idx  # still escaping a blocked start cell
            if best_idx is None:  # reached the goal seed (or a numerical plateau)
                break
            row, col = best_idx
        return np.asarray(points)

    def debug_cells(self):
        """Cell centers for the rerun overlay: (free_xy, free_g, raster_xy, band_xy).
        `band` is the inflation margin (inflated but not raster) -- rendering it
        separately makes rasterization vs. inflation bugs distinguishable."""
        ys, xs = np.nonzero(np.isfinite(self.g) & ~self.inflated)
        free_xy = np.stack([self.x0 + (xs + 0.5) * self.cell, self.y0 + (ys + 0.5) * self.cell], axis=1)
        free_g = self.g[ys, xs]
        ys, xs = np.nonzero(self.raster)
        raster_xy = np.stack([self.x0 + (xs + 0.5) * self.cell, self.y0 + (ys + 0.5) * self.cell], axis=1)
        ys, xs = np.nonzero(self.inflated & ~self.raster)
        band_xy = np.stack([self.x0 + (xs + 0.5) * self.cell, self.y0 + (ys + 0.5) * self.cell], axis=1)
        return free_xy, free_g, raster_xy, band_xy


def build_geodesic_field(wall_segments, circle_walls, goal_x, goal_y, goal_radius, robot_radius, cell):
    """
    Builds a GeodesicField by rasterizing the static level geometry and flooding
    geodesic distances outward from the goal.

    :param wall_segments: iterable of ((x1, y1), (x2, y2)) wall lines
    :param circle_walls: iterable of (cx, cy, r) circle obstacles
    """
    segs = np.asarray([(s[0][0], s[0][1], s[1][0], s[1][1]) for s in wall_segments], dtype=np.float64)
    circles = np.asarray([(c[0], c[1], c[2]) for c in circle_walls], dtype=np.float64).reshape(-1, 3)

    xs = np.concatenate([segs[:, 0], segs[:, 2], circles[:, 0] - circles[:, 2], circles[:, 0] + circles[:, 2]])
    ys = np.concatenate([segs[:, 1], segs[:, 3], circles[:, 1] - circles[:, 2], circles[:, 1] + circles[:, 2]])
    pad = int(math.ceil(robot_radius / cell)) + 1
    x0 = xs.min() - pad * cell
    y0 = ys.min() - pad * cell
    width = int(math.ceil((xs.max() - x0) / cell)) + pad + 1
    height = int(math.ceil((ys.max() - y0) / cell)) + pad + 1

    raster = np.zeros((height, width), dtype=bool)
    for sx, sy, ex, ey in segs:
        n = max(2, int(math.ceil(math.hypot(ex - sx, ey - sy) / (cell * 0.5))) + 1)
        cols = ((np.linspace(sx, ex, n) - x0) / cell).astype(int)
        rows = ((np.linspace(sy, ey, n) - y0) / cell).astype(int)
        keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        raster[rows[keep], cols[keep]] = True
    cell_x = x0 + (np.arange(width) + 0.5) * cell
    cell_y = y0 + (np.arange(height) + 0.5) * cell
    for cx, cy, r in circles:
        raster |= ((cell_x[None, :] - cx) ** 2 + (cell_y[:, None] - cy) ** 2) <= r * r

    # inflate by the robot radius (disk dilation as OR of shifted rasters)
    r_cells = int(math.ceil(robot_radius / cell))
    inflated = raster.copy()
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if (dy or dx) and math.hypot(dy, dx) * cell <= robot_radius + 1e-9:
                inflated |= _shifted(raster, dy, dx, False)

    # multi-source seed: all free cells within the station radius (arriving means
    # the robot center enters that disk; also robust when the goal hugs a wall)
    seed = (((cell_x[None, :] - goal_x) ** 2 + (cell_y[:, None] - goal_y) ** 2) <= goal_radius ** 2) & ~inflated
    if not seed.any():
        free_ys, free_xs = np.nonzero(~inflated)
        if len(free_ys) == 0:
            raise ValueError('flow field: level rasterized to no free cells -- cell size too coarse?')
        nearest = np.argmin((cell_x[free_xs] - goal_x) ** 2 + (cell_y[free_ys] - goal_y) ** 2)
        seed = np.zeros_like(inflated)
        seed[free_ys[nearest], free_xs[nearest]] = True

    # geodesic distances: vectorized relaxation (each sweep propagates one cell
    # ring, iterations ~ longest shortest-path in cells)
    moves = [(dy, dx, cell * math.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    g = np.where(seed, 0.0, np.inf)
    g[inflated] = np.inf
    while True:
        g_new = g
        for dy, dx, cost in moves:
            g_new = np.minimum(g_new, _shifted(g, dy, dx, np.inf) + cost)
        g_new = np.where(inflated, np.inf, g_new)
        if not (g_new < g).any():
            break
        g = g_new

    # extend the field a little into the inflated band: cell quantization can put
    # a physically legal robot center into a blocked cell, which must still read
    # a finite, smooth value for the reward. These values are for distance() only:
    # at wall tips a band cell inherits the far side's distance, so path() must
    # never descend through inflated cells or it cuts through wall corners
    for _ in range(r_cells + 1):
        nmin = np.full_like(g, np.inf)
        for dy, dx, cost in moves:
            nmin = np.minimum(nmin, _shifted(g, dy, dx, np.inf) + cost)
        fill = np.isinf(g) & np.isfinite(nmin)
        if not fill.any():
            break
        g = np.where(fill, nmin, g)

    return GeodesicField(g, raster, inflated, x0, y0, cell)


class FlowFieldCache:
    """
    Memoizes geodesic fields per (level, goal position, resolution, robot radius).
    Station positions are fixed within a level (randomize_stations only shuffles
    the robot<->station assignment), so each field is built once per training run.
    """

    def __init__(self):
        self._cache = {}
        self.builds = 0

    def get(self, level_id, wall_segments, circle_walls, goal_x, goal_y, goal_radius, robot_radius, cell, store=True):
        # store=False for generated levels: their geometry is unique per
        # episode, so a name-keyed entry would either go stale (key reuse) or
        # leak memory (unique keys, ~88 KB per field, never hit again)
        if not store:
            self.builds += 1
            return build_geodesic_field(wall_segments, circle_walls, goal_x, goal_y,
                                        goal_radius, robot_radius, cell)
        key = (level_id, round(goal_x, 2), round(goal_y, 2), round(goal_radius, 2), round(cell, 3), round(robot_radius, 2))
        field = self._cache.get(key)
        if field is None:
            field = build_geodesic_field(wall_segments, circle_walls, goal_x, goal_y,
                                         goal_radius, robot_radius, cell)
            self._cache[key] = field
            self.builds += 1
        return field
