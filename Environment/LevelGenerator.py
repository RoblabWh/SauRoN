"""
Procedural level generator — the single source of truth for random levels.

Used by the simulator (``gen:`` tokens in --level_files, fresh level per
episode) and by the level editor (POST /api/generate in leveleditor.py).
Emits the same simple-subset SVG the editor saves, which SVGLevelParser is
guaranteed to accept; geometry is in raw SVG px (28.35 px = 1 m, scaling is
parser-side).

Guarantees: free space forms one main region (small sealed pockets between
obstacles are tolerated but never hold markers), every start's goal is drawn
from the start's own BFS tree (reachability proven, not sampled), mazes are
braided (zero dead ends), and all markers keep robot-radius clearance from
walls. Randomness flows through an own mulberry32 PRNG, independent of
Python's global ``random`` state, so a seed reproduces a level regardless of
what the simulation has rolled before.
"""

import math
import os
from urllib.parse import parse_qsl

import numpy as np

PX_PER_M = 28.35
ROBOT_R_PX = 0.175 * PX_PER_M
CELL = 0.1 * PX_PER_M          # matches the sim's flow-field resolution

START_R = 7.5                  # marker radii used across the level files
GOAL_R = 13.99
ORANGE_R = 13.8

PRESETS = ('maze', 'rooms', 'open', 'pillars')

_PARAM_CASTS = {
    'robots_min': int, 'robots_max': int,
    'pool': lambda v: str(v).lower() in ('1', 'true', 'yes'),
    'passage': float, 'density': float,
    'seed': int, 'width': float, 'height': float,
}


def parse_gen_token(token):
    """'gen:maze?density=80&seed=42' -> ('maze', {'density': 80.0, 'seed': 42})"""
    if not token.startswith('gen:'):
        raise ValueError(f'not a generator token: {token!r}')
    preset, _, query = token[4:].partition('?')
    if preset not in PRESETS:
        raise ValueError(f'unknown generator preset {preset!r} in {token!r}; choose from {PRESETS}')
    overrides = {}
    for key, value in parse_qsl(query):
        if key not in _PARAM_CASTS:
            raise ValueError(f'unknown generator parameter {key!r} in {token!r}; '
                             f'valid: {sorted(_PARAM_CASTS)}')
        try:
            overrides[key] = _PARAM_CASTS[key](value)
        except ValueError:
            raise ValueError(f'bad value {value!r} for generator parameter {key!r} in {token!r}')
    return preset, overrides


def _mulberry32(a):
    a &= 0xFFFFFFFF

    def rng():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rng


def _ri(rng, a, b):
    return a + int(rng() * (b - a + 1))


def _shuffle(rng, arr):
    for i in range(len(arr) - 1, 0, -1):
        j = int(rng() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr


class _Level:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.lines = []     # (x1, y1, x2, y2)
        self.circles = []   # (cx, cy, r)
        self.starts = []    # (cx, cy)
        self.goals = []     # (cx, cy, r)
        self.orange = []    # (cx, cy, r)


def _boundary(lvl):
    m, W, H = 2, lvl.width, lvl.height
    lvl.lines += [(m, m, W - m, m), (W - m, m, W - m, H - m),
                  (W - m, H - m, m, H - m), (m, H - m, m, m)]


# ----------------------------------------------------------------- grid ops

def _rasterize(lvl):
    cols = max(1, math.ceil(lvl.width / CELL))
    rows = max(1, math.ceil(lvl.height / CELL))
    occ = np.zeros((rows, cols), dtype=bool)

    def mark(xs, ys):
        cx = np.clip((xs / CELL).astype(int), 0, cols - 1)
        cy = np.clip((ys / CELL).astype(int), 0, rows - 1)
        # the 4-neighbors too, so zero-thickness walls can't be crossed
        # diagonally between two cell corners
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            occ[np.clip(cy + dy, 0, rows - 1), np.clip(cx + dx, 0, cols - 1)] = True

    for x1, y1, x2, y2 in lvl.lines:
        steps = max(1, math.ceil(math.hypot(x2 - x1, y2 - y1) / (CELL / 2)))
        t = np.linspace(0.0, 1.0, steps + 1)
        mark(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)

    if lvl.circles:
        ys = (np.arange(rows) + 0.5)[:, None] * CELL
        xs = (np.arange(cols) + 0.5)[None, :] * CELL
        for cx, cy, r in lvl.circles:
            occ |= (xs - cx) ** 2 + (ys - cy) ** 2 <= (r + CELL / 2) ** 2
    return occ


def _clearance(occ):
    """Manhattan distance (px) to the nearest wall cell — separable two-pass
    distance transform, equivalent to a 4-neighbor BFS from all wall cells."""
    rows, cols = occ.shape
    big = rows + cols + 2
    d = np.where(occ, 0, big).astype(np.int32)
    for y in range(1, rows):
        np.minimum(d[y], d[y - 1] + 1, out=d[y])
    for y in range(rows - 2, -1, -1):
        np.minimum(d[y], d[y + 1] + 1, out=d[y])
    for x in range(1, cols):
        np.minimum(d[:, x], d[:, x - 1] + 1, out=d[:, x])
    for x in range(cols - 2, -1, -1):
        np.minimum(d[:, x], d[:, x + 1] + 1, out=d[:, x])
    return d.astype(np.float64) * CELL


def _dilate4(m):
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _largest_component(mask):
    """Keep only the biggest connected free region. Small sealed pockets are
    inevitable with robot-level inflation — markers must never land in them,
    and free-size tells how much area they waste."""
    free = int(mask.sum())
    unvisited = mask.copy()
    best, best_size = np.zeros_like(mask), 0
    while unvisited.any():
        seed = np.unravel_index(np.argmax(unvisited), mask.shape)
        comp = np.zeros_like(mask)
        comp[seed] = True
        frontier = comp.copy()
        while frontier.any():
            frontier = _dilate4(frontier) & unvisited & ~comp
            comp |= frontier
        unvisited &= ~comp
        size = int(comp.sum())
        if size > best_size:
            best, best_size = comp, size
    return best, best_size, free


def _bfs_dist(mask, start):
    """Geodesic distances (px) over the mask from one cell, 4-neighborhood."""
    dist = np.full(mask.shape, np.inf)
    if not mask[start]:
        return dist
    dist[start] = 0.0
    frontier = np.zeros_like(mask)
    frontier[start] = True
    d = 0.0
    while frontier.any():
        d += CELL
        frontier = _dilate4(frontier) & mask & np.isinf(dist)
        dist[frontier] = d
    return dist


# ------------------------------------------------------------------ presets

def _gen_maze(W, H, rng, passage_px, density):
    # the maze ring doubles as the arena boundary: pitch stretches to fill the
    # arena exactly, so corridors are >= the requested passage in both axes
    cols = int((W - 4) // passage_px)
    rows = int((H - 4) // passage_px)
    if cols < 3 or rows < 3:
        raise ValueError(
            f'arena too small for a maze with {passage_px / PX_PER_M:.1f} m passages — '
            f'needs at least {math.ceil(3 * passage_px + 4)}px per side')
    pX, pY = (W - 4) / cols, (H - 4) / rows
    hw = np.ones((rows + 1, cols), dtype=bool)   # walls above each cell row
    vw = np.ones((rows, cols + 1), dtype=bool)   # walls left of each cell col

    # carve: iterative recursive backtracker
    visited = np.zeros(rows * cols, dtype=bool)
    stack = [_ri(rng, 0, rows * cols - 1)]
    visited[stack[0]] = True
    while stack:
        cell = stack[-1]
        r, c = divmod(cell, cols)
        moves = [(dr, dc) for dr, dc in _shuffle(rng, [(-1, 0), (1, 0), (0, -1), (0, 1)])
                 if 0 <= r + dr < rows and 0 <= c + dc < cols and not visited[(r + dr) * cols + c + dc]]
        if not moves:
            stack.pop()
            continue
        dr, dc = moves[0]
        if dr:
            hw[max(r, r + dr), c] = False
        else:
            vw[r, max(c, c + dc)] = False
        nxt = (r + dr) * cols + (c + dc)
        visited[nxt] = True
        stack.append(nxt)

    # braid: knock one interior wall out of every dead end (cell with exactly
    # 3 standing walls) until none remain — no dead-end traps, by construction
    def cell_walls(r, c):
        return [
            (hw[r, c], r > 0, lambda: hw.__setitem__((r, c), False)),
            (hw[r + 1, c], r < rows - 1, lambda: hw.__setitem__((r + 1, c), False)),
            (vw[r, c], c > 0, lambda: vw.__setitem__((r, c), False)),
            (vw[r, c + 1], c < cols - 1, lambda: vw.__setitem__((r, c + 1), False)),
        ]

    found = True
    while found:
        found = False
        for r in range(rows):
            for c in range(cols):
                walls = cell_walls(r, c)
                if sum(1 for up, _, _ in walls if up) != 3:
                    continue
                options = [knock for up, interior, knock in walls if up and interior]
                if not options:
                    continue
                options[int(rng() * len(options))]()
                found = True

    # density: lower = knock out extra interior walls for more shortcuts
    interior = []
    for r in range(1, rows):
        for c in range(cols):
            if hw[r, c]:
                interior.append(('h', r, c))
    for r in range(rows):
        for c in range(1, cols):
            if vw[r, c]:
                interior.append(('v', r, c))
    _shuffle(rng, interior)
    for kind, r, c in interior[:round((100 - density) / 100 * 0.15 * len(interior))]:
        (hw if kind == 'h' else vw)[r, c] = False

    # emit maximal collinear runs as single editable lines (incl. outer ring)
    lvl = _Level(W, H)
    x0 = y0 = 2
    for r in range(rows + 1):
        c = 0
        while c < cols:
            if not hw[r, c]:
                c += 1
                continue
            c2 = c
            while c2 < cols and hw[r, c2]:
                c2 += 1
            lvl.lines.append((x0 + c * pX, y0 + r * pY, x0 + c2 * pX, y0 + r * pY))
            c = c2
    for c in range(cols + 1):
        r = 0
        while r < rows:
            if not vw[r, c]:
                r += 1
                continue
            r2 = r
            while r2 < rows and vw[r2, c]:
                r2 += 1
            lvl.lines.append((x0 + c * pX, y0 + r * pY, x0 + c * pX, y0 + r2 * pY))
            r = r2
    return lvl


def _gen_rooms(W, H, rng, passage_px, density):
    lvl = _Level(W, H)
    _boundary(lvl)
    # density 0 -> few ~6 m rooms, 100 -> many rooms near the passage scale
    min_room = max(2 * passage_px,
                   6 * PX_PER_M * (1 - density / 100) + 2 * passage_px * (density / 100))
    doors = []  # door gaps already cut: (orient, pos, lo, hi)

    # a new wall must not butt into an existing door gap (it would split the
    # door into two sub-passage openings)
    def hits_door(orient, edge_a, edge_b, coord):
        return any(o == orient and (abs(pos - edge_a) < 1 or abs(pos - edge_b) < 1)
                   and lo - passage_px / 2 < coord < hi + passage_px / 2
                   for o, pos, lo, hi in doors)

    def divide(x, y, w, h):
        if max(w, h) < 2 * min_room or min(w, h) < 2 * passage_px:
            return
        if w >= h:
            wx = -1.0
            for _ in range(12):
                cand = x + min_room + rng() * (w - 2 * min_room)
                if not hits_door('h', y, y + h, cand):
                    wx = cand
                    break
            if wx < 0:
                return
            g_lo = y + passage_px / 2 + rng() * (h - 2 * passage_px)
            if g_lo - y > 0.5:
                lvl.lines.append((wx, y, wx, g_lo))
            if y + h - (g_lo + passage_px) > 0.5:
                lvl.lines.append((wx, g_lo + passage_px, wx, y + h))
            doors.append(('v', wx, g_lo, g_lo + passage_px))
            divide(x, y, wx - x, h)
            divide(wx, y, x + w - wx, h)
        else:
            wy = -1.0
            for _ in range(12):
                cand = y + min_room + rng() * (h - 2 * min_room)
                if not hits_door('v', x, x + w, cand):
                    wy = cand
                    break
            if wy < 0:
                return
            g_lo = x + passage_px / 2 + rng() * (w - 2 * passage_px)
            if g_lo - x > 0.5:
                lvl.lines.append((x, wy, g_lo, wy))
            if x + w - (g_lo + passage_px) > 0.5:
                lvl.lines.append((g_lo + passage_px, wy, x + w, wy))
            doors.append(('h', wy, g_lo, g_lo + passage_px))
            divide(x, y, w, wy - y)
            divide(x, wy, w, y + h - wy)

    divide(2, 2, W - 4, H - 4)
    return lvl


def _gen_open(W, H, rng, passage_px, density):
    lvl = _Level(W, H)
    _boundary(lvl)
    area_m2 = (W / PX_PER_M) * (H / PX_PER_M)
    tries = round(density / 100 * area_m2 / 6)  # ~1 obstacle per 6 m² at 100%
    # accept each obstacle only while the passage-navigable main area stays
    # large and nearly nothing gets walled off into sealed pockets
    infl = max(ROBOT_R_PX + 0.1 * PX_PER_M, passage_px / 2) + CELL / 2

    def stats():
        clear = _clearance(_rasterize(lvl))
        _, size, free = _largest_component(clear > infl)
        return size, free

    base, _ = stats()
    margin = passage_px / 2 + CELL
    for _ in range(tries):
        cx = margin + rng() * (W - 2 * margin)
        cy = margin + rng() * (H - 2 * margin)
        if rng() < 0.5:
            lvl.circles.append((cx, cy, (0.3 + rng() * 0.6) * PX_PER_M))
            undo = lvl.circles.pop
        else:
            hw = (0.25 + rng() * 0.75) * PX_PER_M
            hh = (0.25 + rng() * 0.75) * PX_PER_M
            lvl.lines += [(cx - hw, cy - hh, cx + hw, cy - hh),
                          (cx + hw, cy - hh, cx + hw, cy + hh),
                          (cx + hw, cy + hh, cx - hw, cy + hh),
                          (cx - hw, cy + hh, cx - hw, cy - hh)]
            undo = lambda: lvl.lines.__delitem__(slice(-4, None))
        size, free = stats()
        if size < 0.4 * base or free - size > 0.05 * base:
            undo()
    return lvl


def _gen_pillars(W, H, rng, passage_px, density):
    lvl = _Level(W, H)
    _boundary(lvl)
    pr = 0.3 * PX_PER_M
    s = passage_px + 2 * pr
    y = 2 + s * 0.75
    while y < H - 2 - s * 0.5:
        x = 2 + s * 0.75
        while x < W - 2 - s * 0.5:
            if rng() < density / 100:
                lvl.circles.append((x + (rng() - 0.5) * 0.3 * s,
                                    y + (rng() - 0.5) * 0.3 * s, pr))
            x += s
        y += s
    return lvl


_GEN_FUNCS = {'maze': _gen_maze, 'rooms': _gen_rooms, 'open': _gen_open, 'pillars': _gen_pillars}


# ---------------------------------------------------------------- placement

def _place_markers(lvl, clear, mask, rng, n, use_pool, min_sep_px):
    """Markers only on cells with real clearance inside the main free region;
    each goal is drawn from its start's own BFS tree — reachability is proven."""
    lvl.starts, lvl.goals, lvl.orange = [], [], []
    place_clear = ROBOT_R_PX + 0.15 * PX_PER_M
    rows, cols = mask.shape
    ys, xs = np.nonzero(mask & (clear >= place_clear))
    candidates = list(zip(ys.tolist(), xs.tolist()))
    _shuffle(rng, candidates)
    cell_xy = lambda yx: ((yx[1] + 0.5) * CELL, (yx[0] + 0.5) * CELL)
    placed = []

    def far_enough(x, y):
        return all(math.hypot(px - x, py - y) >= min_sep_px for px, py in placed)

    if use_pool:
        for yx in candidates:
            x, y = cell_xy(yx)
            if not far_enough(x, y):
                continue
            placed.append((x, y))
            lvl.orange.append((x, y, ORANGE_R))
            if len(lvl.orange) == 2 * n + 1:
                return True
        return False

    for _ in range(n):
        s_yx = next((yx for yx in candidates if far_enough(*cell_xy(yx))), None)
        if s_yx is None:
            return False
        sx, sy = cell_xy(s_yx)
        dist = _bfs_dist(mask, s_yx)
        cand_d = np.array([dist[yx] for yx in candidates])
        max_geo = np.max(cand_d[np.isfinite(cand_d)], initial=0.0)
        g_yx = None
        # prefer goals a good trek away; relax if the level is too small
        for floor in (max(2 * PX_PER_M, 0.35 * max_geo), 2 * PX_PER_M, CELL):
            for yx, d in zip(candidates, cand_d):
                if yx == s_yx or not np.isfinite(d) or d < floor:
                    continue
                gx, gy = cell_xy(yx)
                if far_enough(gx, gy) and math.hypot(gx - sx, gy - sy) >= min_sep_px:
                    g_yx = yx
                    break
            if g_yx is not None:
                break
        if g_yx is None:
            return False
        gx, gy = cell_xy(g_yx)
        placed += [(sx, sy), (gx, gy)]
        lvl.starts.append((sx, sy))
        lvl.goals.append((gx, gy, GOAL_R))
    return True


# ------------------------------------------------------------------- export

def _fmt(v):
    return f'{v:.3f}'.rstrip('0').rstrip('.')


def _to_svg(lvl):
    W, H = _fmt(lvl.width), _fmt(lvl.height)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    for i, (cx, cy, r) in enumerate(lvl.goals):
        out.append(f'  <circle id="goal_{i + 1}" cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" '
                   f'fill="#66b32e" stroke="#000000"/>')
    for i, (cx, cy) in enumerate(lvl.starts):
        out.append(f'  <circle id="start_{i + 1}" cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{START_R}" '
                   f'fill="#e52420" stroke="#000000"/>')
    for cx, cy, r in lvl.orange:
        out.append(f'  <circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" fill="none" stroke="orange"/>')
    for cx, cy, r in lvl.circles:
        out.append(f'  <circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" fill="none" stroke="#000000"/>')
    for x1, y1, x2, y2 in lvl.lines:
        out.append(f'  <line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
                   f'fill="none" stroke="#000000" stroke-width="2"/>')
    out += ['</svg>', '']
    return '\n'.join(out)


# -------------------------------------------------------------- entry point

def generate_level(preset, *, width=620.0, height=280.0, robots_min=1, robots_max=2,
                   pool=False, passage=1.2, density=50.0, seed=None):
    """Generate one level; returns (svg_text, used_seed).

    Raises ValueError with a human-readable cure when the parameters cannot
    produce a valid level (arena too small, too dense for the robot count).
    """
    if preset not in _GEN_FUNCS:
        raise ValueError(f'unknown generator preset {preset!r}; choose from {PRESETS}')
    robots_min = min(4, max(1, int(robots_min)))
    robots_max = min(4, max(1, int(robots_max)))
    if robots_min > robots_max:
        robots_min, robots_max = robots_max, robots_min
    passage = min(3.0, max(0.8, float(passage)))
    density = min(100.0, max(0.0, float(density)))
    passage_px = passage * PX_PER_M
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')
    seed = int(seed) & 0xFFFFFFFF

    infl_trav = ROBOT_R_PX + 0.1 * PX_PER_M + CELL / 2
    last_error = None
    for attempt in range(5):
        rng = _mulberry32((seed + 1000 * attempt) & 0xFFFFFFFF)
        try:
            lvl = _GEN_FUNCS[preset](width, height, rng, passage_px, density)
        except ValueError as err:
            raise ValueError(str(err)) from None  # deterministic size error — no retry helps
        clear = _clearance(_rasterize(lvl))
        mask, size, free = _largest_component(clear > infl_trav)
        if not size or size < 0.75 * free:
            last_error = 'free space too fragmented'
            continue
        n = _ri(rng, robots_min, robots_max)
        if not (_place_markers(lvl, clear, mask, rng, n, pool, 1.0 * PX_PER_M) or
                _place_markers(lvl, clear, mask, rng, n, pool, 0.8 * PX_PER_M)):
            last_error = f'could not place {n} robot(s)'
            continue
        return _to_svg(lvl), seed
    raise ValueError(f'generation failed after 5 attempts ({last_error}) — '
                     f'lower density / robot count or enlarge the arena')
