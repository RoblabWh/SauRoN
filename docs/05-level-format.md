# Level Format (SVG)

Levels are plain SVG files in `svg/` parsed by `Environment/SVGParser.py`.
`--level_files` paths are given relative to `svg/`. The visual level editor
(`uv run leveleditor.py`, http://127.0.0.1:8000) reads and writes this format
faithfully — when changing the parser, keep `LevelEditor/index.html`'s
importer/exporter in sync. `svg/alt/` contains legacy levels and is not
auto-loaded.

## Coordinates and arena size

- Coordinates are raw SVG pixels; the parser scales everything by **1/28.35**
  (72 dpi → dots per centimeter) into the simulation's meter units. A 7.5 px
  circle radius therefore becomes ≈ 0.26 m.
- The arena size comes from the `viewBox` attribute, which must be **exactly
  four space-separated numbers**. The default SVG namespace
  (`xmlns="http://www.w3.org/2000/svg"`) is required.

## Walls

The following elements become wall geometry (solid from both sides; line
direction is irrelevant — the `'tareq' in filename` reversal in the parser is a
leftover no-op):

| Element | Notes |
| --- | --- |
| `<line>` | straight wall segment |
| `<rect>` | four wall segments. Quirk: a `matrix(a b c d e f)` transform applies only the first four numbers, as a rotation about the rect **center**; the translation part (`e f`) is ignored |
| `<polyline>` / `<polygon>` | chained segments (polygon closes itself) |
| `<path>` | **only** the relative form `m x,y dx,dy ... z` is supported |
| `<circle>` (no special id/stroke, see below) | circular wall obstacle |

## Robots and goals: `<circle>` semantics

A `<circle>` is classified by the parser in this exact branch order:

1. `id` contains `start` → **robot spawn point**
2. `id` contains `goal` → **goal station**
3. any **other** non-empty meaning in the `id` → **silently dropped** (a typo
   like `strat` makes the circle disappear without an error)
4. `stroke="orange"` → member of the **random start/goal pool**: positions are
   shuffled into starts and goals each episode (the pool needs ≥ 3 circles to
   yield robots)
5. otherwise → circular **wall**

Example from `svg/ez.svg`:

```xml
<circle id="start_2_1_" fill="#e52420" stroke="#000000" cx="238.4" cy="116.9" r="7.5"  />
<circle id="goal_4_1_-3" fill="#66b32e" stroke="#000000" cx="304.7" cy="116.1" r="13.99" />
```

## Robot–goal pairing

Robot *i* (i-th `start` circle in document order) is paired with the i-th
`goal` circle in document order.

- More starts than goals → `IndexError` at load time.
- Extra goals are allowed: they become unused stations (e.g. for multigoal
  levels), and other robots' stations are non-colliding by default
  (`--collide_other_targets False`).

## Parsing a level standalone

To parse a level without booting the whole simulation, stub the robot class
first (constructing real Robots needs the full args namespace):

```python
import Environment.SVGParser as P
P.Robot.Robot = lambda *a, **k: None
p = P.SVGLevelParser('kreuzung.svg', None)  # filename relative to svg/
```
