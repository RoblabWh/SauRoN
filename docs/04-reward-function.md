# Reward Function

This is a technical inventory of the reward machinery: every term that exists
in the code, what triggers it, and its exact formula. Terms currently disabled
in the source are listed in their own section with the same level of detail.

## Interface

The reward for one robot and one step is a **dict of named components**, e.g.
`{'dist': 0.08, 'wall': -0.1}`. The training loop sums the values into the
scalar reward (`PPO/Environment.py:87`); the names exist for logging and
visualization.

The default reward is `Environment.createReward`
(`Environment/Environment.py:197`). A custom reward can be injected without
touching the class:

```python
Environment(args, reward_func=my_reward)
# signature: my_reward(robot, dist_new, dist_old, reachedPickup, collision, runOutOfTime) -> dict
```

where `dist_new` / `dist_old` are the goal distances after / before the step,
in the active distance metric (below).

## Distance metric for shaping

The shaping terms operate on a goal distance that is either **geodesic** or
**Euclidean** (`Environment/Environment.py:152-185`):

- With `--geodesic_reward True` (default), the distance comes from a
  precomputed geodesic flow field: the length of the shortest obstacle-free
  path to the goal, with walls inflated by the robot radius (see
  [Flow field](#geodesic-flow-field) below). Rationale recorded in the code:
  Euclidean shaping creates a local minimum at every wall between robot and
  goal; geodesic shaping pays detours around obstacles correctly. This is
  privileged training-time map knowledge and enters the reward only — the
  observation's goal distance stays Euclidean.
- If the robot's position does not resolve to a finite field value (off-grid,
  or in a region sealed by inflation), the step **falls back to Euclidean**
  straight-line distance. A one-time warning per robot is printed, since a
  mid-episode fallback usually indicates a hole in the flow field
  (`flow_field_cell` too coarse).
- The two metrics are incommensurable. On a metric switch the progress ratchet
  `robot.bestGoalDist` is **re-anchored** to the current distance in the new
  metric and that step pays no progress reward — otherwise the switch would pay
  a large spurious progress reward and then lock the ratchet below any
  reachable value for the rest of the episode.

## Active terms

Constants from `Environment/Environment.py:210-238`:

```python
r_arrival   = 10
r_collision = -10
w_d         = 2.5    # progress weight (max speed ~0.04 m/step -> ~0.1/step)
w_w         = -0.1   # wall proximity
```

`arrival` and `collision` are mutually exclusive with all other terms in the
same step (if either fires, it is the only component that step).

### `arrival` — +10

Paid once, in the step in which the robot reaches its goal (center distance
plus robot radius < 1.2 × station radius). Terminal.

### `collision` — −10

Paid once, in the step in which the robot's body touches a wall, circle
obstacle, or another robot (minimum collision distance ≤ robot radius).
Terminal.

### `dist` — one-sided progress, `w_d · (bestGoalDist − dist_new)`

Paid only when the robot sets a **new episode-best (smallest) goal distance**;
`bestGoalDist` is then updated to `dist_new` (a ratchet). Properties recorded
in the code comments (`Environment.py:221-231`):

- Backing out of dead ends and taking detours costs nothing — symmetric
  shaping punished every step of a correct detour, so agents got stuck in
  alleys or pushed into walls toward the goal.
- Loops are unfarmable: re-covering old ground earns nothing; only a new
  closest approach pays.
- The total progress reward over an episode is bounded by
  `w_d × initial goal distance`.

`bestGoalDist` is initialized to the robot's geodesic start distance on reset
(`Simulation.py:77-81`) and re-anchored on metric switches as described above.

### `wall` — −0.1 per step near obstacles

Paid in every step in which the minimum **pre-noise** lidar distance is below
0.3 m (`np.min(robot.distances[0]) < 0.3`, `Environment.py:236-238`).

### Timeout — no term

Running out of steps pays nothing. The code comment states the rationale: the
time limit is a training artifact the agent cannot observe, not part of the
task. The transition is stored as a truncation (not-done) and the successor
state's value is bootstrapped in the PPO update (`Environment.py:191-194`).

## Inactive terms (present in code, commented out)

These exist in `Environment.createReward` (`Environment.py:240-279`) as
commented-out blocks and document reward variants that the codebase supports
re-enabling. They are **not** part of the current reward.

| Name | Trigger | Formula / value |
| --- | --- | --- |
| `motorReward` | change in linear velocity < 0.15 **and** change in angular velocity < 0.30 between the last two frames | +0.1, else −0.1 |
| `stucked` | robot's recent position history stays inside a 1 m × 1 m bounding box (`is_staying_in_place`, `utils.py:415`) | −10 |
| `stop` | `abs(dist_old − dist_new) < 0.001` (robot effectively not moving) | `r_stop` (constant not defined in the active code; must be supplied when re-enabling) |
| `wall` (alternative) | minimum of the *encoded* lidar state < 0.012 | `w_w` |
| `directional` | angle between heading and goal direction < π/4 | `a_p · (1 − goalangle)` (`a_p` not defined in the active code) |
| `wiggle` | current angular velocity > 0.7 | `w_w · angVel` |
| `pubg` | `dist_new < robot.initialGoalDist` | `w_d · (initialGoalDist − dist_new)`, then `initialGoalDist ← dist_new` — the same one-sided ratchet as `dist`, anchored at the initial distance |

## Geodesic flow field

`Environment/FlowField.py`. Built once per (level, goal position, cell size,
robot radius) and memoized in `FlowFieldCache`; station positions are fixed
within a level, so each field is built exactly once per training run.

Construction (`build_geodesic_field`):

1. **Rasterize** the level's wall segments and circle obstacles onto a grid
   with cell size `--flow_field_cell` (default 0.1 m).
2. **Inflate** the raster by the robot radius (disk dilation), so distances
   reflect paths the robot's body can actually take. The cell size must stay
   well below the narrowest passage width minus the robot diameter, or
   inflation seals passages.
3. **Seed** the zero-distance region with all free cells inside the actual
   arrival disk (`1.2 × station radius − robot radius`, matching the arrival
   condition — seeding with the station radius alone would leave a
   zero-gradient plateau before arrival triggers; `Simulation.py:68-72`). If no
   free seed cell exists, the nearest free cell is used.
4. **Flood** geodesic distances outward with a vectorized 8-neighbor
   relaxation (diagonal moves cost `cell·√2`) until convergence.
5. **Band-fill**: the field is extended a few cells into the inflated margin,
   because cell quantization can put a physically legal robot center into a
   blocked cell — it must still read a finite, smooth value. These band values
   are valid for `distance()` only; the greedy `path()` used by the
   visualization never descends through inflated cells, since band cells at
   wall tips inherit the far side's distance.

`GeodesicField.distance(x, y)` returns the distance in meters or `None`
(off-grid / sealed region), which triggers the Euclidean fallback described
above. `--viz_flow_field True` renders the raster, inflation band, and each
robot's shortest path in the rerun viewer.
