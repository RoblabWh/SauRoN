# Observation & Action Space

## Design constraint: map-free observations

The policy never sees the map. Observations consist only of what an onboard
sensor suite could measure: the lidar scan, the (straight-line) distance and
bearing to the goal, and the robot's own velocities. Map knowledge — the
geodesic flow field — may shape the **reward** only (see
[04 — Reward Function](04-reward-function.md)); it is never part of the network
input. This is a hard project constraint (`CLAUDE.md`).

## Observation

Each robot's state is built in `Robot.lidarReading`
(`Environment/Components/Robot.py:397`) once per simulation step and consists
of four streams, each stacked over the last `--time_frames` frames
(default 4, newest frame first):

| Stream | Shape per frame | Encoding |
| --- | --- | --- |
| Lidar scan | 1081 | inverse depth `x = d0 / (d0 + d)` |
| Orientation to goal | 2 | unit vector of the bearing angle |
| Distance to goal | 1 | saturating `x = d / (d + d0)` |
| Velocity | 2 | linear and angular, normalized to [-1, 1] |

With the defaults this gives a 4×1081 lidar tensor plus 4×2, 4×1 and 4×2
proprioceptive tensors (4344 numbers in total).

### Lidar

- `--number_of_rays 1081` rays over a `--field_of_view 270`° arc, raycast
  against all wall segments, circle obstacles, and other robots' bodies.
- Rays that hit nothing report a synthetic fill distance (the level diagonal),
  so there is no hard maximum sensor range.
- Uniform noise in ±0.04 m is added to the measured distances
  (`Robot.py:490`) before encoding. The pre-noise distances are kept separately
  and used by the wall-proximity reward term.
- **Inverse-depth encoding** `x = d0 / (d0 + d)` with `--lidar_d0 2.0`:
  values are bounded in (0, 1], a wall touching the robot reads ~1, an obstacle
  at `d0` meters reads 0.5, and free space fades smoothly toward 0. The scale
  `d0` sets where the encoding spends its resolution — small `d0` concentrates
  it near the robot. Negative noisy readings are clamped at 0 before encoding
  so the output stays in (0, 1].

### Goal distance and bearing

- Distance is **Euclidean**, deliberately, even when the reward shapes on
  geodesic distance: `x = d / (d + d0)` with `--goal_d0 10.0` — 0 at the goal,
  0.5 at `d0` meters, approaching 1 far away (`Robot.py:469-471`).
- Bearing is the signed angle between the robot's heading and the direction to
  the goal, given to the network as a 2D unit vector
  `(cos α, sin α)` (`Robot.py:473-487`).

### Velocity

Raw linear and angular velocity (m/s, rad/s) mapped to [-1, 1] by
midpoint/half-range scaling against the robot's physical velocity limits
(`Robot.py:683-688`). The stored values are raw per-second velocities, not
dt-scaled — scaling by the timestep would compress the channel and couple it to
`--sim_time_step` (comment at `Robot.py:341-343`). Checkpoints from before this
change (June 2026) are stale.

## Action space

Two continuous actions per robot, each in [-1, 1] (the policy's tanh squash
guarantees the bounds; see [03 — PPO Algorithm](03-ppo-algorithm.md)):

| Action | Mapped to | Range |
| --- | --- | --- |
| `a[0]` | target linear velocity | 0 … 0.175 m/s |
| `a[1]` | target angular velocity | −1 … +1 rad/s |

The mapping is affine, `v = v_min + (a + 1)/2 · (v_max − v_min)`, followed by
discretization into fixed bins (0.005 m/s linear, 0.0555 rad/s angular) in
`Robot.computeNextVelocity` (`Robot.py:534-552`). Target velocities take
effect **immediately** — the acceleration-limited variant
(`computeNextVelocityContinuous`) exists but is not used in the active code
path (`Robot.py:313-316`).

### Kinematics

Unicycle model, integrated once per `--sim_time_step` (default 0.225 s) in
`Robot.update` (`Robot.py:298`):

```
θ ← (θ + ω·dt) mod 2π
x ← x + cos(θ)·v·dt
y ← y + sin(θ)·v·dt
```

Robot body: width 0.35 m, i.e. radius 0.175 m (`Robot.py:81-83`). At maximum
linear velocity a step moves the robot ≈ 0.04 m.

## Termination

Checked per robot per step in `Simulation.update`
(`Environment/Simulation.py:123-145`):

| Condition | Test | Semantics |
| --- | --- | --- |
| Collision | min distance to any wall/circle/robot ≤ robot radius | terminal |
| Goal reached | distance to station center + robot radius < 1.2 × station radius (`Robot.collideWithTargetStationCircular`) | terminal |
| Timeout | episode step budget (`--steps`) exhausted | truncation |

Collision and arrival are true terminal states. Running out of time is a
**truncation**: the transition is stored as not-done so the value of the
successor state is bootstrapped during the PPO update
(`Environment/Environment.py:191-194`). All three deactivate the robot for the
rest of the episode.
