# PPO Algorithm

This document assumes familiarity with PPO and focuses on what *this*
implementation does. Everything lives in `PPO/Algorithm.py` unless noted.

**One-paragraph recap.** PPO collects rollouts with the current policy,
estimates advantages, and then takes several gradient epochs on the clipped
surrogate objective
`L = E[min(r·Â, clip(r, 1−ε, 1+ε)·Â)]` with `r = π_new(a|s) / π_old(a|s)`,
which keeps the new policy close to the data-collecting one without an explicit
trust-region constraint.

## Network architecture

Actor and critic are **fully separate networks** with their own optimizers;
they do not share the encoder (`Algorithm.py:82-97`). Each consists of an input
encoder (`SmallInput` by default; `BigInput` is a larger alternative selected
with `--inputspace big`) followed by a one-layer head.

### `SmallInput` encoder (`PPO/SmallInput.py`)

The four observation streams are embedded separately and concatenated:

```
lidar (4 frames × 1081 rays, frames = conv channels)
  → Conv1d 4→16, k=8, s=3 → ReLU
  → Conv1d 16→32, k=4, s=2 → ReLU
  → Conv1d 32→64, k=2, s=2 → ReLU
  → flatten → Linear → 128 features

orientation (4×2) → Linear 2→8  per frame → ReLU → flatten (32)
distance    (4×1) → Linear 1→4  per frame → ReLU → flatten (16)
velocity    (4×2) → Linear 2→4  per frame → ReLU → flatten (16)

concat (192) → Linear 192→128 → ReLU → Linear 128→128 → ReLU
```

The lidar branch's 128-feature width is deliberate: the lidar carries the
obstacle geometry and must not be squeezed through fewer features than the
trivial proprioceptive inputs (it was once 8 vs. their 64 — comment at
`SmallInput.py:59-61`).

### Heads

- **Actor** (`Algorithm.py:17-49`): `Linear 128→2` producing the *pre-squash*
  mean μ, plus a state-independent learnable `log_std` parameter (one per
  action dim) initialized to −0.5, i.e. σ ≈ 0.6 — small enough to reduce
  saturation at the [-1, 1] action bounds at the start of training.
- **Critic** (`Algorithm.py:52-79`): `Linear 128→1` state value.

### Initialization

All hidden Conv1d/Linear layers use orthogonal initialization with the ReLU
gain; output heads use orthogonal init with gain 0.1 (actor — keeps the initial
policy near-uniform) and gain 1.0 (critic). Biases start at zero
(`utils.py:11-36`).

## Action distribution: tanh-squashed Gaussian

`get_distribution` (`Algorithm.py:99-105`) builds
`Independent(TransformedDistribution(Normal(μ, σ), TanhTransform), 1)`:

- Samples are squashed through `tanh` and therefore live in (−1, 1) by
  construction — no clipping of Gaussian samples, which would distort the
  log-probabilities at the bounds.
- Log-probs include the tanh change-of-variables correction automatically.
- During the update, stored actions are clamped to ±(1 − 1e-6) before the
  implicit `atanh` so actions that sit numerically on the bound stay finite
  (`Algorithm.py:161`).
- The TanhNormal has no closed-form entropy; the **base Normal's entropy is
  used as a proxy** for the entropy bonus (`Algorithm.py:163`).
- Deterministic evaluation (`--mode test`) uses `tanh(μ)` directly
  (`act_certain`, `Algorithm.py:131-142`).

## Experience collection

`SwarmMemory` (`PPO/CoolMemory.py`) keeps one buffer per agent slot; the train
loop adds `(obs, next_obs, action, logprob, reward, done)` per active robot per
step. The update fires on **experience count** (`--update_experience`, default
16384 transitions), not on episode boundaries. Before each update,
`memory.unroll_last_episode(num_agents)` archives the in-flight partial
episodes and keeps collecting with the same robot count afterwards — an earlier
version passed 0 here and silently dropped all experience gathered between a
mid-episode update and the next reset (`Algorithm.py:351-354`).

## The update step (`PPO.update`, `Algorithm.py:335`)

In order:

1. **Reward scaling.** Rewards are divided by a running standard deviation
   (`RunningMeanStd`, the openai/baselines parallel-variance algorithm in
   `utils.py:362`), updated each time with the new batch. Rewards are scaled,
   not mean-shifted (`Algorithm.py:367-368`).
2. **GAE per episode segment.** For each stored segment, the critic values
   `V(s)` are computed, then
   `δ_t = r_t + γ·V(s_{t+1})·mask_t − V(s_t)` and
   `Â_t = δ_t + γλ·mask_t·Â_{t+1}` with `γ = 0.999`, `λ = 0.97`
   (`get_advantages`, `Algorithm.py:283-307`). Returns are `Â_t + V(s_t)`.
   - If a segment ends truncated (`mask == 1` at the end, e.g. timeout or a
     mid-episode update), the value of the last next-observation is appended as
     a **bootstrap value**; true terminals bootstrap from 0
     (`Algorithm.py:377-383`).
   - `γ = 0.999` is paired with the simulation scale: an effective horizon of
     ~1000 steps ≈ 40 m at the robot's ~0.04 m step length, which must cover
     detour lengths in the larger levels (`main.py:76`).
3. **Flatten across agents.** Advantages/returns/actions/logprobs of all
   segments are concatenated. Advantages are deliberately **not** normalized
   per segment — that would destroy the relative quality of episodes against
   each other (`Algorithm.py:389-390`).
4. **K epochs of minibatch SGD.** `--K_epochs 6` passes over the batch, split
   into `--batches 32` random minibatches (`BatchSampler` +
   `SubsetRandomSampler`):
   - Advantages are normalized **per minibatch** to zero mean / unit std
     (`Algorithm.py:418`).
   - The log-ratio `log π_new − log π_old` is clamped to ±10 before
     exponentiation so a single diverged sample cannot overflow `exp()` into
     NaN (`Algorithm.py:422`).
   - **KL early stopping:** if the approximate KL `E[log π_old − log π_new]`
     reaches `--kl_target` (default 0.02), the **entire remaining update** is
     abandoned — not just the current epoch (`Algorithm.py:427-433`;
     deliberately stricter than ROSHAN, which only breaks the current epoch's
     minibatch loop). Disable with `--kl_early_stop False`.
   - Actor loss: clipped surrogate with `--eps_clip 0.2` minus the entropy
     bonus (`--entropy_coeff 0.001`).
   - Critic loss: plain (unclipped) MSE against the GAE returns — value-loss
     clipping is skipped, following the analysis in
     [the ICLR PPO implementation-details blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
     (`Algorithm.py:441-443`).
   - Actor and critic each have their own Adam optimizer
     (`lr = 3e-4`, `betas = (0.9, 0.990)`, `eps = 1e-5`) and are stepped
     independently, each after global gradient-norm clipping at 0.5
     (`Algorithm.py:458-468`).
   - Optional cosine LR annealing, stepped once per minibatch update
     (`--lr_schedule cosine`, `--lr_tmax`).
5. **Diagnostics.** Approx-KL, clip fraction, explained variance of the value
   function, current action std and learning rate are logged to TensorBoard.
6. The memory is cleared; the next rollout starts from the updated policy.

`torch.compile` of actor and critic is available behind `--torch_compile`
(checkpoints written with it active carry an `_orig_mod.` prefix that is
stripped on save and load).

## Tricks summary

| Trick | What it does | Where |
| --- | --- | --- |
| Tanh-squashed Gaussian | bounded actions with correct log-probs instead of clipped Gaussian samples | `Algorithm.py:99-105` |
| Learnable state-independent `log_std`, init −0.5 | σ ≈ 0.6 at start, less boundary saturation | `Algorithm.py:40-41` |
| Orthogonal init, actor head gain 0.1 | near-uniform initial policy | `utils.py:11-36` |
| Inverse-depth lidar encoding | bounded (0, 1] input, resolution concentrated near the robot, no max range | `Robot.py:492-494` |
| Lidar branch width 128 | geometry channel not bottlenecked below trivial inputs | `SmallInput.py:59-62` |
| Raw (not dt-scaled) velocity observations | observation decoupled from `--sim_time_step` | `Robot.py:341-344` |
| Reward scaling by running std | stabilizes value targets across levels | `Algorithm.py:367-368` |
| Truncation bootstrapping | timeouts and mid-episode updates bootstrap `V(s_next)` instead of treating the cut as terminal | `Algorithm.py:377-383`, `Environment.py:191-194` |
| Per-minibatch advantage normalization | preserves cross-episode advantage scale until the last moment | `Algorithm.py:389-390, 418` |
| Log-ratio clamp ±10 | one diverged sample cannot NaN the update | `Algorithm.py:422` |
| KL early stop (whole update) | hard brake on policy divergence | `Algorithm.py:427-433` |
| Gradient norm clipping 0.5 | per-network global-norm clip | `Algorithm.py:461, 467` |
| Mid-episode update without data loss | partial episodes archived, collection continues | `Algorithm.py:351-354` |
| Long horizon γ = 0.999 | matches detour lengths at the robot's step scale | `main.py:76` |
| Geodesic reward shaping | see [04 — Reward Function](04-reward-function.md) | `Environment.py`, `FlowField.py` |
