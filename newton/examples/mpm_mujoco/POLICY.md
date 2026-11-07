# G1 policy: jqur5jhm

`example_mpm_g1_policy.py` and `example_mujoco_mpm_coupled_solver.py` use the final `model_19999.pt` checkpoint from
[W&B run jqur5jhm](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/jqur5jhm).
Its actor has 937 inputs and 29 outputs. The exported inference model contains
only the deterministic actor; it does not sample the training action distribution.

The input order is:

1. Base angular velocity: 10 samples × 3 values, scaled by 0.25.
2. Projected gravity: 10 samples × 3 values.
3. Joint positions relative to the nominal pose: 10 samples × 29 values.
4. Joint velocities: 10 samples × 29 values, scaled by 0.05.
5. Previous actions: 10 samples × 29 values.
6. Current command: 7 values, with **no command history**.

Each history is ordered oldest to newest. The command is
`[vx, vy, yaw_rate, heading_x, heading_y, heading_z, heading_w]`.
The local command generator uses WXYZ internally; the example converts its heading
to XYZW before passing it to this policy. Observation noise is disabled for playback.
The nominal joint pose, gains, armatures, effort limits, and action scale are
recorded in `config/g1_jqur5jhm.yaml` from the training run. Imported actuated joint
names must match that configuration.

## Download and export once

From the Newton repository root, with W&B access to the run:

```bash
uv run --with wandb python -m newton.examples.mpm_mujoco.utils.export_policy
```

The output is `newton/examples/mpm_mujoco/policies/jqur5jhm.pt`.
Weights are cached locally and excluded from Git. A previously downloaded
checkpoint can be used without W&B:

```bash
uv run python -m newton.examples.mpm_mujoco.utils.export_policy \
  --checkpoint /path/to/model_19999.pt
```

## Run

```bash
uv run python newton/examples/mpm_mujoco/example_mpm_g1_policy.py \
  --viewer rtx --sand-type poppy --episode-length-s 13 --max-trial 3
```

The example uses `config/g1_jqur5jhm.yaml` next to the example by default.
`--config` selects another configuration; its `policy_path` is relative to that
configuration file. `--policy` overrides the exported model path. The previous
1,000-input policy with command history is incompatible with this input layout.

## Solver coupling

The coupled example shares the terrain, material presets, robot, policy,
observations, and rendering with the direct-impulse example:

```bash
uv run python newton/examples/mpm_mujoco/example_mujoco_mpm_coupled_solver.py \
  --viewer rtx --sand-type poppy --episode-length-s 13 --max-trial 3
```

Both use a 4 × 2 × 0.3 m particle bed, 0.02 m voxels, and 593,750 particles
with the default sampling. Policy and rendering run at 50 Hz, MPM at 200 Hz,
and MuJoCo at 400 Hz. Both use the selected material preset and the same MPM
and MuJoCo solver settings.

`example_mpm_g1_policy.py` directly collects MPM impulses and applies them as
rigid-body forces. `example_mujoco_mpm_coupled_solver.py` instead uses
`SolverCoupledProxy` with one lagged proxy pass per MPM step. MuJoCo owns
the robot and rigid support shapes; MPM owns the particles and sand support
shapes. Robot bodies marked for particle collision are passed to MPM as
proxies. Proxy mass scaling uses `coupling_relaxation` from the policy YAML
(default 26.6), while the actual robot mass remains unchanged. Episode resets
clear both MPM history and the coupled solver's feedback.

These examples match their inputs; their trajectories need not match exactly
because the coupling algorithms differ.
