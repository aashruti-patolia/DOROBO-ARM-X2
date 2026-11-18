# DOROBO-ARM-X2

The OpenArm v2.0 dual-arm robot learns to pick a ball off a table and drop it into a basket, in MuJoCo. It finds both objects with its head camera. The delicate skills are learned with PPO reinforcement learning, starting from scripted base knowledge; the rest is classical scripted control.

```
python main.py            # the whole pipeline: check -> teacher -> pretrain -> train -> evaluate -> demo
```

## What is scripted and what is learned

The pipeline is **hybrid**. Robust, well-understood parts are scripted. The parts that need feel and adaptation are learned: first by copying a scripted teacher, then improved by reward.

| Part | How | Where |
|---|---|---|
| Physics, joint servos, inverse kinematics, Cartesian control | scripted | `env/controller.py` |
| Safety limits: workspace and per-skill safety boxes, joint-step and rotation-rate limits, idle arm parked | scripted | `env/controller.py`, `env/pick_place_env.py` |
| Locating the ball and basket with the head camera (RGB + depth) | scripted perception | `perception/camera.py` |
| Choosing the arm (the one on the ball's side) | scripted | `env/pick_place_env.py` |
| Free-space moves: to hover above the ball, carrying the ball to above the basket, retreating after release. Each is one IK solution per waypoint plus a smooth joint-space S-curve, through the servos. | scripted | `env/pick_place_env.py` |
| Base knowledge for the learned skills (teacher → imitation, fading during PPO) | scripted teacher, then learned | `env/teacher.py`, `rl/pretrain.py` |
| **GRASP**: final approach, alignment, closing, lifting | **learned (PPO)** | `rl/` |
| **PLACE**: fine positioning over the basket, holding steady, timing the release | **learned (PPO)** | `rl/` |

One episode as the robot experiences it:

```
LOCATE (camera) -> TRANSIT (scripted) -> GRASP (learned) -> TRANSPORT (scripted) -> PLACE (learned) -> RETREAT -> success / miss / failure
```

### Base knowledge: teacher, then imitation, then improvement by reward

Learning GRASP and PLACE from nothing takes PPO millions of steps: an untrained policy just wiggles above the ball. So the robot starts with **base knowledge**:

1. **Scripted teacher** (`env/teacher.py`): a classical controller for GRASP and PLACE that uses only what the robot senses (camera estimates, finger angle, gripper force).
   - GRASP: get over the ball → descend while centring → close → lift.
   - PLACE: get the ball over the basket → steady → open.

   It succeeds in about 90 % of episodes at the basic stage and about 65 % at full difficulty.
2. **Imitation** (`rl/pretrain.py`): the teacher plays a few hundred episodes, with small random action noise so recoveries are covered. The neural policy is trained to copy it (behaviour cloning), and the critic learns the value of those episodes.
3. **PPO fine-tuning** (`rl/ppo.py`): training starts from the copied policy.
   - The critic warms up first.
   - An imitation term then keeps the policy close to the teacher's action on the states the policy itself visits, with a weight that fades to zero over 1.5 M steps.
   - From then on, reward alone drives learning, so the policy can become better than the teacher: more robust to latency, the rolling ball and sensor noise.

`python main.py teacher --view` shows the teacher on its own in the MuJoCo viewer. `python main.py train --no-prior` trains from scratch instead.

### How the environment works

The scripted segments run inside `env.step()`, so PPO only sees the moments where it decides. The policy's action is a 4-number command: hand displacement dx, dy, dz (at most 1 cm per step) plus gripper opening. Its observation holds only what a real robot has: joint encoders, finger angle, gripper motor force, and the camera's estimates of the ball and basket. The simulator's ground truth is used only to compute the reward.

## The dynamic environment

**Every episode, at every curriculum stage:**
- table height 0.17–0.33 m, plus a small table shift
- ball radius ±15 %
- basket width ±15 % and depth ±25 %
- friction and contact softness
- ball-to-basket distance: newly drawn every episode, from 0.12 m up to the farthest the arm can reach
- which side the ball is on, and therefore which arm is used

**Added as training progresses (curriculum, environment difficulty only):**
- a ball that starts rolling (3–17 cm)
- random nudges to the ball
- distractor blocks
- physics randomisation: mass, friction, servo gains, damping, 0–2 ticks of command latency
- visual randomisation: lighting, camera calibration error, colour shifts, depth and pixel noise, dropped frames

| Stage | Distance | Distractors | Rolling / nudges | Physics randomisation | Visual randomisation |
|---|---|---|---|---|---|
| 0 basic | ≤ 0.20 m | 0 | – | off | mild |
| 1 moderate | ≤ 0.32 m | ≤ 1 | 20 % | mild | medium |
| 2 full | full reach | ≤ 2 | 40 % | full + latency | full |

Training moves to the next stage when the success rate of full episodes reaches 60 %.

## Perception (head camera)

The ZED-style head camera is rendered at 192×108 and processed as follows:

1. **Colour segmentation** on chromaticity finds the red ball, the blue basket and the neutral table.
2. **Depth** is measured for those pixels by ray casting from the camera, with a distance-dependent noise model.
3. **Back-projection to 3-D** uses the *nominal* calibration, so camera pose jitter shows up as a real calibration error.
4. **Estimates:**
   - Ball: centre and size.
   - Basket: rim centre and size.
   - Table: height.
5. **Tracking:** velocity is estimated from successive frames. When the hand hides the ball, even partly, the last estimate is kept.

Typical errors are about 3 mm for the ball at the basic stage and about 6 mm with full randomisation.

## Reward

The reward uses staged **potential-based shaping**:

`r = outcome + γ·Φ(s') − Φ(s) − penalties`

- **Φ (progress):**
  - reaching the ball: 0–1
  - holding and lifting it: 2–3
  - aligning it over the basket: 3–4
- **Outcome:**
  - +10 when the released ball rests inside the upright basket
  - −2 when it lands outside
  - −2 when a skill runs out of its step budget. That counts as a failed attempt and ends the episode, so "waiting out the clock" is never safer than trying.
  - −5 if the basket is knocked over or the ball leaves the table
- **Penalties:** touching the basket or distractors, pushing against the safety box, jerky actions, a small time cost.

Potential shaping rewards progress but never pays for hovering in a "good" state.

## Install

```
pip install -r requirements.txt
```

Tested with Python 3.11 on Windows 11, CPU only. Rendering uses MuJoCo's offscreen OpenGL renderer.

## Commands

| Command | What it does |
|---|---|
| `python main.py` | Full pipeline: check → teacher → pretrain → train → evaluate → demo. |
| `python main.py all --quick` | The same pipeline shrunk to a few minutes, as a smoke test. |
| `python main.py check` | Checks assets and reachability maps, camera accuracy per stage, environment speed and the parallel environments. |
| `python main.py teacher` | Teacher success per curriculum stage (`teacher.json`). Add `--view` to watch it. |
| `python main.py pretrain` | Teacher demonstrations + behaviour cloning → `checkpoints/pretrained.pt` and `pretrain.json`. |
| `python main.py train` | PPO fine-tuning from `pretrained.pt` (runs `pretrain` first if needed). Options: `--timesteps N`, `--envs K`, `--resume outputs/main/checkpoints/latest.pt`, `--no-prior`. |
| `python main.py eval` | Evaluates the best checkpoint at full difficulty and writes `eval.json`. |
| `python main.py demo` | Writes a GIF (side view + head camera). Add `--view` for the live MuJoCo viewer. |
| `python main.py watch` | Live MuJoCo viewer that follows a training run: run it in a second terminal while `train` runs. It reloads `latest.pt` whenever training saves it. |
| `python main.py test` | Runs the tests (`--quick` skips the slow ones). |
| `python main.py maps` | Rebuilds the reachability maps after changing the robot, the table or the hover heights. |

Common options:
- `--run-name NAME`: outputs go to `outputs/NAME/`.
- `--seed S`: random seed.
- `--set section.field=value`: overrides any config value, for example `--set ppo.lr=1e-4 --set prior.demo_episodes=1000`.

Every tunable lives in `dual_arm_rl/config.py`.

## Outputs

```
outputs/<run-name>/
  config.json              exact configuration of the run
  log.txt                  console log
  teacher.json             teacher success per stage (the base knowledge on its own)
  demos.npz                teacher demonstrations
  pretrain.json            imitation report (losses, copied policy's success)
  tb/                      TensorBoard logs
  checkpoints/pretrained.pt  the imitation policy (training starts here)
  checkpoints/latest.pt    saved periodically (resume from it)
  checkpoints/best.pt      best success at full difficulty
  eval.json                success rate, outcomes, breakdown by table height / ball size / basket size / distance / rolling / arm
  demo.gif                 a few episodes, side view + head camera
```

Watch training with `tensorboard --logdir outputs`. The most useful charts:

| Chart | What it shows |
|---|---|
| `progress/reached_*` | Fraction of episodes that got as far as grasped, lifted, released and success. The bottleneck stage is the first one that stays low. |
| `charts/full_episode_success` | Success of complete episodes. |
| `prior/bc_coef`, `prior/teacher_mse` | Weight of the teacher term (fades to 0) and how far the policy has moved from the teacher. |
| `pretrain/*` | Imitation loss curves and the copied policy's success. |
| `curriculum/stage` | Current curriculum stage. |
| `eval/success_rate` | Deterministic success at full difficulty. |
| `perception/ball_error_mm` | Camera accuracy during training. |

## Training time

On a 12-thread CPU with 8 parallel environments, training runs at roughly 100–250 decisions per second. The default 5 M decisions is therefore an overnight-plus run, though starting from the imitation policy gives success from the beginning.

`best.pt` is saved throughout. `Ctrl+C` stops cleanly, and `--resume` continues.

## Project layout

```
main.py                         single entry point
dual_arm_rl/
  config.py                     all settings
  assets/                       robot + scene XML, meshes, reachability maps
  env/
    scene.py                    world generation, randomisation, measurements
    controller.py               IK, Cartesian controller, joint-space moves, safety limits
    teacher.py                  scripted teacher for GRASP and PLACE (base knowledge)
    reward.py                   potential-based reward
    pick_place_env.py           Gymnasium environment + scripted supervisor
  perception/camera.py          head-camera detection and tracking
  rl/
    network.py                  actor-critic MLP
    normalizer.py               running observation / reward normalisation
    pretrain.py                 teacher demonstrations + behaviour cloning
    ppo.py                      vectorised PPO trainer with curriculum and fading teacher guidance
    evaluate.py                 evaluation report, demo, viewer, teacher evaluation
  tools/compute_workspace.py    reachability maps
tests/                          pytest suite (oracle.py is a test-only feasibility check)
```

## Tests

`python main.py test` runs the suite:
- scenario geometry and collisions
- randomisation never compounding
- camera accuracy and tracking
- the controller: safety limits and smooth joint-space moves
- the reward's potential-based property
- success and failure rules
- an environment interface test showing the policy never sees the true ball position
- the teacher: uses only sensed information and succeeds at every stage
- the full base-knowledge chain: demonstrations, imitation and fine-tuning in which the guidance fades
- a test-only privileged "oracle" that completes the task through the real pipeline, proving it is solvable at every curriculum stage
- a tiny end-to-end training run with checkpoint save, reload and resume
