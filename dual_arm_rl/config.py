"""
Every tunable of the pipeline, in one place.

The pipeline is a *hybrid*:

* scripted (never learned): physics and actuation, the Cartesian/IK controller,
  safety limits, the free-space moves (transit to the ball, transport to the
  basket, retreat) and the supervisor that sequences them;
* learned by PPO: the two delicate skills, GRASP (final approach, alignment,
  closing, lifting) and PLACE (fine positioning over the basket, release).

Values are plain dataclasses. ``Config`` bundles them, round-trips through a
dict (it is stored inside every checkpoint) and accepts ``section.field=value``
overrides from the command line (``python main.py train --set ppo.lr=1e-4``).
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all resolved from the package, so commands work from any folder)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = PACKAGE_DIR / "assets"
MODEL_XML = ASSETS_DIR / "openarm_pick_and_place.xml"
WORKSPACE_MAPS = ASSETS_DIR / "workspace_maps.npz"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------
@dataclass
class SceneConfig:
    """What the world looks like and how it varies.

    Ball size, basket size, table height and ball-to-basket distance vary in
    EVERY episode, at every curriculum stage.
    """
    table_z: float = 0.25                      # nominal table-top height [m]
    table_z_halfrange: float = 0.08            # -> 0.17 ... 0.33 m
    table_shift: float = 0.02                  # +/- table shift in X and Y [m]
    ball_radius_scale: tuple = (0.85, 1.15)    # x nominal 22 mm
    box_half_scale: tuple = (0.85, 1.15)       # x nominal 45 mm half-width
    box_depth_scale: tuple = (0.75, 1.25)      # x nominal 60 mm wall height
    table_friction_scale: tuple = (0.7, 1.3)
    box_friction_scale: tuple = (0.7, 1.3)
    ball_solref_scale: tuple = (0.6, 1.6)      # >1 = softer contact
    min_distance: float = 0.12                 # ball-to-basket distance lower bound [m]
    distance_tolerance: float = 0.02           # how closely the drawn distance is matched [m]
    midline_band: float = 0.03                 # no balls within this of y=0 (camera picks the arm by side)
    push_speed: tuple = (0.05, 0.20)           # random nudge to a resting ball [m/s]
    roll_distance: tuple = (0.05, 0.15)        # how far a rolling ball travels before stopping [m]
    max_carry_tilt_deg: float = 60.0           # basket cells need the hand within this of vertical
    max_distractors_in_xml: int = 3


# ---------------------------------------------------------------------------
# Camera perception
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    camera: str = "openarm_head_camera_center"
    width: int = 192                           # native 16:9 aspect of the ZED (91 deg horizontal FOV)
    height: int = 108
    every_n_ticks: int = 2                     # 25 Hz control -> 12.5 Hz camera (learned skills)
    scripted_every_n_ticks: int = 4            # 6 Hz while the scripted moves run (enough to track)
    min_ball_pixels: int = 6
    min_box_pixels: int = 15
    # Colour segmentation on chromaticity (r, g, b divided by their sum)
    red_chroma: float = 0.50                   # ball: r/(r+g+b) above this ...
    red_min_value: int = 40                    # ... and r above this (0-255)
    blue_chroma: float = 0.52                  # basket: b/(r+g+b) above this ...
    blue_min_value: int = 50
    neutral_max_chroma: float = 0.38           # table: no channel above this share
    # Sensor model (scaled by the curriculum's visual level)
    depth_noise_base: float = 0.0005           # depth std = base + k * depth^2 [m]
    depth_noise_quad: float = 0.002
    rgb_noise: float = 6.0                     # pixel noise std (0-255)
    frame_drop_prob: float = 0.03              # chance a camera frame is lost
    velocity_smoothing: float = 0.5            # low-pass factor on the ball velocity estimate
    occlusion_ratio: float = 0.55              # fewer pixels than this share of the expected disc = partly hidden
    max_age_ticks: int = 25                    # age is reported relative to this
    visibility_margin_px: int = 8              # objects are only placed this far inside the image


# ---------------------------------------------------------------------------
# Control, safety limits and the scripted free-space moves
# ---------------------------------------------------------------------------
@dataclass
class ControlConfig:
    substeps: int = 20                         # 20 x 2 ms physics = 25 Hz control
    policy_max_step: float = 0.010             # policy hand move per tick at |a| = 1 [m]
    target_lead_max: float = 0.03              # controller target never leads the hand by more [m]
    policy_windup: float = 0.015               # learned skills: commanded target stays this close to the hand [m]
    ik_iterations: int = 8                     # per control tick (warm-started from the current joints)
    ik_orientation_weight: float = 0.6         # weight of "fingers down" vs position in IK
    max_joint_step: float = 0.15               # servo setpoint never leads a joint by more [rad]
    max_rot_step: float = 0.08                 # hand orientation goal change per tick [rad]
    gripper_open: float = 0.65                 # finger joint angle when open [rad]
    gripper_closed: float = 0.05               # finger joint angle when fully closed [rad]
    # Scripted free-space moves
    scripted_speed: float = 0.50               # (Cartesian path speed, retreat) [m/s]
    carry_speed: float = 0.30                  # with the ball in hand [m/s]
    scripted_joint_speed: float = 1.5          # peak joint speed of scripted joint-space moves [rad/s]
    carry_joint_speed: float = 1.0             # ... while carrying the ball [rad/s]
    carry_waypoints: int = 4                   # IK waypoints along the carry (keeps the hand upright)
    replan_distance: float = 0.015             # re-plan a transit when the tracked ball moved this far [m]
    scripted_tolerance: float = 0.008          # waypoint reached within this [m]
    scripted_rot_tolerance: float = 0.12       # ... and the hand orientation within this [rad]
    scripted_max_ticks: int = 200              # per scripted move
    safe_height: float = 0.14                  # travel height above the table top [m]
    pregrasp_hover: float = 0.06               # pinch point this far above the ball centre at GRASP start
    preplace_hover: float = 0.08               # pinch point this far above the basket rim at PLACE start
    retreat_height: float = 0.06               # lift after release [m]
    settle_max_ticks: int = 60                 # wait for a released ball to settle (2.4 s)
    # Safety boxes the learned skills move inside (pinch point; xy half-widths)
    grasp_box_half_xy: float = 0.07
    grasp_box_up: float = 0.08                 # above the pre-grasp hover point
    fingertip_below_pinch: float = 0.013       # fingertips are this far below the pinch point
    table_margin: float = 0.003                # fingertips never commanded below table + this
    place_box_half_xy: float = 0.06
    place_box_min_above_rim: float = 0.02      # pinch point stays this far above the rim
    place_box_max_above_rim: float = 0.16
    # Skill limits (policy decisions)
    grasp_max_steps: int = 100
    place_max_steps: int = 60
    episode_max_steps: int = 250               # across retries
    # Grasp / release detection from robot sensors (finger angle, gripper force)
    grasp_close_cmd: float = 0.35              # gripper command below this counts as "closing"
    release_open_cmd: float = 0.65             # gripper command above this counts as "releasing"
    held_min_angle: float = 0.115              # finger stalls above this when something is in the hand [rad]
    held_max_finger_speed: float = 0.15        # ... and has stopped moving [rad/s]
    held_min_force: float = 0.8                # ... while the motor squeezes [N]
    lift_success: float = 0.03                 # ball lifted this far above its resting height [m]
    ball_near_hand: float = 0.04               # camera: ball within this of the pinch point [m]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    success_bonus: float = 10.0                # ball released and resting inside the upright basket
    miss_penalty: float = 2.0                  # released ball ended up outside the basket
    timeout_penalty: float = 2.0               # a skill ran out of its step budget (terminal)
    failure_penalty: float = 5.0               # basket tipped over / ball off the table
    time_penalty: float = 0.01                 # per policy step
    box_contact_penalty: float = 0.5           # per step the arm/gripper touches the basket
    distractor_contact_penalty: float = 0.2    # per step it touches a distractor block
    safety_push_penalty: float = 0.05          # per step the policy pushes against its safety box
    action_rate_penalty: float = 0.005         # x ||a_t - a_(t-1)||^2
    reach_scale: float = 10.0                  # tanh steepness of the reach term [1/m]
    align_scale: float = 10.0                  # tanh steepness of the basket alignment term [1/m]
    box_tip_deg: float = 30.0                  # basket tipped beyond this = failure
    box_upright_deg: float = 15.0              # ball only counts inside a basket within this of upright
    settled_speed: float = 0.10                # ball slower than this inside the basket ...
    settled_ticks: int = 5                     # ... for this many ticks = success


# ---------------------------------------------------------------------------
# Curriculum: environment difficulty only (never actions)
# ---------------------------------------------------------------------------
@dataclass
class CurriculumStage:
    name: str
    max_distance: float        # ball-to-basket distance upper bound [m] (inf = as far as reachable)
    max_distractors: int
    push_prob: float           # per-tick chance a resting ball is nudged before the grasp
    roll_prob: float           # chance the ball starts the episode rolling
    physics_level: float       # 0..1: mass, friction, servo gains, damping, latency
    visual_level: float        # 0..1: lighting, camera pose error, colour shifts, depth/pixel noise
    sensor_level: float        # 0..1: encoder noise, dropped camera frames
    place_start_prob: float    # chance an episode starts at PLACE with the ball already in hand


def _default_stages():
    return [
        CurriculumStage("basic", 0.20, 0, 0.000, 0.0, 0.0, 0.3, 0.3, 0.5),
        CurriculumStage("moderate", 0.32, 1, 0.005, 0.2, 0.5, 0.6, 0.6, 0.3),
        CurriculumStage("full", float("inf"), 2, 0.010, 0.4, 1.0, 1.0, 1.0, 0.2),
    ]


@dataclass
class CurriculumConfig:
    stages: list = field(default_factory=_default_stages)
    start_stage: int = 0
    advance_success: float = 0.6               # success rate (full episodes) needed to advance
    window: int = 200                          # ... measured over this many full episodes
    min_episodes: int = 100                    # ... and at least this many at the current stage
    # Training shortcut (initial state only): this share of GRASP episodes starts
    # with the hand already at the pre-grasp hover instead of running the
    # scripted transit (which teaches nothing and costs ~70 control ticks).
    # Evaluation and demos always run the full pipeline.
    hover_start_prob: float = 0.8


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
@dataclass
class PriorConfig:
    """Base knowledge: a scripted teacher (camera + gripper sensors only)
    that the policy first imitates, then outgrows with PPO."""
    enabled: bool = True
    # teacher behaviour
    teacher_gain: float = 0.4                  # action = gain * error / max step (clipped to [-1, 1])
    teacher_damping: float = 0.5               # minus damping * (hand velocity * tick) / max step
    coarse_align: float = 0.015                # descend only once within this in XY [m]
    align_tol: float = 0.006                   # XY alignment before closing / releasing [m]
    steady_speed: float = 0.03                 # release only when the hand moves slower than this [m/s]
    height_tol: float = 0.004                  # pinch point this close to the ball centre -> close [m]
    place_lower: float = 0.08                  # PLACE: release with the pinch point at rim + this [m]
                                               # (= the pre-place hover, reachable everywhere per the maps)
    # demonstrations
    demo_episodes: int = 600
    demo_noise: float = 0.2                    # DART: executed = teacher + N(0, noise); label = teacher
    demo_stage_mix: tuple = (0.6, 0.25, 0.15)  # share of demo episodes per curriculum stage
    demo_envs: int = 8
    # behaviour cloning
    bc_epochs: int = 30
    bc_lr: float = 1e-3
    bc_batch: int = 512
    init_log_std: float = -1.2                 # exploration std 0.30 after imitation
    # PPO fine-tuning with fading teacher guidance (DAgger-style labels on visited states)
    bc_coef: float = 1.0                       # weight of the teacher-imitation loss at the start ...
    bc_decay_steps: int = 1_500_000            # ... decaying linearly to 0 over this many decisions
    value_warmup_updates: int = 5              # critic-only updates before the policy moves
    finetune_lr: float = 1e-4
    finetune_target_kl: float = 0.015


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000_000           # policy decisions (scripted ticks are not counted)
    n_envs: int = 8
    n_steps: int = 256                         # per env per rollout -> batch 2048
    n_epochs: int = 10
    minibatch_size: int = 256
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    hidden_sizes: tuple = (256, 256)
    init_log_std: float = -0.5
    obs_clip: float = 10.0
    normalize_reward: bool = True
    device: str = "cpu"                        # this machine has no GPU


@dataclass
class EvalConfig:
    interval_steps: int = 100_000              # evaluate during training every this many steps
    train_episodes: int = 20                   # episodes per in-training evaluation
    final_episodes: int = 100                  # episodes for `main.py eval`
    demo_episodes: int = 3
    demo_fps: int = 25


@dataclass
class RunConfig:
    seed: int = 1
    run_name: str = "main"
    checkpoint_every_updates: int = 10


@dataclass
class Config:
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for section in fields(cls):
            if section.name not in d:
                continue
            sub = getattr(cfg, section.name)
            for f in fields(sub):
                if f.name not in d[section.name]:
                    continue
                value = d[section.name][f.name]
                if section.name == "curriculum" and f.name == "stages":
                    value = [CurriculumStage(**s) if isinstance(s, dict) else s for s in value]
                elif isinstance(getattr(sub, f.name), tuple):
                    value = tuple(value)
                setattr(sub, f.name, value)
        return cfg

    # -- command-line overrides ----------------------------------------------
    def apply_overrides(self, items: list[str] | None) -> "Config":
        """Apply ``section.field=value`` strings, e.g. ``ppo.lr=1e-4``."""
        for item in items or []:
            if "=" not in item or "." not in item.split("=", 1)[0]:
                raise ValueError(f"Override must look like section.field=value, got {item!r}")
            key, raw = item.split("=", 1)
            section, name = key.split(".", 1)
            if not hasattr(self, section) or not is_dataclass(getattr(self, section)):
                raise ValueError(f"Unknown config section {section!r}")
            sub = getattr(self, section)
            if not hasattr(sub, name):
                valid = [f.name for f in fields(sub)]
                raise ValueError(f"Unknown field {key!r}; {section} has {valid}")
            current = getattr(sub, name)
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
            if isinstance(current, bool):
                value = str(raw).lower() in ("1", "true", "yes") if isinstance(value, str) else bool(value)
            elif isinstance(current, float):
                value = float(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, tuple):
                value = tuple(value)
            setattr(sub, name, value)
        return self


def quick_config(cfg: Config) -> Config:
    """Shrink everything for a few-minute smoke run of the whole pipeline."""
    cfg.ppo.total_timesteps = 20_000
    cfg.ppo.n_envs = 4
    cfg.ppo.n_steps = 128
    cfg.ppo.n_epochs = 4
    cfg.eval.interval_steps = 10_000
    cfg.eval.train_episodes = 4
    cfg.eval.final_episodes = 10
    cfg.eval.demo_episodes = 1
    cfg.run.run_name = "quick"
    cfg.run.checkpoint_every_updates = 5
    cfg.prior.demo_episodes = 40
    cfg.prior.demo_envs = 4
    cfg.prior.bc_epochs = 5
    cfg.prior.bc_decay_steps = 20_000
    cfg.prior.value_warmup_updates = 2
    return cfg
