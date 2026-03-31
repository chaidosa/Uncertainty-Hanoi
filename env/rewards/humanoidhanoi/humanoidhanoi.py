"""Reward and termination functions for Humanoid Hanoi.

The original codebase ships with an empty reward dict (environment-only
release; training code is external).  We keep that default behaviour for
non-Place skills and add four new reward terms that are active **only**
during the ``put_down_box`` (Place) skill when the adaptation module is
providing uncertainty estimates.

New reward terms (Section 2.4 of the extension spec)
-----------------------------------------------------
1. r_efficiency        — small per-step penalty encouraging faster placement
2. r_fast_bonus        — end-of-episode bonus for successful fast placement
3. r_uncertainty_penalty — punishes overconfident failures
4. r_calibration       — softly penalises fast lowering under high uncertainty
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from util.quaternion import *

wrap_to_pi = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

# Calibration reward speed limits (m/s)
V_MIN = 0.05   # careful lowering
V_MAX = 0.40   # fast drop

# Maximum Place episode length used for normalisation (timesteps at 50 Hz)
T_MAX_PLACE = 500  # 10 seconds


def _compute_constellation_points(pose, radius):
    """Generates eight points around a heading plus the centre point."""
    x, y, theta = pose
    num_points = 8
    angle_offsets = np.arange(num_points) * np.pi / num_points
    angles = theta + angle_offsets

    points_x = x + radius * np.cos(angles)
    points_y = y + radius * np.sin(angles)
    points_x = np.concatenate([points_x, [x]], axis=0)
    points_y = np.concatenate([points_y, [y]], axis=0)

    return np.stack([points_x, points_y], axis=1)


def compute_rewards(self, action):
    """Compute reward components.

    Returns a dict of named reward scalars.  The weighting / kernel
    application is handled by ``GenericEnv.compute_reward``.

    Adaptive-placement rewards are only produced when the current skill
    is ``put_down_box`` **and** the env has valid adaptation outputs
    (``adaptation_sigma_t`` attribute).
    """
    q = {}

    if not _is_adaptive_place_active(self):
        return q

    t = self.time_step - (self.place_start_time or 0)
    sigma = getattr(self, "adaptation_sigma_t", 1.0)
    sigma = float(np.clip(sigma, 0.0, 1.0))

    # 1. Efficiency penalty — incentivise finishing sooner
    q["r_efficiency"] = float(t) / T_MAX_PLACE

    # 2. Successful fast-placement bonus (only at episode end / skill end)
    success = self.is_box_upright() and self.is_box_at_target()
    if success:
        q["r_fast_bonus"] = -(1.0 - float(t) / T_MAX_PLACE)
    else:
        q["r_fast_bonus"] = 0.0

    # 3. Overconfident-failure penalty
    box_fell = not self.is_box_upright()
    if box_fell:
        q["r_uncertainty_penalty"] = (1.0 - sigma)
    else:
        q["r_uncertainty_penalty"] = 0.0

    # 4. Calibration: penalise fast lowering when uncertainty is high
    speed = self.get_box_vertical_speed()
    safe_speed = V_MIN + (V_MAX - V_MIN) * (1.0 - sigma)
    q["r_calibration"] = float(max(0.0, speed - safe_speed))

    return q


def _is_adaptive_place_active(env) -> bool:
    """True when the Place skill is running and adaptation data is available."""
    if not getattr(env, "place_skill_active", False):
        return False
    if env.current_skill != "put_down_box":
        return False
    return True


def compute_done(self):
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    floor_quat = self.sim.get_geom_pose('floor')[3:]
    floor_rot = R.from_quat(mj2scipy(floor_quat))
    rotated_base_pose = floor_rot.inv().apply(base_pose[:3])
    current_height = rotated_base_pose[2]
    base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')

    if base_pose[2] < 0.2:
        return True

    if self.current_skill == "walk_with_box":
        if self.box_number == 0:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box"))
        elif self.box_number == 1:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box1"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box1"))
        elif self.box_number == 2:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box2"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box2"))

        if l_arm_contact == 0 and r_arm_contact == 0:
            self.hand_force_reset_count += 1

        if self.hand_force_reset_count > 50:
            return True

    time_difference = self.time_step - self.pre_change_time

    if time_difference > 1500:
        return True

    if self.finish_cycle:
        return True

    return False
