import torch
import math
import numpy as np
import time
import random
import json
import mujoco as mj

from util.quaternion import *
from env.genericenv import GenericEnv
from util.colors import FAIL, ENDC
from env.util.interactivecommandsmixin import BoxManipulationCmd
from scipy.spatial.transform import Rotation as R

wrap_to_pi = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

# ---------------------------------------------------------------------------
# Observation dimensions for the adaptation module
# ---------------------------------------------------------------------------
# base_orient(4) + base_ang_vel(3) + motor_pos(20) + motor_vel(20)
# + hand_contact_force(6) + box_pose_relative(6) = 59
ADAPTATION_OBS_DIM = 59

# Normalization ranges for privileged labels (box physical properties)
BOX_MASS_RANGE = (0.0, 3.5)       # kg
BOX_FRICTION_RANGE = (0.1, 1.0)   # sliding friction coefficient
BOX_COM_OFFSET_RANGE = (-0.05, 0.05)  # metres per axis
BOX_SIZE_RANGE = (0.1, 0.25)      # half-extents in metres

class BoxTowerOfHanoiEnv(GenericEnv, BoxManipulationCmd):

    def __init__(
        self,
        reward_name: str,
        policy_rate: int,
        simulator_type: str,
        dynamics_randomization: bool,
        state_noise: float,
        integral_action: bool = False,
        **kwargs,
    ):
    
        super().__init__(
            reward_name=reward_name,
            simulator_type=simulator_type,
            policy_rate=policy_rate,
            dynamics_randomization=dynamics_randomization,
            state_noise=state_noise,
            integral_action=integral_action,
            **kwargs,
        )

        self.initial_variables_for_delta_env()

        self.initialize_variables()

        np.random.seed(24)

    def _get_state(self):

        return np.zeros(80)

    # ------------------------------------------------------------------
    # Adaptation-module observation builder
    # ------------------------------------------------------------------

    def get_adaptation_obs(self) -> np.ndarray:
        """Build the 59-D observation vector consumed by the adaptation module.

        Layout:
          [0:4]   base orientation quaternion (wxyz)
          [4:7]   base angular velocity (IMU gyro)
          [7:27]  motor positions (20 actuated joints)
          [27:47] motor velocities (20 actuated joints)
          [47:50] left-hand contact force (3D)
          [50:53] right-hand contact force (3D)
          [53:59] current box pose relative to robot base (x, y, z, roll, pitch, yaw)
        """
        obs = np.zeros(ADAPTATION_OBS_DIM)

        # Proprioception
        obs[0:4] = self.sim.get_base_orientation()
        obs[4:7] = self.sim.data.sensor('torso/base/imu-gyro').data
        obs[7:27] = self.sim.get_motor_position()
        obs[27:47] = self.sim.get_motor_velocity()

        # Hand contact forces (3D each, from elbow bodies)
        obs[47:50] = self.sim.get_body_contact_force(self.sim.hand_body_name[0])
        obs[50:53] = self.sim.get_body_contact_force(self.sim.hand_body_name[1])

        # Box pose relative to robot base
        obs[53:59] = self._get_current_box_pose_relative()

        return obs

    def _get_current_box_pose_relative(self) -> np.ndarray:
        """Get the current box pose (xyz + rpy) in the robot base frame."""
        base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        box_idx = self._active_box_index()
        box_names = ["box", "box1", "box2"]
        box_pose = self.sim.get_body_pose(box_names[box_idx])
        rel_pose = self.sim.get_relative_pose(base_pose, box_pose)
        rel_xyz = rel_pose[:3]
        rel_rpy = R.from_quat(mj2scipy(rel_pose[3:])).as_euler('xyz')
        return np.concatenate([rel_xyz, rel_rpy])

    def _active_box_index(self) -> int:
        """Return the index (0, 1, 2) of the box currently being manipulated."""
        if self.current_hold_box_number is not None:
            return self.current_hold_box_number
        if self.box_finish_count >= 0 and self.box_finish_count < len(self.box_pick_order):
            return self.box_pick_order[self.box_finish_count]
        return 0

    # ------------------------------------------------------------------
    # Privileged ground-truth box properties (for Phase 1 training)
    # ------------------------------------------------------------------

    def get_privileged_box_properties(self) -> np.ndarray:
        """Return normalized ground-truth physical properties for the active box.

        Layout (8-D, matches latent_dim):
          [0]   mass             normalized to [0, 1]
          [1]   sliding friction normalized to [0, 1]
          [2:5] CoM offset (xyz) normalized to [-1, 1]
          [5:8] box half-extents normalized to [0, 1]
        """
        idx = self._active_box_index()
        box_body_names = ["box", "box1", "box2"]
        box_geom_names = ["box0", "box1", "box2"]

        # Mass
        raw_mass = float(self.sim.model.body(box_body_names[idx]).mass)
        norm_mass = np.clip(
            (raw_mass - BOX_MASS_RANGE[0]) / (BOX_MASS_RANGE[1] - BOX_MASS_RANGE[0]), 0, 1
        )

        # Sliding friction (first element of 3-vector)
        geom_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_GEOM, box_geom_names[idx])
        raw_fric = float(self.sim.model.geom_friction[geom_id, 0])
        norm_fric = np.clip(
            (raw_fric - BOX_FRICTION_RANGE[0]) / (BOX_FRICTION_RANGE[1] - BOX_FRICTION_RANGE[0]),
            0, 1,
        )

        # CoM offset from geometric center
        raw_ipos = self.sim.model.body(box_body_names[idx]).ipos.copy()
        norm_ipos = np.clip(
            raw_ipos / max(abs(BOX_COM_OFFSET_RANGE[0]), abs(BOX_COM_OFFSET_RANGE[1])),
            -1, 1,
        )

        # Box half-extents
        raw_size = self.sim.model.geom(box_geom_names[idx]).size.copy()
        norm_size = np.clip(
            (raw_size - BOX_SIZE_RANGE[0]) / (BOX_SIZE_RANGE[1] - BOX_SIZE_RANGE[0]),
            0, 1,
        )

        return np.concatenate([[norm_mass, norm_fric], norm_ipos, norm_size])

    # ------------------------------------------------------------------
    # Placement dynamics helpers (used by adaptive-place rewards)
    # ------------------------------------------------------------------

    def get_box_vertical_speed(self) -> float:
        """Return absolute vertical velocity of the active box (m/s)."""
        idx = self._active_box_index()
        box_body_names = ["box", "box1", "box2"]
        vel = self.sim.get_body_velocity(box_body_names[idx])
        return abs(float(vel[2]))  # z-component of linear velocity

    def is_box_upright(self, tilt_threshold_deg: float = 30.0) -> bool:
        """Check whether the active box is roughly upright."""
        idx = self._active_box_index()
        box_body_names = ["box", "box1", "box2"]
        pose = self.sim.get_body_pose(box_body_names[idx])
        rpy = R.from_quat(mj2scipy(pose[3:])).as_euler('xyz')
        tilt = np.sqrt(rpy[0] ** 2 + rpy[1] ** 2)
        return tilt < np.deg2rad(tilt_threshold_deg)

    def is_box_at_target(self, pos_threshold: float = 0.10) -> bool:
        """Check whether the active box xy-position is within threshold of target."""
        idx = self._active_box_index()
        qpos_slices = [slice(-21, -14), slice(-14, -7), slice(-7, None)]
        box_pos = self.sim.data.qpos[qpos_slices[idx]][:3]
        target_xy = self.box_target[:2] if len(self.box_target) >= 2 else np.zeros(2)
        return float(np.linalg.norm(box_pos[:2] - target_xy)) < pos_threshold

    def reset(self, interactive_evaluation=False):

        if self.round_count >= 0:
            self.log_benchmark_data()

        if self.round_count == self.total_evaluation_number:
            self.save_log_benchmark_data()

        self.reset_simulation()

        self.reset_variables()

        self.rand_target_position()

        self.rand_box_size()

        self.set_box_poses()

        self.rand_box_mass()

        self.rand_box_friction()

        self.load_plan()

        self.time_step = 0
        self.last_action = None

        self.log_init_status()

        # Freeze base (3 slides + ball) so the viewer demo does not tip without a balance policy.
        # See MujocoSim.hold(); matches Digit XML layout vs a single free joint.
        self.sim.hold()
        mj.mj_forward(self.sim.model, self.sim.data)

        return self.get_state()

    def step(self):

        if self.sim.viewer is not None:
            self.draw_markers()

        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        # Hold current motor positions (zero delta): stable stand without a learned policy.
        # Fixed offset + zero action tracks nominal pose and tends to collapse.
        self.step_simulation(np.zeros(self.robot.n_actuators), simulator_repeat_steps, integral_action=True)

        self.time_step += 1

    def get_action_mirror_indices(self):
        return self.robot.motor_mirror_indices

    def get_observation_mirror_indices(self):
        return np.zeros(80)

    def _init_interactive_key_bindings(self):

        self.input_key_dict = {}

    def _init_interactive_xbox_bindings(self):
        pass

    def _update_control_commands_dict(self):
        pass 

    @staticmethod
    def get_env_args():
        return {
            "simulator-type"     : ("box", "Which simulator to use (\"mujoco\" or \"libcassie\" or \"ar\")"),
            "policy-rate"        : (50, "Rate at which policy runs in Hz"),
            "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
            "state-noise"        : ([0,0,0,0,0,0], "Amount of noise to add to proprioceptive state."),
            "reward-name"        : ("pos_delta", "Which reward to use"),
        }

    def get_box_world_pose(self, box_finish_count = None):

        if box_finish_count is not None:
            temp_box_finish_count = self.box_finish_count
            self.box_finish_count = box_finish_count

        if self.box_pick_order[self.box_finish_count] == 2:
            box_pose = self.sim.data.qpos[-7:].copy()
            self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
            self.box_height = box_pose[2].copy()
            self.box_rotation_all = box_pose[3:].copy()
        elif self.box_pick_order[self.box_finish_count] == 1:
            box_pose = self.sim.data.qpos[-14:-7].copy()
            self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
            self.box_height = box_pose[2].copy()
            self.box_rotation_all = box_pose[3:].copy()
        elif self.box_pick_order[self.box_finish_count] == 0:
            box_pose = self.sim.data.qpos[-21:-14].copy()
            self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
            self.box_height = box_pose[2].copy()
            self.box_rotation_all = box_pose[3:].copy()

        if box_finish_count is not None:
            self.box_finish_count = temp_box_finish_count

        return box_pose

    def rand_target_position(self):

        radius = np.random.uniform(1.5, 2.5)
        self.area_radius = radius

        angle_0 = np.random.uniform(0.0, 2*np.pi)
        target_position_0 = np.array([radius * np.cos(angle_0), radius * np.sin(angle_0), 0.0])
        while True:
            angle_1 = np.random.uniform(0.0, 2*np.pi)
            target_position_1 = np.array([radius * np.cos(angle_1), radius * np.sin(angle_1), 0.0])
            if np.linalg.norm(target_position_0[:2] - target_position_1[:2]) > 0.9:
                break
        while True:
            angle_2 = np.random.uniform(0.0, 2*np.pi)
            target_position_2 = np.array([radius * np.cos(angle_2), radius * np.sin(angle_2), 0.0])
            if np.linalg.norm(target_position_0[:2] - target_position_2[:2]) > 0.9 and np.linalg.norm(target_position_1[:2] - target_position_2[:2]) > 0.9:
                break
        self.desk_position = np.array([target_position_0, target_position_1, target_position_2])
        self.desk_rotation = np.array([angle_0, angle_1, angle_2])

    def rand_box_size(self):

        box_size_0 = 0.1 + np.array([np.random.uniform(0.03, 0.045), np.random.uniform(0.03, 0.045), np.random.uniform(0.03, 0.045)])
        self.sim.set_geom_size("box0", box_size_0)
        box_size_1 = 0.1 + np.array([np.random.uniform(0.045, 0.06), np.random.uniform(0.045, 0.06), np.random.uniform(0.045, 0.06)])
        self.sim.set_geom_size("box1", box_size_1)
        box_size_2 = 0.1 + np.array([np.random.uniform(0.06, 0.075), np.random.uniform(0.06, 0.075), np.random.uniform(0.06, 0.075)])
        self.sim.set_geom_size("box2", box_size_2)

        self.box_height_list = [box_size_0[2], box_size_1[2], box_size_2[2]]
        self.box_size_list = [box_size_0, box_size_1, box_size_2]

    def rand_box_mass(self):
        box_mass_0 = 0.2 + np.random.uniform(0, 2.5)
        box_mass_1 = 0.2 + np.random.uniform(0, 2.5)
        box_mass_2 = 0.2 + np.random.uniform(0, 2.5)
        self.sim.model.body("box").mass = box_mass_0
        self.sim.model.body("box1").mass = box_mass_1
        self.sim.model.body("box2").mass = box_mass_2
        box_size = self.box_size_list[0]
        box_size_1 = self.box_size_list[1]
        box_size_2 = self.box_size_list[2]
        self.sim.model.body_inertia[self.sim.box_body_id, 0] = 0.6 * box_mass_0 * (box_size[1]**2 + box_size[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box_body_id, 1] = box_mass_0 * (box_size[0]**2 + box_size[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box_body_id, 2] = box_mass_0 * (box_size[0]**2 + box_size[1]**2) / 3
        self.sim.model.body_inertia[self.sim.box1_body_id, 0] = 0.6 * box_mass_1 * (box_size_1[1]**2 + box_size_1[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box1_body_id, 1] = box_mass_1 * (box_size_1[0]**2 + box_size_1[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box1_body_id, 2] = box_mass_1 * (box_size_1[0]**2 + box_size_1[1]**2) / 3
        self.sim.model.body_inertia[self.sim.box2_body_id, 0] = 0.6 * box_mass_2 * (box_size_2[1]**2 + box_size_2[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box2_body_id, 1] = box_mass_2 * (box_size_2[0]**2 + box_size_2[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box2_body_id, 2] = box_mass_2 * (box_size_2[0]**2 + box_size_2[1]**2) / 3

    def rand_box_friction(self):
        
        sliding_friction = np.random.uniform(0.5, 0.7)
        rolling_friction = 0.02
        spinning_friction = 0.005

        self.sim.model.geom_friction[self.sim.box_geom_id] = np.array([sliding_friction, rolling_friction, spinning_friction])

        sliding_friction = np.random.uniform(0.5, 0.7)
        self.sim.model.geom_friction[self.sim.box1_geom_id] = np.array([sliding_friction, rolling_friction, spinning_friction])

        sliding_friction = np.random.uniform(0.5, 0.7)
        self.sim.model.geom_friction[self.sim.box2_geom_id] = np.array([sliding_friction, rolling_friction, spinning_friction])

    def set_box_poses(self):

        initial_box_x = self.desk_position[0][0]
        initial_box_y = self.desk_position[0][1]

        # set box0 position
        box0_position = np.array([initial_box_x, initial_box_y, np.random.uniform(0.78, 0.9)])
        box0_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, self.desk_rotation[0]], degrees = False).as_quat())
        box0_pose = np.concatenate([box0_position, box0_quat])
        self.sim.data.qpos[-21:-14] = box0_pose
        # set box1 position
        box1_position = np.array([initial_box_x, initial_box_y, np.random.uniform(0.44, 0.5)])
        box1_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, self.desk_rotation[0]], degrees = False).as_quat())
        box1_pose = np.concatenate([box1_position, box1_quat])
        self.sim.data.qpos[-14:-7] = box1_pose
        # set box2 position
        box2_position = np.array([initial_box_x, initial_box_y, np.random.uniform(0.14, 0.15)])
        box2_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, self.desk_rotation[0]], degrees = False).as_quat())
        box2_pose = np.concatenate([box2_position, box2_quat])
        self.sim.data.qpos[-7:] = box2_pose

        # Update forward kinematics to reflect new box positions in visualization
        mj.mj_forward(self.sim.model, self.sim.data)

        # sort box position by distance
        box_positions = np.array([box0_position, box1_position, box2_position])
        box_distances = np.linalg.norm(box_positions, axis=1)
        box_order = np.argsort(box_distances)
        self.box_pick_order = box_order

    def load_plan(self):

        box_order = []
        pos_order = []
        act_order = []

        box_map = {"b1": 0, "b2": 1, "b3": 2}
        loc_map = {"l1": 0, "l2": 1, "l3": 2}


        # load hanoi forward plan
        with open("./box-world-json/hanoi.json", "r") as f:
            plan_dict = json.load(f)

            even_counter = 0

            for step in plan_dict["plan"]:
                action, args = next(iter(step.items()))

                act_order.append(action)

                if action == "pickup" or action == "unstack":
                    box = args[0]
                    box_order.append(box_map[box])
                elif action == "locomotion" and even_counter % 2 == 0:
                    loc = args[-1]
                    pos_order.append(loc_map[loc])
                
                if action == "locomotion":
                    even_counter += 1

        self.box_pick_order = box_order
        self.target_position_order = pos_order
        self.act_order = act_order

    def initialize_variables(self):

        self.time = 0

        self.target_marker = None
        self.target_marker_inside = None

        self.area_marker = None
        self.desk_0_marker = None
        self.desk_1_marker = None
        self.desk_2_marker = None
        self.label_0_marker = None
        self.label_1_marker = None
        self.label_2_marker = None
        self.arrow_0_height = 1.75
        self.arrow_1_height = 1.8
        self.arrow_2_height = 1.85
        self.arrow_0_change = 0.002
        self.arrow_1_change = 0.002
        self.arrow_2_change = -0.002
        self.target_marker_0 = None
        self.target_marker_1 = None
        self.target_marker_2 = None

        # benchmark data variables
        self.success_count = 0
        self.finish_count = 0
        self.round_count = -1
        
        self.benchmark_data = []
        self.total_evaluation_number = 100
        self.start_time = time.time()

    def reset_variables(self):

        self.pick_up_bit = False
        self.put_down_bit = False
        self.box_number = 0
        self.box_start_pose = np.array([0.45, 0.0, 0.557, 1.0, 0.0, 0.0, 0.0])
        self.box_size = np.array([0.157, 0.157, 0.157])
        self.box_target = np.array([0.4, 0.0, 0.0])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.box_pose = np.array([0.45, 0.0, 0.687, 1.0, 0.0, 0.0, 0.0])
        self.last_update = 0

        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.turn_rate = 0.0
        self.orient_add = 0.0
        self.height = 0.85
        self.stand_bit = 0
        self.stand_cmd_bit = 0
        self.stand_mode = True
        self.freeze_upper_body = True
        self.put_down_box = False

        self.time_step = 0
        self.pre_change_time = 0
        self.locomotion_timer = 0
        self.use_correct_time = False
        self.current_time = False
        self.unlock_change_mode = True
        self.only_drop_box_once = False
        self.wait_time = 350
        self.pre_stand_bit = 0
        self.hand_force_reset_count = 0

        self.yaw = -0.1
        self.box_angle_local = 0.0
        self.box_x_local = 0.0
        self.box_y_local = 0.0
        self.target_position = np.array([1.5, 1.0])
        self.target_rotation = 0.0
        self.box_target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.desk_position = np.array([[1.5, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.5, 0.0, 0.0]])
        self.desk_rotation = np.array([0.0, -np.pi/2, np.pi])
        self.area_radius = 0.0

        self.with_box = False
        self.record_time = False
        self.disable_once = False
 
        self.box_number = 0
        self.box_world_pose = np.array([0.0, 0.0, 0.0])
        self.box_height = 0.0
        self.box_world_rotation = 0.0
        self.box_rotation_all = None
        self.int_to_change_box_number = 1

        self.box_finish_count = -1
        self.box_pick_order = [0, 1, 2]
        self.target_position_order = [0, 0, 0]
        self.action_process = ["walk_without_box", "pick_up_box", "walk_with_box", "put_down_box"]
        self.action_process_index = 0
        self.current_skill = "walk_without_box"
        self.last_command = self.sim.reset_qpos[self.sim.arm_motor_position_inds].copy()

        self.stack_1_box_count = [0,1,2]
        self.stack_2_box_count = []
        self.stack_3_box_count = []
        self.current_hold_box_number = None

        self.renew = False
        
        self.finish_cycle = False
        
        self.walk_marker_change = False

        # Reset env counter variables
        self.interactive_evaluation = False
        self.traj_idx = 0

        self.last_action = None
        self.last_llc_cmd = None
        self.new_llc_cmd = None
        self.feet_air_time = np.array([0, 0])

        self.round_count += 1
        

        # benchmark data variables
        self.round_status = []
        self.stack_finish_status = []
        self.success_status = []
        self.time_status = []
        self.current_skill_status = []
        self.box_pose_status = []
        self.box_target_position_status = []
        self.box_target_rotation_status = []

        self.robot_position_status = []
        self.robot_rotation_status = []
        self.robot_target_position_status = []
        self.robot_target_rotation_status = []

        self.dr_status = []
        self.energy_status = []
        self.energy_sum = 0.0

        self.first_frame_pose = None

        # Adaptation-module tracking
        self.adaptation_z_t = np.zeros(8)
        self.adaptation_sigma_t = 1.0
        self.adaptation_z_log = []
        self.adaptation_sigma_log = []
        self.place_start_time = None
        self.place_skill_active = False
        self.prev_box_z = None

    def log_init_status(self):
        initial_dyn_params = {"damping": self.sim.get_dof_damping().copy(),
                                   "mass": self.sim.get_body_mass().copy(),
                                   "ipos": self.sim.get_body_ipos().copy(),
                                   "spring": self.sim.get_joint_stiffness().copy(),
                                   "friction": self.sim.get_geom_friction().copy(),
                                   "solref": self.sim.get_geom_solref().copy()}

        self.dr_status.append(initial_dyn_params)
        

    def log_benchmark_data(self):

        base_pose = self.sim.get_body_pose(self.sim.base_body_name)

        all_box_pose = self.get_all_box_pose()

        self.time_status.append(self.time_step)
        self.stack_finish_status.append(self.box_finish_count)
        self.current_skill_status.append(self.current_skill)
        self.box_pose_status.append(all_box_pose)
        self.box_target_position_status.append(self.box_target[:3])
        self.box_target_rotation_status.append(R.from_quat(mj2scipy(self.box_target_quat)).as_euler('xyz'))
        self.robot_position_status.append(base_pose[:3])
        self.robot_rotation_status.append(R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz'))
        self.robot_target_position_status.append(self.target_position)
        self.robot_target_rotation_status.append(self.target_rotation)
        self.energy_status.append(self.energy_sum)
        self.energy_sum = 0.0

        self.log_current_data()

    def log_current_data(self):

        benchmark_data = {
            "round_status": self.round_count,
            "success_status": self.success_count,
            "stack_finish_status": self.stack_finish_status,
            "time_status": self.time_status,
            "current_skill_status": self.current_skill_status,
            "box_pose_status": self.box_pose_status,
            "box_target_position_status": self.box_target_position_status,
            "box_target_rotation_status": self.box_target_rotation_status,
            "robot_position_status": self.robot_position_status,
            "robot_rotation_status": self.robot_rotation_status,
            "robot_target_position_status": self.robot_target_position_status,
            "robot_target_rotation_status": self.robot_target_rotation_status,
            "adaptation_z_log": self.adaptation_z_log,
            "adaptation_sigma_log": self.adaptation_sigma_log,
            "dr_status": self.dr_status,
            "energy_status": self.energy_status
        }

        self.benchmark_data.append(benchmark_data)

    def save_log_benchmark_data(self):
        #save as npz file
        filename = f"Tower_of_hanoi_benchmark/TOH_base_benchmark.npz"
        np.savez(filename, benchmark_data=self.benchmark_data)

        print()
        print(f"\nSaved benchmark data to {filename}")

        exit()

    def initial_variables_for_delta_env(self):
        # Re-initialize interactive key bindings to ensure PosDelta controls take precedence
        # over PickUpBoxCmd controls where there are conflicts.
        self._init_interactive_key_bindings()

        # Command randomization ranges
        self._x_delta_bounds = [-2.0, 2.0]
        self._y_delta_bounds = [-2.0, 2.0]
        self._yaw_delta_bounds = [-3.14, 3.14] # rad/s
        self._randomize_commands_bounds = [300, 500] # in episode length
        self._clip_commands = True
        self._clip_norm = 2.0

        self.del_x = 0
        self.del_y = 0
        self.del_yaw = 0
        self.stand_bit = 0

        self.user_del_x = 0
        self.user_del_y = 0
        self.user_del_yaw = 0
        # self.update_orient = True

        self.local_x, self.local_y, self.local_yaw = 0, 0, 0
        self.command_x, self.command_y, self.command_yaw = 0, 0, 0

        # feet air time tracking
        self.feet_air_time = np.array([0, 0]) # 2 feet
        # reward variables
        self.feet_contact_buffer = []
        self.pos_marker = None

    def get_all_box_pose(self):
        all_box_pose = []
        box_pose = self.sim.data.qpos[-21:-14].copy()
        all_box_pose.append(box_pose)
        box_pose = self.sim.data.qpos[-14:-7].copy()
        all_box_pose.append(box_pose)
        box_pose = self.sim.data.qpos[-7:].copy()
        all_box_pose.append(box_pose)
        return all_box_pose

    def draw_markers(self):

        so3 = euler2so3(z=0, x=0, y=0)
        size = [self.area_radius, self.area_radius, 0.1]
        rgba =  [0.1, 0.9, 0.9, 0.01]
        marker_params = ["sphere", "", [0, 0, 0], size, rgba, so3]
        if self.area_marker is None:
            self.area_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_position(self.area_marker, [0, 0, 0.005])
        self.sim.viewer.update_marker_size(self.area_marker, [self.area_radius, self.area_radius, 0.1])


        so3 = euler2so3(z=0, x=np.pi, y=0)
        size = [0.03, 0.03, 0.5]
        rgba =  [0.1, 0.8, 0.8, 0.5]
        marker_params = ["arrow", "", [0, 0, 0], size, rgba, so3]
        if self.desk_0_marker is None:
            self.desk_0_marker = self.sim.viewer.add_marker(*marker_params)

        self.arrow_0_height += self.arrow_0_change
        if self.arrow_0_height >= 1.85 or self.arrow_0_height <= 1.75:
            self.arrow_0_change = -self.arrow_0_change
                
        self.sim.viewer.update_marker_position(self.desk_0_marker, [self.desk_position[0][0], self.desk_position[0][1], self.arrow_0_height])

        so3 = euler2so3(z=0, x=np.pi, y=0)
        size = [0.03, 0.03, 0.5]
        rgba =  [0.1, 0.8, 0.8, 0.5]
        marker_params = ["arrow", "", [0, 0, 0], size, rgba, so3]
        if self.desk_1_marker is None:
            self.desk_1_marker = self.sim.viewer.add_marker(*marker_params)    
  
        self.arrow_1_height += self.arrow_1_change
        if self.arrow_1_height >= 1.85 or self.arrow_1_height <= 1.75:
            self.arrow_1_change = -self.arrow_1_change
                
        self.sim.viewer.update_marker_position(self.desk_1_marker, [self.desk_position[1][0], self.desk_position[1][1], self.arrow_1_height])

        so3 = euler2so3(z=0, x=np.pi, y=0)
        size = [0.03, 0.03, 0.5]
        rgba =  [0.1, 0.8, 0.8, 0.5]
        marker_params = ["arrow", "", [0, 0, 0], size, rgba, so3]
        if self.desk_2_marker is None:
            self.desk_2_marker = self.sim.viewer.add_marker(*marker_params)    

        self.arrow_2_height += self.arrow_2_change
        if self.arrow_2_height >= 1.85 or self.arrow_2_height <= 1.75:
            self.arrow_2_change = -self.arrow_2_change
                
        self.sim.viewer.update_marker_position(self.desk_2_marker, [self.desk_position[2][0], self.desk_position[2][1], self.arrow_2_height])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.4]
        rgba =  [0.1, 0.8, 0.8, 0.6]
        marker_params = ["label", "", [0, 0, 0], size, rgba, so3]
        if self.label_0_marker is None:
            self.label_0_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_name(self.label_0_marker, "Tower 0")
        self.sim.viewer.update_marker_position(self.label_0_marker, [self.desk_position[0][0], self.desk_position[0][1], self.arrow_0_height])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.4]
        rgba =  [0.1, 0.8, 0.8, 0.6]
        marker_params = ["label", "", [0, 0, 0], size, rgba, so3]
        if self.label_1_marker is None:
            self.label_1_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_name(self.label_1_marker, "Tower 1")
        self.sim.viewer.update_marker_position(self.label_1_marker, [self.desk_position[1][0], self.desk_position[1][1], self.arrow_1_height])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.4]
        rgba =  [0.1, 0.8, 0.8, 0.6]
        marker_params = ["label", "", [0, 0, 0], size, rgba, so3]
        if self.label_2_marker is None:
            self.label_2_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_name(self.label_2_marker, "Tower 2")
        self.sim.viewer.update_marker_position(self.label_2_marker, [self.desk_position[2][0], self.desk_position[2][1], self.arrow_2_height])


        target_position_0 = [self.desk_position[0][0], self.desk_position[0][1]]
        target_position_1 = [self.desk_position[1][0], self.desk_position[1][1]]
        target_position_2 = [self.desk_position[2][0], self.desk_position[2][1]]

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.2, 0.2, 0.0025]
        rgba =  [0.1, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position_0[0], target_position_0[1], 0.001], size, rgba, so3]
        if self.target_marker_0 is None:
            self.target_marker_0 = self.sim.viewer.add_marker(*marker_params)
        self.sim.viewer.update_marker_position(self.target_marker_0, [target_position_0[0], target_position_0[1], 0.005])
        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.18, 0.18, 0.0025]
        rgba = [0.1, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position_1[0], target_position_1[1], 0.001], size, rgba, so3]
        if self.target_marker_1 is None:
            self.target_marker_1 = self.sim.viewer.add_marker(*marker_params)
        self.sim.viewer.update_marker_position(self.target_marker_1, [target_position_1[0], target_position_1[1], 0.005])
        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.2, 0.2, 0.0025]
        rgba =  [0.1, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position_2[0], target_position_2[1], 0.001], size, rgba, so3]
        if self.target_marker_2 is None:
            self.target_marker_2 = self.sim.viewer.add_marker(*marker_params)
        self.sim.viewer.update_marker_position(self.target_marker_2, [target_position_2[0], target_position_2[1], 0.005])
        
        self.sim.viewer.update_tasks_skills_menu(self.action_process[self.action_process_index], str(self.finish_count))