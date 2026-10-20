import yaml
import rospkg
from os import path
from copy import copy

import rospy
import numpy as np

from std_msgs.msg import Int32, String, Float64, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

from op3_online_walking_module_msgs.msg import (
    FootStepCommand,
    JointPose,
)
from op3_online_walking_module_msgs.msg import (
    WalkingParam as OnlineWalkingParam,
)
from op3_walking_module_msgs.msg import WalkingParam
from op3_action_module_msgs.srv import IsRunning
from robotis_controller_msgs.srv import (
    SetModule,
    SetJointModule,
    SetJointModuleRequest,
)


rospack = rospkg.RosPack()
DEGREE2RADIAN = np.pi / 180.0


class Robot(object):
    """OP3 ROS control utility.

    This version preserves the original public API while fixing several issues:

    1. ``walkVelocities(th=0)`` now really means zero rotation.
       The old version silently applied ``th - 1``.
    2. ``z_offset=0`` can now be applied correctly.
    3. Joint effort values are stored using the correct joint names.
    4. YAML files are loaded safely and closed properly.
    5. Python 2/3 compatibility is retained.
    """

    def __init__(self):
        # Publishers
        self.joint_state_pub = rospy.Publisher(
            "/robotis/set_joint_states",
            JointState,
            queue_size=1,
        )

        self.action_pub = rospy.Publisher(
            "/robotis/action/page_num",
            Int32,
            queue_size=1,
        )

        self.walk_command_pub = rospy.Publisher(
            "/robotis/walking/command",
            String,
            queue_size=1,
        )

        self.walk_set_param_pub = rospy.Publisher(
            "/robotis/walking/set_params",
            WalkingParam,
            queue_size=1,
        )

        self.online_walk_set_body_offset_pub = rospy.Publisher(
            "/robotis/online_walking/body_offset",
            Pose,
            queue_size=1,
        )

        self.online_walk_foot_distance_pub = rospy.Publisher(
            "/robotis/online_walking/foot_distance",
            Float64,
            queue_size=1,
        )

        self.online_walk_command_pub = rospy.Publisher(
            "/robotis/online_walking/foot_step_command",
            FootStepCommand,
            queue_size=1,
        )

        self.online_walk_balance_pub = rospy.Publisher(
            "/robotis/online_walking/wholebody_balance_msg",
            String,
            queue_size=1,
        )

        self.online_walk_param_pub = rospy.Publisher(
            "/robotis/online_walking/walking_param",
            OnlineWalkingParam,
            queue_size=1,
        )

        self.online_walk_reset_pose_pub = rospy.Publisher(
            "/robotis/online_walking/reset_body",
            Bool,
            queue_size=1,
        )

        self.online_walk_goal_pose_pub = rospy.Publisher(
            "/robotis/online_walking/goal_joint_pose",
            JointPose,
            queue_size=1,
        )

        # Load calibrated walking parameters once.
        self.base_walking_params = getWalkingParams()

        # Services
        self.set_joint_ctrl_module = rospy.ServiceProxy(
            "/robotis/set_present_joint_ctrl_modules",
            SetJointModule,
        )

        self.set_ctrl_module = rospy.ServiceProxy(
            "/robotis/set_present_ctrl_modules",
            SetModule,
        )

        self.action_is_running = rospy.ServiceProxy(
            "/robotis/action/is_running",
            IsRunning,
        )

        # Subscribers
        rospy.Subscriber(
            "/robotis/present_joint_states",
            JointState,
            self.updateJointPos,
            queue_size=1,
        )

        rospy.Subscriber(
            "/robotis/open_cr/button",
            String,
            self.updateBtnStatus,
            queue_size=1,
        )

        self.joint_pos = {}
        self.joint_effort = {}
        self.btn_pressed = {
            "mode": False,
            "start": False,
            "user": False,
        }

    # ------------------------------------------------------------------
    # Button handling
    # ------------------------------------------------------------------

    def updateBtnStatus(self, msg):
        if msg.data in self.btn_pressed:
            self.btn_pressed[msg.data] = True

    def buttonCheck(self, btn):
        if btn not in self.btn_pressed:
            rospy.logwarn(
                "[buttonCheck] Unknown button name: {}".format(btn)
            )
            return False

        if self.btn_pressed[btn]:
            self.btn_pressed[btn] = False
            return True

        return False

    # ------------------------------------------------------------------
    # Online walking
    # ------------------------------------------------------------------

    def onlineWalkSetup(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        foot_dist=0.070,
        foot_height=0.05,
        dsp_ratio=0.20,
        zmp_offset_x=0.0,
        zmp_offset_y=0.0,
    ):
        self.setGeneralControlModule("online_walking_module")
        rospy.sleep(0.5)

        self.onlineWalkIniPose()
        self.online_walk_reset_pose_pub.publish(True)
        rospy.sleep(3.0)

        # Online walking requires whole-body balance.
        self.online_walk_balance_pub.publish("balance_on")
        rospy.sleep(1.0)

        walking_params = OnlineWalkingParam()
        walking_params.foot_height_max = foot_height
        walking_params.dsp_ratio = dsp_ratio
        walking_params.zmp_offset_x = zmp_offset_x
        walking_params.zmp_offset_y = zmp_offset_y
        walking_params.lipm_height = 0.12
        self.online_walk_param_pub.publish(walking_params)

        body_offsets = Pose()
        body_offsets.position.x = x
        body_offsets.position.y = y
        body_offsets.position.z = z
        body_offsets.orientation.x = 0.0
        body_offsets.orientation.y = 0.0
        body_offsets.orientation.z = 0.0
        body_offsets.orientation.w = 0.0
        self.online_walk_set_body_offset_pub.publish(body_offsets)

        self.online_walk_foot_distance_pub.publish(foot_dist)
        rospy.sleep(5.0)

    def onlineWalkCommand(
        self,
        direction,
        start_leg,
        step_num,
        front_length=0.0,
        side_length=0.0,
        step_angle=0.0,
        step_time=0.5,
    ):
        step_cmd = FootStepCommand()
        step_cmd.command = direction
        step_cmd.start_leg = start_leg
        step_cmd.step_num = step_num
        step_cmd.step_length = front_length
        step_cmd.side_length = side_length
        step_cmd.step_angle = step_angle
        step_cmd.step_time = step_time

        print(direction)
        self.online_walk_command_pub.publish(step_cmd)

    def onlineWalkIniPose(self):
        ini_path = path.join(
            rospack.get_path("op3_gui_demo"),
            "config",
            "init_pose.yaml",
        )

        with open(ini_path, "r") as ini_file:
            yf = yaml.safe_load(ini_file)

        msg = JointPose()
        msg.mov_time = yf["mov_time"]

        # Iterate over the same item pairs so names and values always match.
        pose_items = list(yf["tar_pose"].items())
        msg.pose.name = [item[0] for item in pose_items]
        msg.pose.position = [
            DEGREE2RADIAN * item[1]
            for item in pose_items
        ]

        self.online_walk_goal_pose_pub.publish(msg)

    # ------------------------------------------------------------------
    # Control modules and joints
    # ------------------------------------------------------------------

    def setGeneralControlModule(self, module_name):
        self.set_ctrl_module(module_name)

    def setJointsControlModule(self, joint_names, module_names):
        req = SetJointModuleRequest()

        if len(module_names) == 1:
            module_names = module_names * len(joint_names)

        if len(joint_names) != len(module_names):
            raise ValueError(
                "joint_names and module_names must have the same length"
            )

        req.joint_name = joint_names
        req.module_name = module_names
        self.set_joint_ctrl_module(req)

    def setJointPos(self, joints, pos):
        if len(joints) != len(pos):
            raise ValueError(
                "joints and pos must have the same length"
            )

        msg = JointState()
        msg.name = joints
        msg.position = pos
        self.joint_state_pub.publish(msg)

    def playMotion(self, motion_number, wait_for_end=False):
        self.action_pub.publish(motion_number)

        if wait_for_end:
            while (
                not rospy.is_shutdown()
                and self.action_is_running().is_running
            ):
                rospy.sleep(0.1)

    def updateJointPos(self, msg):
        # Position
        for index, joint_name in enumerate(msg.name):
            if index < len(msg.position):
                self.joint_pos[joint_name] = msg.position[index]

        # FIX: use msg.name as dictionary keys instead of effort values.
        for index, joint_name in enumerate(msg.name):
            if index < len(msg.effort):
                self.joint_effort[joint_name] = msg.effort[index]

    # ------------------------------------------------------------------
    # Standard walking module
    # ------------------------------------------------------------------

    def walkStart(self):
        self.walk_command_pub.publish("start")

    def walkStop(self):
        self.walk_command_pub.publish("stop")

    def walkVelocities(
        self,
        x=0.0,
        y=0.0,
        th=0.0,
        z_move_amplitude=None,
        balance=None,
        feet_pitch=None,
        z_offset=None,
        hip_pitch=None,
    ):
        """Publish walking parameters.

        Args:
            x: Forward/backward walking amplitude in the same scale used by
               the original utility. Internally divided by 1000.
            y: Sideways walking amplitude. Internally divided by 1000.
            th: Turning amplitude in degrees.

                IMPORTANT:
                In this corrected version, th=0 means zero turning.
                There is no hidden ``-1 degree`` offset.
            z_move_amplitude: Optional foot lift height.
            balance: Optional balance enable flag.
            feet_pitch: Optional initial pitch offset in degrees.
            z_offset: Optional initial Z offset. Zero is accepted.
            hip_pitch: Optional hip pitch offset in degrees.
        """
        mod_params = copy(self.base_walking_params)

        mod_params.x_move_amplitude = float(x) / 1000.0
        mod_params.y_move_amplitude = float(y) / 1000.0
        mod_params.angle_move_amplitude = (
            float(th) * DEGREE2RADIAN
        )

        if z_move_amplitude is not None:
            mod_params.z_move_amplitude = z_move_amplitude

        if balance is not None:
            mod_params.balance_enable = balance

        # FIX: zero is a valid offset and must not be ignored.
        if z_offset is not None:
            mod_params.init_z_offset = z_offset

        if feet_pitch is not None:
            mod_params.init_pitch_offset = (
                float(feet_pitch) * DEGREE2RADIAN
            )

        if hip_pitch is not None:
            mod_params.hip_pitch_offset = (
                float(hip_pitch) * DEGREE2RADIAN
            )

        self.walk_set_param_pub.publish(mod_params)

    def reloadWalkingParams(self, file_name="param.yaml"):
        """Reload the base walking parameters from a YAML file."""
        self.base_walking_params = getWalkingParams(file_name)


def getWalkingParams(file_name="param.yaml"):
    """Load OP3 walking parameters from a YAML configuration file."""
    config_path = path.join(
        rospack.get_path("op3_walking_module"),
        "config",
    )
    yaml_path = path.join(config_path, file_name)

    with open(yaml_path, "r") as yaml_file:
        yf = yaml.safe_load(yaml_file)

    walking_param = WalkingParam()

    # Initial pose offsets
    walking_param.init_x_offset = yf["x_offset"]
    walking_param.init_y_offset = yf["y_offset"]
    walking_param.init_z_offset = yf["z_offset"]
    walking_param.init_roll_offset = (
        yf["roll_offset"] * DEGREE2RADIAN
    )
    walking_param.init_pitch_offset = (
        yf["pitch_offset"] * DEGREE2RADIAN
    )
    walking_param.init_yaw_offset = (
        yf["yaw_offset"] * DEGREE2RADIAN
    )
    walking_param.hip_pitch_offset = (
        yf["hip_pitch_offset"] * DEGREE2RADIAN
    )

    print(
        "FEET DEFAULT VALUE: {}".format(
            yf["pitch_offset"]
        )
    )
    print(
        "DEFAULT Z OFFSET VALUE: {}".format(
            yf["z_offset"]
        )
    )
    print(
        "DEFAULT HIP PITCH OFFSET VALUE: {}".format(
            yf["hip_pitch_offset"]
        )
    )

    # Timing
    walking_param.period_time = (
        yf["period_time"] * 0.001
    )
    walking_param.dsp_ratio = yf["dsp_ratio"]
    walking_param.step_fb_ratio = (
        yf["step_forward_back_ratio"]
    )

    # Walking amplitudes
    walking_param.x_move_amplitude = 0.0
    walking_param.y_move_amplitude = 0.0
    walking_param.z_move_amplitude = yf["foot_height"]
    walking_param.angle_move_amplitude = 0.0

    # Balance
    walking_param.balance_hip_roll_gain = (
        yf["balance_hip_roll_gain"]
    )
    walking_param.balance_knee_gain = (
        yf["balance_knee_gain"]
    )
    walking_param.balance_ankle_roll_gain = (
        yf["balance_ankle_roll_gain"]
    )
    walking_param.balance_ankle_pitch_gain = (
        yf["balance_ankle_pitch_gain"]
    )
    walking_param.y_swap_amplitude = (
        yf["swing_right_left"]
    )
    walking_param.z_swap_amplitude = (
        yf["swing_top_down"]
    )
    walking_param.pelvis_offset = (
        yf["pelvis_offset"] * DEGREE2RADIAN
    )
    walking_param.arm_swing_gain = yf["arm_swing_gain"]

    # PID gains
    walking_param.p_gain = yf["p_gain"]
    walking_param.i_gain = yf["i_gain"]
    walking_param.d_gain = yf["d_gain"]

    walking_param.move_aim_on = False
    walking_param.balance_enable = False

    return walking_param