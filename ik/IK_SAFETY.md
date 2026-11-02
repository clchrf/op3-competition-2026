# OP3 IK safety values (for AI assistants)

Give this file to your AI assistant BEFORE it writes or changes any IK motion (arm reach,
squat, pick, lift) for the ROBOTIS OP3. It lists which joint values are SAFE, which are UNSAFE,
why, and how to build a new IK motion step by step. The numbers come from the robot model
(ROBOTIS URDF) and from real robot tests in Oct 2026. It contains no program code: write
your own, but keep every limit and check below.

Rules that win over everything here: the robot never falls, never hits itself, new motions are
slow and in small steps, MODE button = emergency stop. The OP3 runs Python 2.7 + ROS Kinetic.

## 0. The 5 most important facts

1. **Joint targets sent on `/robotis/set_joint_states` (joints whose control module is `none`)
   go straight to the motors: NO range check and NO smoothing in the ROBOTIS controller.**
   Your own checks are the only protection. Never send a pose that did not pass section 3.
2. **Leg limit: |hip_pitch| <= 1.65, |knee| <= 2.47, |ank_pitch| <= 1.80 rad.**
   The real robot BLOCKED at hip_pitch 2.06 rad (squat test 2026-10-20).
3. **Arm vs leg: keep >= 4 cm free space in the model** (surface to surface). The model said
   17 mm and the real hand HIT the leg (2026-10-28): gripper fingers stick out and the knee is
   bigger than a simple model leg. 15 mm is enough near the body, NOT near the legs.
4. **Center of mass (CoM) >= 15 mm inside the feet area (20 mm when lifting a load).** The
   model has about 1 cm error; a heavy load in the hands falls FORWARD.
5. **Slow: legs 0.3-0.35 rad/s, arms 0.3-0.6 rad/s, never above 1.0-1.5 rad/s peak, every move
   >= 1 s.** Check the end pose AND poses on the way (every 0.05 rad of joint change).

## 1. Frames and signs (get these wrong = wrong side / wrong direction)

- **body frame**: origin between the hips (`body_link`), x forward, y LEFT, z up. m, rad.
- **feet frame**: on the floor between the two soles, x forward, y LEFT, z up. Good for
  targets from the camera ("object 0.16 m in front of the feet center, 0.06 m high").
- Joint names and signs = `/robotis/present_joint_states`. Angle 0 = URDF zero pose
  (arms straight out to the sides, legs straight).
- **Left and right are mirrored**: `r_hip_pitch = -l_hip_pitch`, `r_knee = -l_knee`,
  `r_ank_pitch = -l_ank_pitch`; arm limits of the right arm are the mirror of the left.
  Compare legs with `abs()`.
- A squat is easiest to describe as the body pose `[body_x, body_z, lean]` in the feet frame
  (lean + = forward). The OP3 walking stand is about `[0.041 m, 0.257 m, 8 deg]`; its joints
  are about hip_pitch 0.64, knee 1.16, ank_pitch 0.64 rad (left leg values).

## 2. Joint ranges: safe / unsafe

### Legs

| joint (abs value) | ROBOTIS kinematics limit | seniors' proven motion pages (max) | limit to use (reject above) | robot result |
|---|---|---|---|---|
| hip_pitch | 1.26 | 1.61 | **1.65** | 2.06 -> BLOCKED |
| knee | 2.20 | 2.45 | **2.47** | - |
| ank_pitch | 1.41 | 1.76 | **1.80** | - |

- ROBOTIS' kinematics limits are smaller than the proven pages, and the pages work. So the real
  stop is the mechanics: use the right column and never go above it.
- For a squat keep hip_yaw, hip_roll, ank_roll at 0 (soles flat, legs parallel).
- Squat proven on the robot: `[body_x 0.04 m, body_z 0.145 m, lean 26 deg]` (IMU showed
  +18 deg, model +17). Good model range: body_x 0.04-0.05, body_z 0.13-0.16, lean 20-26.
- body_x 0.08 (seniors' squat page) puts the CoM 7 mm OVER the front edge of the feet:
  keep body_x <= 0.05.

### Leg IK (2 links, sole flat)

Thigh (hip pitch axis -> knee axis) = 0.11015 m, shin (knee axis -> ankle pitch axis) = 0.110 m.
For the LEFT leg, with (dx, dz) = hip pitch axis minus ankle pitch axis (m), r = sqrt(dx^2+dz^2):

    cos(knee) = (r^2 - thigh^2 - shin^2) / (2 * thigh * shin)   (outside -1..1 = can not reach)
    psi = atan2(dx, dz)
    f1  = psi - atan2(shin * sin(knee), thigh + shin * cos(knee))
    hip_pitch = f1 - lean,   knee = knee,   ank_pitch = f1 + knee

Right leg = the same values with the opposite sign. The hip pitch axis sits about (0.0001,
-0.0285) m from the body origin (x, z), the sole is 0.0305 m below the ankle axis. Fix the small
offsets by running forward kinematics and correcting body_x / body_z 3-4 times.
Check: the seniors' stand page gives hip -30.2 / knee 50.5 / ankle 29.0 deg; this IK gives
-29.7 / 51.0 / 29.3 deg.

### Arms (3 joints: shoulder pitch, shoulder roll, elbow; no wrist)

"use" = normal working range (cut targets to it). "reject" = outside means do not move.

| joint | use | reject outside | sign |
|---|---|---|---|
| l_sho_pitch | -2.5 .. 1.3 | -2.6 .. 1.5 | - = arm forward / up |
| r_sho_pitch | -1.3 .. 2.5 | -1.5 .. 2.6 | + = arm forward / up |
| l_sho_roll | -1.3 .. 1.55 | -1.4 .. 1.6 | + = arm down (1.57 = along the body) |
| r_sho_roll | -1.55 .. 1.3 | -1.6 .. 1.4 | mirror |
| l_el / r_el | -1.4 .. 1.4 | -1.6 .. 1.6 | |

Arms over the head (weightlifting) need wider ranges: l_sho_pitch -3.0 .. 1.0,
r_sho_pitch -1.0 .. 3.0, l_el -2.0 .. 1.0, r_el -1.0 .. 2.0. Use them only for that.

- Arm hanging at the side: roll about +-1.5 (seniors' home: left (0.2, 1.5, 0), right
  (-0.2, -1.5, 0)). Swinging it forward on a straight joint line with bent knees hits the knee:
  first move it out to |roll| 1.35, then swing.
- In a squat the free arm at its rest pose hits its own foot: move it forward / out first
  (left arm about (-0.9, 0.9, 0.0), right arm mirrored).
- Arm IK has 2 solutions. Between two points on a line no joint may change more than 0.3 rad,
  else the arm flips to the other solution. Make straight hand lines from many IK points
  about 4 mm apart, and accept a point only if the hand is within 3 mm of the target.
- Model sizes: upper arm 0.0904 m, forearm 0.1465 m (elbow axis -> hand end, without gripper).
  A gripper makes the forearm longer: measure elbow axis -> finger tip and add it.

### Head

pan -1.0 .. 1.0, tilt -1.3 .. 0.2 (looking down is -). In a squat the arm reaching forward can
hit the head at about pan -0.6 / tilt -1.1: keep the head away from the working arm.

### Grippers

Close slowly (about 0.2 rad/s). When the closing motor falls about 0.10 rad behind its goal,
something is between the fingers: squeeze at most 0.04-0.06 rad more (motor gets hot / the
object breaks).

## 3. Checks every pose and every move must pass

Use a model from the ROBOTIS URDF (`op3_description`: joint origins + link masses, about
3.15 kg; the real robot is about 3.5 kg, so keep the margins). Check the end pose and a pose
every 0.05 rad on the way.

| check | safe value | unsafe when |
|---|---|---|
| arm joint ranges | table above | outside |
| leg joint ranges | leg table above | above |
| arm vs body / head / feet / other arm | >= 15 mm free (10 mm with care) | closer |
| arm vs body SIDE | >= 4 mm (the real gap at the shoulder is only ~8 mm) | closer |
| **arm / hand vs legs** | **>= 40 mm** | closer |
| hand vs floor | >= 30 mm (floor picks: finger tips >= 3-10 mm) | lower |
| soles flat on one plane | tilt <= 4 deg, height difference <= 12 mm | more |
| balance (CoM above the feet area) | CoM >= 15 mm inside (20 mm with a load) | less |
| balance in motion (ZMP) | >= 6-8 mm inside | less |
| joint speed (smooth min-jerk move, peak) | <= 1.0-1.5 rad/s | faster |
| static load on one arm motor | <= 1.0 N m (XM430-W350 stalls at ~4) | more |

Notes:
- Model parts as capsules / boxes: arm about 22 mm radius, thigh / shin about 25 mm radius,
  torso box x -0.104 .. 0.0705, y +-0.0644, z -0.005 .. 0.141 m (body frame).
- Feet area = convex hull of both soles, each sole made 4 mm smaller (round edges).
  Sole in the ankle roll frame: x -0.0395 .. 0.0875, z -0.0305 m.
- ZMP of a smooth (min-jerk) move: peak acceleration = 5.77 * distance / T^2; the ZMP moves
  CoM height * acceleration / 9.81 away from the CoM.
- The real walking stand looks like soles tilted ~5 deg in the model (hip roll of the walking
  module). With a 4 deg limit every pose was refused. Fix: measure the soles after standing up
  and allow measured + 2 deg / 5 mm, never more than 8 deg.

## 4. Guards WHILE the robot moves (stop and hold the pose)

| guard | stop when |
|---|---|
| a motor does not follow (hit something) | > 0.15 rad from its target for 0.3 s |
| body lean (IMU pitch) differs from the planned lean | > 15 deg (20 for weightlifting) |
| body roll (IMU) | > 12-20 deg |
| standing arm move tilts the body | > 8 deg |

Check the IMU pitch sign once on the robot (does + mean leaning forward like the model?).

## 5. How to make a new IK motion

1. **Start from a proven motion** (the seniors' motion pages / code). Measure their poses with
   forward kinematics and start your numbers there.
2. **Legs**: compute the squat with the leg IK above. Reject it if a leg joint is above the leg
   limits. Make a LIST of squats and use the first one that passes every check.
3. **Arm**: numeric IK for the 3 arm joints (start from a table of poses near the target, then
   damped least squares), clipped to the "use" range. Reject if the hand error > 3 mm.
4. **Check every move** from the current pose with section 3. If a straight move fails, try a
   via point or move the arm out first. If nothing passes: DO NOT MOVE; walk / reposition.
5. **Order**: free arm out of the way -> squat (arms still) -> hand above the target (3-4 cm, a
   gripper at 1.5 cm knocked the object over) -> straight line down -> grip -> same line up ->
   stand up. Legs move alone and slowly.
6. **Send**: small steps (about 50 per second), peak speed legs 0.3, arms 0.3-0.6 rad/s, each
   move at least 1 s. Run the guards of section 4 every step.
7. **Test order**: on the PC (unit test / simulation) -> debug view that shows the model and why
   a pose is refused -> robot HUNG in the air (balance checks off) -> robot on the floor, one
   stage at a time, hand on MODE. Change one value at a time.
8. **Log** the planned pose, every check result (free space, CoM margin, ZMP margin) and the
   real joints. When the robot differs from the model, write it down with the date and make
   the margin bigger, never smaller.

## 6. Robot lessons (why the numbers are what they are)

| date (2026) | what happened | rule now |
|---|---|---|
| 10-14/15 | real stand = 5.2 deg sole tilt in the model -> every pose refused | measure soles after standing, +2 deg |
| 10-15 | squat motion page moved the arms while squatting, right hand HIT THE LEG | legs and arms move separately |
| 10-15 | a joint taken out of `action_module` disables it -> the page was ignored | check the knee really bent |
| 10-20 | squat with hip_pitch 2.06 -> r_hip_pitch BLOCKED | hip_pitch <= 1.65 |
| 10-28 | model 17 mm arm-leg gap, real hand HIT the leg | arm-leg >= 40 mm |
| 10-28 | gripper 1.5 cm over the object knocked it over | 3-4 cm above, then straight down |
| 11-02 | 12 mm CoM margin + 1 cm model error + heavy bar = falls forward | 20 mm with a load |

## 7. ROBOTIS references (in any OP3 workspace)

- Leg IK: `op3_kinematics_dynamics/src/op3_kinematics_dynamics.cpp`,
  `calcInverseKinematicsForLeg` (also has the ROBOTIS joint limits).
- Robot model: `ROBOTIS-OP3-Common/op3_description/urdf/` (joints, masses), `meshes/` (sizes).
- Whole-body Cartesian control (another way to squat): `op3_online_walking_module`
  (`/robotis/online_walking/goal_kinematics_pose`, needs `balance_on` first). Untested for deep
  squats.
- Joint control with smoothing and a simple arm collision check: `op3_direct_control_module`
  (`/robotis/direct_control/set_joint_states`).
