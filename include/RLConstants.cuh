#pragma once

#include "Cuda/Math.cuh"

// --- Math constants ---

constexpr float PI = 3.1415926535897932384626433832795029;
constexpr float PI_2 = PI / 2;
constexpr float PI_4 = PI / 4;
constexpr float SQRT_1_2 = 0.7071067811865475244f;

constexpr CARL_HD Vec3 WORLD_X = { 1.f, 0.f, 0.f };
constexpr CARL_HD Vec3 WORLD_Y = { 0.f, 1.f, 0.f };
constexpr CARL_HD Vec3 WORLD_Z = { 0.f, 0.f, 1.f };
constexpr CARL_HD Vec3 WORLD_AXES[3] = { WORLD_X, WORLD_Y, WORLD_Z };

// --- Arena dimensions ---

constexpr CARL_HD Vec3 ARENA_MIN = { -4108.f, -6000.f,  -14.f };
constexpr CARL_HD Vec3 ARENA_MAX = {  4108.f,  6000.f, 2076.f };

// --- Car dimensions ---

constexpr CARL_HD Vec3 CAR_EXTENTS = { 120.507f, 86.6994f,  38.6591f };
constexpr CARL_HD Vec3 CAR_HALF_EX = { 60.2535f, 43.3497f, 19.32955f };
constexpr CARL_HD Vec3 CAR_OFFSETS = { 13.8757f,      0.f,   20.755f };

constexpr float CAR_REST_Z = 17.f;
constexpr float CAR_RESPAWN_Z = 36.f;

// --- Ball dimensions ---

constexpr float BALL_REST_Z = 93.15f;
constexpr float BALL_COLLISION_RADIUS = 91.25f;

// --- Spawn locations ---

struct CarSpawn
{
    float x, y, z, yaw;
};

constexpr CARL_HD CarSpawn KICKOFF_LOCATIONS[5] = {
    { -2048.f, -2560.f, CAR_REST_Z, PI_4 * 1 },  // Right corner
    {  2048.f, -2560.f, CAR_REST_Z, PI_4 * 3 },  // Left corner
    {  -256.f, -3840.f, CAR_REST_Z, PI_4 * 2 },  // Back right
    {   256.f, -3840.f, CAR_REST_Z, PI_4 * 2 },  // Back left
    {     0.f, -4608.f, CAR_REST_Z, PI_4 * 2 }   // Back center
};

constexpr CARL_HD CarSpawn RESPAWN_LOCATIONS[4] = {
    {  2688.f, -4608.f, CAR_REST_Z, PI_2 * 1 },  // Left outside
    {  2304.f, -4608.f, CAR_REST_Z, PI_2 * 1 },  // Left inside
    { -2304.f, -4608.f, CAR_REST_Z, PI_2 * 1 },  // Right inside
    { -2688.f, -4608.f, CAR_REST_Z, PI_2 * 1 }   // Right outside
};

extern CARL_D CARL_C int KICKOFF_PERMUTATIONS[120][4];

// --- Boost pads ---

constexpr int NUM_BOOST_PADS = 34;
constexpr float BOOST_MAX = 100.f;
constexpr float BOOST_SPAWN_AMOUNT = BOOST_MAX / 3.f;

// --- Wheels and suspension geometry ---

constexpr int NUM_WHEELS = 4;
constexpr float FRONT_WHEEL_RADIUS = 12.5f;
constexpr float BACK_WHEEL_RADIUS = 15.f;
constexpr float FRONT_SUSPENSION_REST = 26.755f;
constexpr float BACK_SUSPENSION_REST = 25.055f;
constexpr float MAX_SUSPENSION_TRAVEL = 12.f;
constexpr float SUSPENSION_SUBTRACTION = 2.5f;

constexpr float FRONT_SUSP_X = 37.3743f;
constexpr float FRONT_SUSP_Y = 25.9f;
constexpr float BACK_SUSP_X = -47.6257f;
constexpr float BACK_SUSP_Y = 29.5f;

constexpr float FRONT_SUSP_RAY_LEN = FRONT_SUSPENSION_REST
    + MAX_SUSPENSION_TRAVEL + FRONT_WHEEL_RADIUS - SUSPENSION_SUBTRACTION;
constexpr float BACK_SUSP_RAY_LEN = BACK_SUSPENSION_REST
    + MAX_SUSPENSION_TRAVEL + BACK_WHEEL_RADIUS - SUSPENSION_SUBTRACTION;

// --- Collision capacity ---

constexpr int MAX_CAR_TRI_PAIRS = 32;
constexpr int MAX_PAIR_CONTACTS = 8;
constexpr int MAX_CAR_MANIFOLD_POINTS = 4;
constexpr float CAR_ANGULAR_MOTION_THRESH = PI_4;
constexpr float CAR_WORLD_RESTITUTION = 0.3f;
constexpr float CAR_WORLD_FRICTION = 0.3f;
constexpr float CAR_CONTACT_BREAK = 2.03425f;

// --- Rigid body ---

constexpr float PHYS_DT = 1.f / 120.f;
constexpr float CAR_MASS = 180.f;
constexpr float CAR_INV_MASS = 1.f / CAR_MASS;
constexpr float CAR_MAX_SPEED = 2300.f;
constexpr float CAR_MAX_ANG_SPEED = 5.5f;

constexpr CARL_D Vec3 CAR_INV_INERTIA = {
    1.f / 135169.80f,
    1.f / 240247.05f,
    1.f / 330580.95f
};
constexpr CARL_D Vec3 WORLD_GRAVITY = { 0.f, 0.f, -650.f };

// --- Solver ---

constexpr int CAR_SOLVER_ITERS = 10;
constexpr float CAR_CONTACT_ERP = 0.8f;
constexpr float CAR_SOLVER_WARMSTART = 0.85f;
constexpr float CAR_SOLVER_LINEAR_SLOP = 0.f;
constexpr float CAR_SPLIT_IMPULSE_TURN_ERP = 0.1f;
constexpr float CAR_SPLIT_PENETRATION_THRESH = 1.0e30f;
constexpr float CAR_RESTITUTION_VEL_THRESH = 10.f;

// --- Suspension forces ---

constexpr float SUSP_STIFFNESS = 500.f;
constexpr float SUSP_DAMPING_COMPRESSION = 25.f;
constexpr float SUSP_DAMPING_RELAXATION = 40.f;
constexpr float SUSP_FORCE_SCALE_FRONT = 35.75f;
constexpr float SUSP_FORCE_SCALE_BACK = 54.265f;
constexpr float SUSP_PUSHBACK_ERP = 0.2f;
constexpr float SUSP_PUSHBACK_DT_FIRST_TICK = 1.f / 60.f;
constexpr float CAR_STOPPING_VEL = 25.f;
constexpr float CAR_AUTOROLL_FORCE = 100.f;
constexpr float CAR_AUTOROLL_TORQUE = 80.f;

// --- Jump and dodge ---

constexpr float JUMP_ACCEL = 4375.f / 3.f;
constexpr float JUMP_IMMEDIATE_FORCE = 875.f / 3.f;
constexpr float JUMP_MIN_TIME = 0.025f;
constexpr float JUMP_RESET_TIME_PAD = 1.f / 40.f;
constexpr float JUMP_MAX_TIME = 0.2f;
constexpr float DOUBLEJUMP_MAX_DELAY = 1.25f;

constexpr float FLIP_Z_DAMP_120 = 0.35f;
constexpr float FLIP_Z_DAMP_START = 0.15f;
constexpr float FLIP_Z_DAMP_END = 0.21f;
constexpr float FLIP_TORQUE_TIME = 0.65f;
constexpr float FLIP_PITCHLOCK_EXTRA_TIME = 0.3f;
constexpr float FLIP_INITIAL_VEL_SCALE = 500.f;
constexpr float FLIP_TORQUE_X = 260.f;
constexpr float FLIP_TORQUE_Y = 224.f;
constexpr float FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1.f;
constexpr float FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = 1.9f;
constexpr float FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5f;
constexpr float FLIP_BACKWARD_IMPULSE_SCALE_X = 16.f / 15.f;
constexpr float DODGE_DEADZONE = 0.5f;

// --- Drive and boost ---

constexpr float THROTTLE_AIR_ACCEL = 200.f / 3.f;
constexpr float THROTTLE_GROUND_ACCEL = 1600.f;
constexpr float THROTTLE_DEADZONE = 0.001f;
constexpr float COASTING_BRAKE_FACTOR = 0.15f;

constexpr float BOOST_USED_PER_SECOND = BOOST_MAX / 3.f;
constexpr float BOOST_MIN_TIME = 0.1f;
constexpr float BOOST_ACCEL_GROUND = 2975.f / 3.f;
constexpr float BOOST_ACCEL_AIR = 3175.f / 3.f;

// --- Steering ---

constexpr float POWERSLIDE_RISE_RATE = 5.f;
constexpr float POWERSLIDE_FALL_RATE = 2.f;

// --- Air control and autoflip ---

constexpr float CAR_TORQUE_SCALE = 2.f * PI / 65536.f * 1000.f;
constexpr CARL_HD Vec3 CAR_AIR_CONTROL_TORQUE = { 130.f, 95.f, 400.f };
constexpr CARL_HD Vec3 CAR_AIR_CONTROL_DAMPING = { 30.f, 20.f, 50.f };
constexpr float CAR_AUTOFLIP_IMPULSE = 200.f;
constexpr float CAR_AUTOFLIP_TORQUE = 50.f;
constexpr float CAR_AUTOFLIP_TIME = 0.4f;
constexpr float CAR_AUTOFLIP_NORMZ_THRESH = 0.70710678f;
constexpr float CAR_AUTOFLIP_ROLL_THRESH = 2.8f;

// --- Car-car collision ---

constexpr float CARCAR_COLLISION_FRICTION = 0.09f;
constexpr float CARCAR_COLLISION_RESTITUTION = 0.1f;
constexpr float BUMP_COOLDOWN_TIME = 0.25f;
constexpr float BUMP_MIN_FORWARD_DIST = 64.5f;
constexpr float DEMO_RESPAWN_TIME = 3.f;
constexpr float DEMO_RESPAWN_INPUT_DEADZONE = 0.5f;
constexpr float SUPERSONIC_START_SPEED = 2200.f;

// --- Ball physics ---

constexpr float BALL_MASS = CAR_MASS / 6.f;
constexpr float BALL_INV_MASS = 1.f / BALL_MASS;
constexpr float BALL_RADIUS = BALL_COLLISION_RADIUS;
constexpr float BALL_DRAG = 0.03f;
constexpr float BALL_MAX_SPEED = 6000.f;
constexpr float BALL_MAX_ANG_SPEED = 6.f;
constexpr float BALL_WORLD_RESTITUTION = 0.6f;
constexpr float BALL_WORLD_FRICTION = 0.35f;
constexpr float BALL_INERTIA =
    0.4f * BALL_MASS * BALL_RADIUS * BALL_RADIUS;
constexpr float BALL_INV_INERTIA = 1.f / BALL_INERTIA;

// --- Car-ball collision ---

constexpr float CAR_BALL_RESTITUTION = 0.f;
constexpr float CAR_BALL_FRICTION = 2.f;
constexpr float CAR_BALL_CONTACT_BREAK = 1.825f;
constexpr float CAR_BALL_SHAPE_MARGIN = 2.f;
constexpr CARL_D Vec3 CAR_BALL_SOLVER_HALF_EX = {
    CAR_HALF_EX.x + CAR_BALL_SHAPE_MARGIN,
    CAR_HALF_EX.y + CAR_BALL_SHAPE_MARGIN,
    CAR_HALF_EX.z + CAR_BALL_SHAPE_MARGIN
};
constexpr CARL_D Vec3 CAR_BALL_CORE_HALF_EX = {
    CAR_HALF_EX.x - CAR_BALL_SHAPE_MARGIN,
    CAR_HALF_EX.y - CAR_BALL_SHAPE_MARGIN,
    CAR_HALF_EX.z - CAR_BALL_SHAPE_MARGIN
};

constexpr float BALL_CAR_EXTRA_IMPULSE_Z_SCALE = 0.35f;
constexpr float BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65f;
constexpr float BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU = 4600.f;

// --- Goal detection ---

constexpr float SOCCAR_GOAL_CENTER_Y = 5120.f;
constexpr float SOCCAR_GOAL_CENTER_Z = 321.3875f;
constexpr float SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y = 5124.25f;

// --- Steering curves ---

CARL_HD CARL_FI float steerAngleFromSpeed(float x)
{
    if (x <= 0.f)    return 0.53356f;
    if (x <= 500.f)  return 0.53356f + (0.31930f - 0.53356f) * (x / 500.f);
    if (x <= 1000.f) return 0.31930f + (0.18203f - 0.31930f) * ((x - 500.f) / 500.f);
    if (x <= 1500.f) return 0.18203f + (0.10570f - 0.18203f) * ((x - 1000.f) / 500.f);
    if (x <= 1750.f) return 0.10570f + (0.08507f - 0.10570f) * ((x - 1500.f) / 250.f);
    if (x <= 3000.f) return 0.08507f + (0.03454f - 0.08507f) * ((x - 1750.f) / 1250.f);
    return 0.03454f;
}

CARL_HD CARL_FI float powerslideSteerAngleFromSpeed(float x)
{
    if (x <= 0.f)    return 0.39235f;
    if (x <= 2500.f) return 0.39235f + (0.12610f - 0.39235f) * (x / 2500.f);
    return 0.12610f;
}

// --- Drive torque curve ---

CARL_HD CARL_FI float driveSpeedTorqueFactor(float x)
{
    if (x <= 0.f)    return 1.f;
    if (x >= 1410.f) return 0.f;
    if (x <= 1400.f) return 1.f - (x / 1400.f) * 0.9f;
    return 0.1f * (1.f - (x - 1400.f) / 10.f);
}
