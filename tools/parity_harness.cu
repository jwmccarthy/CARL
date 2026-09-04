#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "RLEnvironment.cuh"
#include "Cuda/Common.cuh"
#include "Cuda/Math.cuh"

#include "RocketSim.h"

namespace
{

bool g_traceMode = false;

struct Scenario
{
    const char* name;

    int totalTicks;
    int resetTick;
    bool contactReset;

    Vec3 ballPos;
    Vec3 ballVel;
    Vec3 ballAng;

    Vec3 carPos;
    Quat carRot;
    Vec3 carVel;
    Vec3 carAng;

    int carDemoed;

    float throttle;
    float steer;
    float yaw;
    float pitch;
    float roll;

    bool jump;
    bool boost;
    bool handbrake;
};

bool isTraceTarget(const Scenario& s)
{
    return g_traceMode;
}

struct StateSnapshot
{
    Vec3 ballPos;
    Vec3 ballVel;
    Vec3 ballAng;

    Vec3 carPos;
    Vec3 carVel;
    Vec3 carAng;
    Quat carRot;

    CarInternalState carInternal;
    int carDemoed;

    int ballContactTick;
    int ballHitTick;
};

RocketSim::RotMat quatToRotMat(const Quat& q)
{
    RocketSim::RotMat m;

    m.forward = RocketSim::Vec(
        1.f - 2.f * (q.y * q.y + q.z * q.z),
        2.f * (q.x * q.y + q.w * q.z),
        2.f * (q.x * q.z - q.w * q.y));

    m.right = RocketSim::Vec(
        2.f * (q.x * q.y - q.w * q.z),
        1.f - 2.f * (q.x * q.x + q.z * q.z),
        2.f * (q.y * q.z + q.w * q.x));

    m.up = RocketSim::Vec(
        2.f * (q.x * q.z + q.w * q.y),
        2.f * (q.y * q.z - q.w * q.x),
        1.f - 2.f * (q.x * q.x + q.y * q.y));

    return m;
}

__global__ void setupScenarioKernel(
    GameState* state,
    Workspace* space,
    Scenario scenario)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx != 0) return;

    state->tickCount = 0;
    state->episodeTicks[0] = 0;
    state->lastBallTouchTicks[0] = 0;
    state->goals[0] = {};

    for (int pad = 0; pad < NUM_BOOST_PADS; pad++)
    {
        state->boostPadCooldowns[pad] = 0.f;
    }

    state->ball.pos[0] = scenario.ballPos;
    state->ball.vel[0] = scenario.ballVel;
    state->ball.ang[0] = scenario.ballAng;
    state->ball.imp[0] = Vec3::zero();

    constexpr int carIdx = 0;

    state->cars.pos[carIdx] = scenario.carPos;
    state->cars.rot[carIdx] = scenario.carRot;
    state->cars.vel[carIdx] = scenario.carVel;
    state->cars.ang[carIdx] = scenario.carAng;
    state->cars.cen[carIdx] = scenario.carPos + scenario.carRot.toWorld(CAR_OFFSETS);
    state->cars.imp[carIdx] = Vec3::zero();

    state->cars.isDemoed[carIdx] = scenario.carDemoed;
    state->cars.demoRespawnTimer[carIdx] = scenario.carDemoed
        ? 1e6f
        : 0.f;
    state->cars.carContactIdx[carIdx] = -1;
    state->cars.carContactCooldown[carIdx] = 0.f;
    state->cars.ballHitTick[carIdx] = -1;
    state->cars.ballContactTick[carIdx] = -1;
    state->cars.controls[carIdx] = {};

    CarInternalState internal{};
    internal.isOnGround = true;
    internal.airTime = 0.f;
    internal.airTimeSinceJump = 0.f;
    internal.handbrakeVal = 0.f;
    internal.hasJumped = 0;
    internal.isJumping = 0;
    internal.lastJump = 0;
    internal.jumpTime = 0.f;
    internal.hasDoubleJumped = 0;
    internal.hasFlipped = 0;
    internal.isFlipping = 0;
    internal.flipTime = 0.f;
    internal.isAutoFlipping = 0;
    internal.autoFlipTimer = 0.f;
    internal.autoFlipTorqueScale = 0.f;
    internal.flipRelTorque = Vec3::zero();
    internal.isBoosting = 0;
    internal.boost = BOOST_SPAWN_AMOUNT;
    internal.boostingTime = 0.f;
    internal.timeSinceBoosted = 0.f;
    state->cars.internal[carIdx] = internal;

    space->bp.numTris[carIdx] = 0;
    space->ctHit.carHitCount[carIdx] = 0;
    space->ctMan.count[carIdx] = 0;

    for (int pair = 0; pair < MAX_CAR_TRI_PAIRS; pair++)
    {
        space->ctNrw.conPairCount[carIdx * MAX_CAR_TRI_PAIRS + pair] = 0;
    }

    space->susp.brakeFactor[carIdx] = 0.f;
    space->susp.brakeFactorPrev[carIdx] = 0.f;
    space->susp.throttleVal[carIdx] = 0.f;
    space->susp.rawThrottle[carIdx] = 0.f;
    space->susp.handbrakeVal[carIdx] = 0.f;
    space->susp.steerAngle[carIdx] = 0.f;
    space->susp.jumpImpulse[carIdx] = Vec3::zero();
    space->susp.engineDrive[carIdx] = 0.f;
    space->susp.engineDrivePrev[carIdx] = 0.f;

    for (int wheel = 0; wheel < NUM_WHEELS; wheel++)
    {
        const int wheelIdx = carIdx * NUM_WHEELS + wheel;
        space->susp.rayDist[wheelIdx] = 0.f;
        space->susp.rayNormal[wheelIdx] = Vec3::zero();
        space->susp.latFrictionPrev[wheelIdx] = 0.f;
        space->susp.lonFrictionPrev[wheelIdx] = 0.f;
    }

    for (int pair = 0; pair < space->ccMan.maxPairsPerSim; pair++)
    {
        space->ccMan.count[pair] = 0;
    }
}

__global__ void captureStateKernel(
    const GameState* state,
    StateSnapshot* out)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx != 0) return;

    out->ballPos = state->ball.pos[0];
    out->ballVel = state->ball.vel[0];
    out->ballAng = state->ball.ang[0];

    constexpr int carIdx = 0;
    out->carPos = state->cars.pos[carIdx];
    out->carVel = state->cars.vel[carIdx];
    out->carAng = state->cars.ang[carIdx];
    out->carRot = state->cars.rot[carIdx];
    out->carInternal = state->cars.internal[carIdx];
    out->carDemoed = state->cars.isDemoed[carIdx];

    out->ballContactTick = state->cars.ballContactTick[carIdx];
    out->ballHitTick = state->cars.ballHitTick[carIdx];
}

__global__ void resetStateKernel(
    GameState* state,
    Workspace* space,
    StateSnapshot snap,
    bool contactAware,
    int contactTick)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx != 0) return;

    state->ball.pos[0] = snap.ballPos;
    state->ball.vel[0] = snap.ballVel;
    state->ball.ang[0] = snap.ballAng;
    state->ball.imp[0] = Vec3::zero();

    constexpr int carIdx = 0;

    state->cars.pos[carIdx] = snap.carPos;
    state->cars.rot[carIdx] = snap.carRot;
    state->cars.vel[carIdx] = snap.carVel;
    state->cars.ang[carIdx] = snap.carAng;
    state->cars.cen[carIdx] = snap.carPos + snap.carRot.toWorld(CAR_OFFSETS);
    state->cars.imp[carIdx] = Vec3::zero();

    state->cars.isDemoed[carIdx] = snap.carDemoed;
    state->cars.demoRespawnTimer[carIdx] = snap.carDemoed
        ? DEMO_RESPAWN_TIME + PHYS_DT
        : 0.f;
    state->cars.carContactIdx[carIdx] = -1;
    state->cars.carContactCooldown[carIdx] = 0.f;

    if (contactAware && contactTick >= 0)
    {
        state->cars.ballHitTick[carIdx] = contactTick;
        state->cars.ballContactTick[carIdx] = contactTick;
    }
    else
    {
        state->cars.ballHitTick[carIdx] = -1;
        state->cars.ballContactTick[carIdx] = -1;
    }

    state->cars.controls[carIdx] = {};
    state->cars.internal[carIdx] = snap.carInternal;

    space->bp.numTris[carIdx] = 0;
    space->ctHit.carHitCount[carIdx] = 0;
    space->ctMan.count[carIdx] = 0;

    for (int pair = 0; pair < MAX_CAR_TRI_PAIRS; pair++)
    {
        space->ctNrw.conPairCount[carIdx * MAX_CAR_TRI_PAIRS + pair] = 0;
    }

    space->susp.brakeFactor[carIdx] = 0.f;
    space->susp.brakeFactorPrev[carIdx] = 0.f;
    space->susp.throttleVal[carIdx] = 0.f;
    space->susp.rawThrottle[carIdx] = 0.f;
    space->susp.handbrakeVal[carIdx] = 0.f;
    space->susp.steerAngle[carIdx] = 0.f;
    space->susp.jumpImpulse[carIdx] = Vec3::zero();
    space->susp.engineDrive[carIdx] = 0.f;
    space->susp.engineDrivePrev[carIdx] = 0.f;

    for (int wheel = 0; wheel < NUM_WHEELS; wheel++)
    {
        const int wheelIdx = carIdx * NUM_WHEELS + wheel;
        space->susp.rayDist[wheelIdx] = 0.f;
        space->susp.rayNormal[wheelIdx] = Vec3::zero();
        space->susp.latFrictionPrev[wheelIdx] = 0.f;
        space->susp.lonFrictionPrev[wheelIdx] = 0.f;
    }

    for (int pair = 0; pair < space->ccMan.maxPairsPerSim; pair++)
    {
        space->ccMan.count[pair] = 0;
    }
}

int axisIndex(float value)
{
    if (value < 0.f) return 1;
    if (value > 0.f) return 2;
    return 0;
}

DiscreteControls makeDiscreteControls(const Scenario& s)
{
    DiscreteControls d{};
    d.horizontal = axisIndex(s.steer);
    d.vertical = axisIndex(s.pitch);
    d.throttle = axisIndex(s.throttle);
    d.powerslide = s.handbrake ? 1 : 0;
    d.boost = s.boost ? 1 : 0;
    d.airRoll = axisIndex(s.roll);
    d.jump = s.jump ? 1 : 0;
    return d;
}

RocketSim::CarControls makeRocketControls(const Scenario& s)
{
    RocketSim::CarControls c;
    c.throttle = s.throttle;
    c.steer = s.steer;
    c.yaw = s.yaw;
    c.pitch = s.pitch;
    c.roll = s.roll;
    c.jump = s.jump;
    c.boost = s.boost;
    c.handbrake = s.handbrake;
    return c;
}

RocketSim::Vec toRocketVec(const Vec3& v)
{
    return RocketSim::Vec(v.x, v.y, v.z);
}

float vecDiff(const Vec3& a, const RocketSim::Vec& b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float rotDiff(const Quat& q, const RocketSim::RotMat& m)
{
    const RocketSim::RotMat carlM = quatToRotMat(q);
    float maxErr = 0.f;

    for (int i = 0; i < 3; i++)
    {
        const RocketSim::Vec col = carlM[i];
        const RocketSim::Vec ref = m[i];
        maxErr = std::max(maxErr, std::fabs(col.x - ref.x));
        maxErr = std::max(maxErr, std::fabs(col.y - ref.y));
        maxErr = std::max(maxErr, std::fabs(col.z - ref.z));
    }

    return maxErr;
}

struct ErrorStats
{
    float carPos = 0.f;
    float carVel = 0.f;
    float carAng = 0.f;
    float carRot = 0.f;
    float ballPos = 0.f;
    float ballVel = 0.f;
    float ballAng = 0.f;

    float carlBallSpeed = 0.f;
    float rsBallSpeed = 0.f;
    float carlBallAngSpeed = 0.f;
    float rsBallAngSpeed = 0.f;
};

ErrorStats runPostResetSegment(
    RLEnvironment& env,
    DiscreteControls* dActions,
    StateSnapshot* dSnapshot,
    const StateSnapshot& snap,
    bool contactAware,
    int contactTick,
    int ticks,
    const RocketSim::CarControls& rsControls,
    const DiscreteControls& carlControls,
    int groupTicks)
{
    resetStateKernel<<<1, 1, 0, env.getStream()>>>(
        env.getDeviceState(), env.getDeviceWorkspace(), snap,
        contactAware, contactTick);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

    RocketSim::Arena* arena = RocketSim::Arena::Create(
        RocketSim::GameMode::SOCCAR);
    RocketSim::Car* car = arena->AddCar(
        RocketSim::Team::BLUE, RocketSim::CAR_CONFIG_OCTANE);
    RocketSim::Ball* ball = arena->ball;

    RocketSim::CarState rsCarState = {};
    rsCarState.pos = toRocketVec(snap.carPos);
    rsCarState.rotMat = quatToRotMat(snap.carRot);
    rsCarState.vel = toRocketVec(snap.carVel);
    rsCarState.angVel = toRocketVec(snap.carAng);
    rsCarState.isOnGround = snap.carInternal.isOnGround;
    rsCarState.isDemoed = snap.carDemoed != 0;
    rsCarState.demoRespawnTimer = snap.carDemoed != 0
        ? 1e6f
        : 0.f;
    rsCarState.boost = snap.carInternal.boost;
    car->SetState(rsCarState);

    RocketSim::BallState rsBallState = {};
    rsBallState.pos = toRocketVec(snap.ballPos);
    rsBallState.vel = toRocketVec(snap.ballVel);
    rsBallState.angVel = toRocketVec(snap.ballAng);
    ball->SetState(rsBallState);

    car->controls = rsControls;

    ErrorStats err;

    for (int tick = 0; tick < ticks; tick += groupTicks)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            dActions, &carlControls, sizeof(DiscreteControls),
            cudaMemcpyHostToDevice, env.getStream()));

        car->controls = rsControls;

        StateSnapshot carlSnap;

        for (int i = 0; i < groupTicks; i++)
        {
            env.step(dActions);

            captureStateKernel<<<1, 1, 0, env.getStream()>>>(
                env.getDeviceState(), dSnapshot);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(
                &carlSnap, dSnapshot, sizeof(StateSnapshot),
                cudaMemcpyDeviceToHost, env.getStream()));
            CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

            arena->Step(1);

            const RocketSim::CarState rsCar = car->GetState();
            const RocketSim::BallState rsBall = ball->GetState();

            err.carlBallSpeed = std::max(err.carlBallSpeed, carlSnap.ballVel.len());
            err.rsBallSpeed = std::max(err.rsBallSpeed, rsBall.vel.Length());
            err.carlBallAngSpeed = std::max(err.carlBallAngSpeed, carlSnap.ballAng.len());
            err.rsBallAngSpeed = std::max(err.rsBallAngSpeed, rsBall.angVel.Length());
        }

        const RocketSim::CarState rsCar = car->GetState();
        const RocketSim::BallState rsBall = ball->GetState();

        err.carPos = std::max(err.carPos, vecDiff(carlSnap.carPos, rsCar.pos));
        err.carVel = std::max(err.carVel, vecDiff(carlSnap.carVel, rsCar.vel));
        err.carAng = std::max(err.carAng, vecDiff(carlSnap.carAng, rsCar.angVel));
        err.carRot = std::max(err.carRot, rotDiff(carlSnap.carRot, rsCar.rotMat));
        err.ballPos = std::max(err.ballPos, vecDiff(carlSnap.ballPos, rsBall.pos));
        err.ballVel = std::max(err.ballVel, vecDiff(carlSnap.ballVel, rsBall.vel));
        err.ballAng = std::max(err.ballAng, vecDiff(carlSnap.ballAng, rsBall.angVel));
    }

    delete arena;
    return err;
}

struct TraceRecord
{
    int tick;
    StateSnapshot carl;
    RocketSim::CarState rsCar;
    RocketSim::BallState rsBall;
    Vec3 carlCarDelta;
    Vec3 carlBallDelta;
    RocketSim::Vec rsCarDelta;
    RocketSim::Vec rsBallDelta;

    Vec3 carlBallExpectedDv;
    Vec3 carlBallContactDv;
    Vec3 carlBallPosCorr;

    RocketSim::Vec rsBallExpectedDv;
    RocketSim::Vec rsBallContactDv;
    RocketSim::Vec rsBallPosCorr;

    float ballPosError;
    float ballVelError;
    bool isEvent;
    std::string eventDesc;
};

void printTraceRecord(const TraceRecord& r, bool showEvent)
{
    std::printf("tick=%3d%s%s\n", r.tick,
        showEvent && r.isEvent ? " *** " : "",
        showEvent && r.isEvent ? r.eventDesc.c_str() : "");
    std::printf(
        "  CARL car P=%.2f,%.2f,%.2f V=%.2f,%.2f,%.2f W=%.2f,%.2f,%.2f\n",
        r.carl.carPos.x, r.carl.carPos.y, r.carl.carPos.z,
        r.carl.carVel.x, r.carl.carVel.y, r.carl.carVel.z,
        r.carl.carAng.x, r.carl.carAng.y, r.carl.carAng.z);
    std::printf(
        "       ball P=%.2f,%.2f,%.2f V=%.2f,%.2f,%.2f W=%.2f,%.2f,%.2f bct=%d bht=%d\n",
        r.carl.ballPos.x, r.carl.ballPos.y, r.carl.ballPos.z,
        r.carl.ballVel.x, r.carl.ballVel.y, r.carl.ballVel.z,
        r.carl.ballAng.x, r.carl.ballAng.y, r.carl.ballAng.z,
        r.carl.ballContactTick, r.carl.ballHitTick);
    std::printf(
        "  RS   car P=%.2f,%.2f,%.2f V=%.2f,%.2f,%.2f W=%.2f,%.2f,%.2f\n",
        r.rsCar.pos.x, r.rsCar.pos.y, r.rsCar.pos.z,
        r.rsCar.vel.x, r.rsCar.vel.y, r.rsCar.vel.z,
        r.rsCar.angVel.x, r.rsCar.angVel.y, r.rsCar.angVel.z);
    std::printf(
        "       ball P=%.2f,%.2f,%.2f V=%.2f,%.2f,%.2f W=%.2f,%.2f,%.2f bhi=%s hit=%llu extra=%llu extraVel=%.2f,%.2f,%.2f\n",
        r.rsBall.pos.x, r.rsBall.pos.y, r.rsBall.pos.z,
        r.rsBall.vel.x, r.rsBall.vel.y, r.rsBall.vel.z,
        r.rsBall.angVel.x, r.rsBall.angVel.y, r.rsBall.angVel.z,
        r.rsCar.ballHitInfo.isValid ? "Y" : "N",
        static_cast<unsigned long long>(r.rsCar.ballHitInfo.tickCountWhenHit),
        static_cast<unsigned long long>(r.rsCar.ballHitInfo.tickCountWhenExtraImpulseApplied),
        r.rsCar.ballHitInfo.extraHitVel.x,
        r.rsCar.ballHitInfo.extraHitVel.y,
        r.rsCar.ballHitInfo.extraHitVel.z);
    std::printf(
        "  dV   CARL car=%.2f,%.2f,%.2f ball=%.2f,%.2f,%.2f | RS car=%.2f,%.2f,%.2f ball=%.2f,%.2f,%.2f\n",
        r.carlCarDelta.x, r.carlCarDelta.y, r.carlCarDelta.z,
        r.carlBallDelta.x, r.carlBallDelta.y, r.carlBallDelta.z,
        r.rsCarDelta.x, r.rsCarDelta.y, r.rsCarDelta.z,
        r.rsBallDelta.x, r.rsBallDelta.y, r.rsBallDelta.z);
    std::printf(
        "  exp  CARL ball=%.2f,%.2f,%.2f | RS ball=%.2f,%.2f,%.2f\n",
        r.carlBallExpectedDv.x, r.carlBallExpectedDv.y, r.carlBallExpectedDv.z,
        r.rsBallExpectedDv.x, r.rsBallExpectedDv.y, r.rsBallExpectedDv.z);
    std::printf(
        "  con  CARL ball=%.2f,%.2f,%.2f mag=%.2f | RS ball=%.2f,%.2f,%.2f mag=%.2f\n",
        r.carlBallContactDv.x, r.carlBallContactDv.y, r.carlBallContactDv.z,
        r.carlBallContactDv.len(),
        r.rsBallContactDv.x, r.rsBallContactDv.y, r.rsBallContactDv.z,
        r.rsBallContactDv.Length());
    std::printf(
        "  pcor CARL ball=%.3f,%.3f,%.3f | RS ball=%.3f,%.3f,%.3f\n",
        r.carlBallPosCorr.x, r.carlBallPosCorr.y, r.carlBallPosCorr.z,
        r.rsBallPosCorr.x, r.rsBallPosCorr.y, r.rsBallPosCorr.z);
    std::printf(
        "  ERR  ballPos=%.3f ballVel=%.3f\n",
        r.ballPosError, r.ballVelError);
}

void runTrace(
    RLEnvironment& env,
    DiscreteControls* dActions,
    StateSnapshot* dSnapshot,
    const Scenario& s,
    const RocketSim::CarControls& rsControls,
    const DiscreteControls& carlControls,
    RocketSim::Arena* arena,
    RocketSim::Car* car,
    RocketSim::Ball* ball)
{
    constexpr float rsDt = 1.f / 120.f;
    const RocketSim::Vec rsGravity(0.f, 0.f, RocketSim::RLConst::GRAVITY_Z);
    const Vec3 carlGravity = WORLD_GRAVITY * PHYS_DT;
    const float dragFactor = std::pow(1.f - BALL_DRAG, PHYS_DT);

    captureStateKernel<<<1, 1, 0, env.getStream()>>>(
        env.getDeviceState(), dSnapshot);
    CUDA_CHECK(cudaGetLastError());

    StateSnapshot prevCarl;
    CUDA_CHECK(cudaMemcpyAsync(
        &prevCarl, dSnapshot, sizeof(StateSnapshot),
        cudaMemcpyDeviceToHost, env.getStream()));
    CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

    RocketSim::CarState prevRsCar = car->GetState();
    RocketSim::BallState prevRsBall = ball->GetState();

    std::vector<TraceRecord> records;
    records.reserve(s.totalTicks + 1);

    int firstContactCarl = -1;
    int firstContactRs = -1;
    int carlFloorContactTick = -1;
    int rsFloorContactTick = -1;

    ErrorStats err;

    std::printf("\n--- event trace: %s (full %d ticks) ---\n", s.name, s.totalTicks);

    for (int tick = 1; tick <= s.totalTicks; ++tick)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            dActions, &carlControls, sizeof(DiscreteControls),
            cudaMemcpyHostToDevice, env.getStream()));

        car->controls = rsControls;

        env.step(dActions);

        captureStateKernel<<<1, 1, 0, env.getStream()>>>(
            env.getDeviceState(), dSnapshot);
        CUDA_CHECK(cudaGetLastError());

        StateSnapshot carlSnap;
        CUDA_CHECK(cudaMemcpyAsync(
            &carlSnap, dSnapshot, sizeof(StateSnapshot),
            cudaMemcpyDeviceToHost, env.getStream()));
        CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

        arena->Step(1);

        const RocketSim::CarState rsCar = car->GetState();
        const RocketSim::BallState rsBall = ball->GetState();

        const Vec3 carlCarDelta = carlSnap.carVel - prevCarl.carVel - carlGravity;
        const Vec3 carlBallDelta = carlSnap.ballVel - prevCarl.ballVel - carlGravity;
        const RocketSim::Vec rsCarDelta = rsCar.vel - prevRsCar.vel - rsGravity * rsDt;
        const RocketSim::Vec rsBallDelta = rsBall.vel - prevRsBall.vel - rsGravity * rsDt;

        const Vec3 carlBallExpectedVel = prevCarl.ballVel * dragFactor + carlGravity;
        const Vec3 carlBallExpectedDv = carlBallExpectedVel - prevCarl.ballVel;
        const Vec3 carlBallContactDv = carlSnap.ballVel - carlBallExpectedVel;
        const Vec3 carlBallPosCorr = carlSnap.ballPos - prevCarl.ballPos - carlBallExpectedVel * PHYS_DT;

        const RocketSim::Vec rsBallExpectedVel = prevRsBall.vel * dragFactor + rsGravity * rsDt;
        const RocketSim::Vec rsBallExpectedDv = rsBallExpectedVel - prevRsBall.vel;
        const RocketSim::Vec rsBallContactDv = rsBall.vel - rsBallExpectedVel;
        const RocketSim::Vec rsBallPosCorr = rsBall.pos - prevRsBall.pos - rsBallExpectedVel * rsDt;

        const float ballPosError = vecDiff(carlSnap.ballPos, rsBall.pos);
        const float ballVelError = vecDiff(carlSnap.ballVel, rsBall.vel);

        err.carPos = std::max(err.carPos, vecDiff(carlSnap.carPos, rsCar.pos));
        err.carVel = std::max(err.carVel, vecDiff(carlSnap.carVel, rsCar.vel));
        err.carAng = std::max(err.carAng, vecDiff(carlSnap.carAng, rsCar.angVel));
        err.carRot = std::max(err.carRot, rotDiff(carlSnap.carRot, rsCar.rotMat));
        err.ballPos = std::max(err.ballPos, ballPosError);
        err.ballVel = std::max(err.ballVel, ballVelError);
        err.ballAng = std::max(err.ballAng, vecDiff(carlSnap.ballAng, rsBall.angVel));
        err.carlBallSpeed = std::max(err.carlBallSpeed, carlSnap.ballVel.len());
        err.rsBallSpeed = std::max(err.rsBallSpeed, rsBall.vel.Length());
        err.carlBallAngSpeed = std::max(err.carlBallAngSpeed, carlSnap.ballAng.len());
        err.rsBallAngSpeed = std::max(err.rsBallAngSpeed, rsBall.angVel.Length());

        TraceRecord rec;
        rec.tick = tick;
        rec.carl = carlSnap;
        rec.rsCar = rsCar;
        rec.rsBall = rsBall;
        rec.carlCarDelta = carlCarDelta;
        rec.carlBallDelta = carlBallDelta;
        rec.rsCarDelta = rsCarDelta;
        rec.rsBallDelta = rsBallDelta;
        rec.carlBallExpectedDv = carlBallExpectedDv;
        rec.carlBallContactDv = carlBallContactDv;
        rec.carlBallPosCorr = carlBallPosCorr;
        rec.rsBallExpectedDv = rsBallExpectedDv;
        rec.rsBallContactDv = rsBallContactDv;
        rec.rsBallPosCorr = rsBallPosCorr;
        rec.ballPosError = ballPosError;
        rec.ballVelError = ballVelError;
        rec.isEvent = false;

        std::string eventDesc;

        if (firstContactCarl < 0 && carlSnap.ballContactTick >= 0)
        {
            firstContactCarl = tick;
            eventDesc += "CARL_FIRST_CONTACT ";
            rec.isEvent = true;
        }

        if (firstContactRs < 0 && rsCar.ballHitInfo.isValid)
        {
            firstContactRs = tick;
            eventDesc += "RS_FIRST_CONTACT ";
            rec.isEvent = true;
        }

        if (!records.empty())
        {
            const TraceRecord& prev = records.back();

            if (carlSnap.ballContactTick != prev.carl.ballContactTick)
            {
                eventDesc += "CARL_bct_change ";
                rec.isEvent = true;
            }
            if (carlSnap.ballHitTick != prev.carl.ballHitTick)
            {
                eventDesc += "CARL_bht_change ";
                rec.isEvent = true;
            }
            if (rsCar.ballHitInfo.tickCountWhenHit != prev.rsCar.ballHitInfo.tickCountWhenHit)
            {
                eventDesc += "RS_hit_tick_change ";
                rec.isEvent = true;
            }
            if (rsCar.ballHitInfo.tickCountWhenExtraImpulseApplied != prev.rsCar.ballHitInfo.tickCountWhenExtraImpulseApplied)
            {
                eventDesc += "RS_extra_tick_change ";
                rec.isEvent = true;
            }
            if (ballVelError > prev.ballVelError + 5.0f)
            {
                eventDesc += "BALL_VEL_ERROR_JUMP ";
                rec.isEvent = true;
            }

            if (tick >= 139 && tick <= 225)
            {
                if (carlBallContactDv.len() > 1.0f || rsBallContactDv.Length() > 1.0f)
                {
                    eventDesc += "BALL_CONTACT ";
                    rec.isEvent = true;

                    const bool nearFloor = carlSnap.ballPos.z < 95.0f || rsBall.pos.z < 95.0f;
                    if (nearFloor && carlBallContactDv.z > 0.0f && carlFloorContactTick < 0)
                    {
                        carlFloorContactTick = tick;
                    }
                    if (nearFloor && rsBallContactDv.z > 0.0f && rsFloorContactTick < 0)
                    {
                        rsFloorContactTick = tick;
                    }
                }
            }
        }

        rec.eventDesc = eventDesc;
        records.push_back(rec);

        prevCarl = carlSnap;
        prevRsCar = rsCar;
        prevRsBall = rsBall;
    }

    int maxBallVelErrorTick = -1;
    float maxBallVelError = 0.f;
    for (const auto& r : records)
    {
        if (r.ballVelError > maxBallVelError)
        {
            maxBallVelError = r.ballVelError;
            maxBallVelErrorTick = r.tick;
        }
    }

    if (maxBallVelErrorTick >= 1)
    {
        records[maxBallVelErrorTick - 1].isEvent = true;
        records[maxBallVelErrorTick - 1].eventDesc += "MAX_BALL_VEL_ERROR ";
    }

    std::vector<bool> printed(records.size(), false);
    for (size_t i = 0; i < records.size(); ++i)
    {
        if (!records[i].isEvent) continue;

        int start = static_cast<int>(i) - 2;
        if (start < 0) start = 0;
        int end = static_cast<int>(i) + 2;
        if (end >= static_cast<int>(records.size())) end = static_cast<int>(records.size()) - 1;

        for (int j = start; j <= end; ++j)
        {
            if (!printed[j])
            {
                printTraceRecord(records[j], records[j].isEvent);
                printed[j] = true;
            }
        }
    }

    std::printf("\n--- summary ---\n");
    std::printf("first contact: CARL=%d RS=%d\n", firstContactCarl, firstContactRs);
    std::printf("max ball velocity error: %.3f at tick %d\n", maxBallVelError, maxBallVelErrorTick);

    std::printf("\n--- floor-bounce analysis (ticks 139-225) ---\n");
    std::printf("first shallow ball-floor contact: CARL=%d RS=%d\n", carlFloorContactTick, rsFloorContactTick);

    if (carlFloorContactTick >= 1 && carlFloorContactTick <= static_cast<int>(records.size()))
    {
        const TraceRecord& r = records[carlFloorContactTick - 1];
        std::printf("CARL floor impulse at tick %d: (%.2f, %.2f, %.2f) mag=%.2f, posCorr=(%.3f, %.3f, %.3f)\n",
            carlFloorContactTick,
            r.carlBallContactDv.x, r.carlBallContactDv.y, r.carlBallContactDv.z, r.carlBallContactDv.len(),
            r.carlBallPosCorr.x, r.carlBallPosCorr.y, r.carlBallPosCorr.z);
    }
    if (rsFloorContactTick >= 1 && rsFloorContactTick <= static_cast<int>(records.size()))
    {
        const TraceRecord& r = records[rsFloorContactTick - 1];
        std::printf("RS   floor impulse at tick %d: (%.2f, %.2f, %.2f) mag=%.2f, posCorr=(%.3f, %.3f, %.3f)\n",
            rsFloorContactTick,
            r.rsBallContactDv.x, r.rsBallContactDv.y, r.rsBallContactDv.z, r.rsBallContactDv.Length(),
            r.rsBallPosCorr.x, r.rsBallPosCorr.y, r.rsBallPosCorr.z);
    }

    const int bothBouncedTick = std::max(carlFloorContactTick, rsFloorContactTick);
    if (bothBouncedTick >= 1 && bothBouncedTick <= static_cast<int>(records.size()))
    {
        const TraceRecord& r = records[bothBouncedTick - 1];
        std::printf("remaining error after both bounced (tick %d): ballPos=%.3f ballVel=%.3f\n",
            bothBouncedTick, r.ballPosError, r.ballVelError);
    }

    std::printf("\n%-30s %4s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
        (std::string(s.name) + " (trace)").c_str(),
        "-",
        err.carPos, err.carVel, err.carAng, err.carRot,
        err.ballPos, err.ballVel, err.ballAng,
        err.carlBallSpeed, err.rsBallSpeed,
        err.carlBallAngSpeed, err.rsBallAngSpeed);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string filter;
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--trace") == 0)
            {
                g_traceMode = true;
            }
            else if (filter.empty())
            {
                filter = argv[i];
            }
        }

        RocketSim::Init(ROCKETSIM_COLLISION_MESH_ROOT);

        std::vector<Scenario> scenarios = {
            {
                "grounded rest",
                240, 0, false,
                {0.f, 2000.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, 0.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "drive straight",
                240, 0, false,
                {0.f, 2000.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, 0.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "ball wall from rest",
                240, 0, false,
                {0.f, 0.f, BALL_REST_Z},
                {3000.f, 0.f, 0.f},
                Vec3::zero(),
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "ball corner from rest",
                240, 0, false,
                {0.f, 0.f, BALL_REST_Z},
                {2500.f, 2500.f, 800.f},
                Vec3::zero(),
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "isolated side wall bounce",
                240, 0, false,
                {2000.f, 0.f, 1000.f},
                {2000.f, 0.f, 0.f},
                Vec3::zero(),
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "curved corner bounce",
                300, 0, false,
                {0.f, 0.f, 800.f},
                {2200.f, 3000.f, 300.f},
                Vec3::zero(),
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "replay curved corner bounce",
                4, 0, false,
                {-3951.9f, 3513.9f, 1261.1f},
                {-2262.8f, -666.1f, -100.8f},
                {-2.403f, -4.358f, -3.351f},
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "replay back wall bounce",
                4, 0, false,
                {-289.9683f, 5009.8071f, 653.5578f},
                {-295.8386f, 1130.0035f, 626.5620f},
                {-5.2210f, 1.3350f, -1.7778f},
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "replay back wall no spin",
                4, 0, false,
                {-289.9683f, 5009.8071f, 653.5578f},
                {-295.8386f, 1130.0035f, 626.5620f},
                Vec3::zero(),
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "diaz side wall bounce",
                4, 0, false,
                {3978.2136f, 1290.3167f, 676.5325f},
                {2019.3026f, 461.3783f, -299.4360f},
                {-3.2065f, 0.5611f, 5.0401f},
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "diaz back wall pre-bounce",
                4, 0, false,
                {89.3319f, 4963.0225f, 776.7418f},
                {1495.0435f, 1606.4583f, -389.4022f},
                {-0.2003f, -2.7157f, 2.6681f},
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "diaz corner pre-bounce",
                4, 0, false,
                {3345.5166f, 4502.4385f, 191.9960f},
                {1922.9760f, -994.6473f, -310.8632f},
                {1.8175f, 1.0434f, 5.5744f},
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "diaz g04 floor bounce",
                4, 0, false,
                {3952.5571f, -1806.1447f, 197.6181f},
                {-63.5163f, -519.1342f, -1879.7622f},
                {-1.0243f, -5.5882f, 1.9293f},
                {0.f, 2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "diaz g04 side wall bounce",
                4, 0, false,
                {-3961.3650f, 363.8687f, 237.0670f},
                {-764.7848f, 575.5236f, 268.7508f},
                {-0.1331f, -5.2124f, 1.5843f},
                {0.f, 2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "ball rolling on floor",
                600, 0, false,
                {-2000.f, 0.f, BALL_REST_Z},
                {1500.f, 0.f, 0.f},
                Vec3::zero(),
                {0.f, -2000.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                1, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "isolated car-ball ground",
                240, 0, false,
                {0.f, 0.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, -300.f, CAR_REST_Z},
                Quat::angle(PI / 2.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "wall pinch",
                80, 0, false,
                {4000.f, 0.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {3700.f, 0.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "offset wall pinch",
                80, 0, false,
                {4000.f, 500.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {3700.f, 420.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "drive straight reset",
                240, 120, false,
                {0.f, 2000.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, 0.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "wall pinch reset",
                240, 120, false,
                {4000.f, 0.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {3700.f, 0.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "wall pinch contact reset",
                240, 0, true,
                {4000.f, 0.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {3700.f, 0.f, CAR_REST_Z},
                Quat::angle(0.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "car-ball ground contact reset",
                240, 0, true,
                {0.f, 0.f, BALL_REST_Z},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, -300.f, CAR_REST_Z},
                Quat::angle(PI / 2.f),
                Vec3::zero(),
                Vec3::zero(),
                0, 1.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "highspeed head-on car-ball",
                4, 0, false,
                {0.f, 0.f, 500.f},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, -100.f, 479.245f},
                Quat::angle(PI / 2.f),
                {0.f, 15000.f, 0.f},
                Vec3::zero(),
                0, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "moderate head-on car-ball",
                4, 0, false,
                {0.f, 0.f, 500.f},
                Vec3::zero(),
                Vec3::zero(),
                {0.f, -100.f, 479.245f},
                Quat::angle(PI / 2.f),
                {0.f, 1000.f, 0.f},
                Vec3::zero(),
                0, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "moderate grazing car-ball",
                4, 0, false,
                {0.f, 0.f, 500.f},
                Vec3::zero(),
                Vec3::zero(),
                {50.f, -100.f, 479.245f},
                Quat::angle(PI / 2.f),
                {0.f, 1000.f, 0.f},
                Vec3::zero(),
                0, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            },
            {
                "grazing car-ball spin",
                4, 0, false,
                {0.f, 0.f, 500.f},
                Vec3::zero(),
                Vec3::zero(),
                {50.f, -100.f, 479.245f},
                Quat::angle(PI / 2.f),
                {0.f, 15000.f, 0.f},
                Vec3::zero(),
                0, 0.f, 0.f, 0.f, 0.f, 0.f, false, false, false
            }
        };

        if (!filter.empty())
        {
            std::vector<Scenario> filtered;
            for (const auto& s : scenarios)
            {
                if (std::strstr(s.name, filter.c_str()) != nullptr)
                {
                    filtered.push_back(s);
                }
            }
            scenarios = filtered;
        }

        if (scenarios.empty())
        {
            std::printf("No scenarios matched filter '%s'\n", filter.c_str());
            return 0;
        }

        RLEnvironment env(1, 1, 0, 0);

        DiscreteControls* dActions = nullptr;
        CUDA_CHECK(cudaMalloc(&dActions, sizeof(DiscreteControls)));

        StateSnapshot* dSnapshot = nullptr;
        CUDA_CHECK(cudaMalloc(&dSnapshot, sizeof(StateSnapshot)));

        constexpr int groupTicks = 4;

        std::printf("%-30s %4s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
            "scenario",
            "tick",
            "car_pos", "car_vel", "car_ang", "car_rot",
            "ball_pos", "ball_vel", "ball_ang",
            "carl_bspd", "rs_bspd", "carl_bang", "rs_bang");

        for (const auto& s : scenarios)
        {
            env.reset();
            setupScenarioKernel<<<1, 1, 0, env.getStream()>>>(
                env.getDeviceState(), env.getDeviceWorkspace(), s);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

            RocketSim::Arena* arena = RocketSim::Arena::Create(
                RocketSim::GameMode::SOCCAR);
            RocketSim::Car* car = arena->AddCar(
                RocketSim::Team::BLUE, RocketSim::CAR_CONFIG_OCTANE);
            RocketSim::Ball* ball = arena->ball;

            RocketSim::CarState rsCarState = {};
            rsCarState.pos = toRocketVec(s.carPos);
            rsCarState.rotMat = quatToRotMat(s.carRot);
            rsCarState.vel = toRocketVec(s.carVel);
            rsCarState.angVel = toRocketVec(s.carAng);
            rsCarState.isOnGround = true;
            rsCarState.isDemoed = s.carDemoed != 0;
            rsCarState.demoRespawnTimer = s.carDemoed != 0
                ? 1e6f
                : 0.f;
            rsCarState.boost = BOOST_SPAWN_AMOUNT;
            car->SetState(rsCarState);

            RocketSim::BallState rsBallState = {};
            rsBallState.pos = toRocketVec(s.ballPos);
            rsBallState.vel = toRocketVec(s.ballVel);
            rsBallState.angVel = toRocketVec(s.ballAng);
            ball->SetState(rsBallState);

            const RocketSim::CarControls rsControls = makeRocketControls(s);
            car->controls = rsControls;

            const DiscreteControls carlControls = makeDiscreteControls(s);

            if (s.contactReset)
            {
                int triggerTick = -1;
                StateSnapshot triggerSnap;

                for (int tick = 0; tick < s.totalTicks; tick += groupTicks)
                {
                    CUDA_CHECK(cudaMemcpyAsync(
                        dActions, &carlControls, sizeof(DiscreteControls),
                        cudaMemcpyHostToDevice, env.getStream()));

                    for (int i = 0; i < groupTicks; i++)
                    {
                        env.step(dActions);
                    }

                    captureStateKernel<<<1, 1, 0, env.getStream()>>>(
                        env.getDeviceState(), dSnapshot);
                    CUDA_CHECK(cudaGetLastError());

                    StateSnapshot carlSnap;
                    CUDA_CHECK(cudaMemcpyAsync(
                        &carlSnap, dSnapshot, sizeof(StateSnapshot),
                        cudaMemcpyDeviceToHost, env.getStream()));
                    CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

                    car->controls = rsControls;
                    arena->Step(groupTicks);

                    if (triggerTick < 0 && carlSnap.ballContactTick >= 0)
                    {
                        triggerTick = tick + groupTicks;
                        triggerSnap = carlSnap;
                        break;
                    }
                }

                if (triggerTick < 0)
                {
                    std::printf("%-30s %4s no contact in %d ticks\n",
                        s.name, "-", s.totalTicks);
                }
                else
                {
                    const int postTicks = s.totalTicks - triggerTick;

                    const ErrorStats envioErr = runPostResetSegment(
                        env, dActions, dSnapshot, triggerSnap,
                        false, -1, postTicks,
                        rsControls, carlControls, groupTicks);

                    const ErrorStats contactErr = runPostResetSegment(
                        env, dActions, dSnapshot, triggerSnap,
                        true, triggerTick, postTicks,
                        rsControls, carlControls, groupTicks);

                    std::printf("%-30s %4d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        (std::string(s.name) + " (envio)").c_str(),
                        triggerTick,
                        envioErr.carPos, envioErr.carVel, envioErr.carAng, envioErr.carRot,
                        envioErr.ballPos, envioErr.ballVel, envioErr.ballAng,
                        envioErr.carlBallSpeed, envioErr.rsBallSpeed,
                        envioErr.carlBallAngSpeed, envioErr.rsBallAngSpeed);

                    std::printf("%-30s %4d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        (std::string(s.name) + " (contact-aware)").c_str(),
                        triggerTick,
                        contactErr.carPos, contactErr.carVel, contactErr.carAng, contactErr.carRot,
                        contactErr.ballPos, contactErr.ballVel, contactErr.ballAng,
                        contactErr.carlBallSpeed, contactErr.rsBallSpeed,
                        contactErr.carlBallAngSpeed, contactErr.rsBallAngSpeed);
                }
            }
            else
            {
                if (isTraceTarget(s))
                {
                    runTrace(env, dActions, dSnapshot, s, rsControls, carlControls, arena, car, ball);
                }
                else
                {
                    ErrorStats err[2];
                    int phase = 0;

                    for (int tick = 0; tick < s.totalTicks; tick += groupTicks)
                    {
                    CUDA_CHECK(cudaMemcpyAsync(
                        dActions, &carlControls, sizeof(DiscreteControls),
                        cudaMemcpyHostToDevice, env.getStream()));

                    car->controls = rsControls;

                    StateSnapshot carlSnap;

                    for (int i = 0; i < groupTicks; i++)
                    {
                        env.step(dActions);

                        captureStateKernel<<<1, 1, 0, env.getStream()>>>(
                            env.getDeviceState(), dSnapshot);
                        CUDA_CHECK(cudaGetLastError());

                        CUDA_CHECK(cudaMemcpyAsync(
                            &carlSnap, dSnapshot, sizeof(StateSnapshot),
                            cudaMemcpyDeviceToHost, env.getStream()));
                        CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

                        arena->Step(1);

                        const RocketSim::CarState rsCar = car->GetState();
                        const RocketSim::BallState rsBall = ball->GetState();

                        err[phase].carlBallSpeed = std::max(err[phase].carlBallSpeed, carlSnap.ballVel.len());
                        err[phase].rsBallSpeed = std::max(err[phase].rsBallSpeed, rsBall.vel.Length());
                        err[phase].carlBallAngSpeed = std::max(err[phase].carlBallAngSpeed, carlSnap.ballAng.len());
                        err[phase].rsBallAngSpeed = std::max(err[phase].rsBallAngSpeed, rsBall.angVel.Length());
                    }

                    const RocketSim::CarState rsCar = car->GetState();
                    const RocketSim::BallState rsBall = ball->GetState();

                    err[phase].carPos = std::max(err[phase].carPos, vecDiff(carlSnap.carPos, rsCar.pos));
                    err[phase].carVel = std::max(err[phase].carVel, vecDiff(carlSnap.carVel, rsCar.vel));
                    err[phase].carAng = std::max(err[phase].carAng, vecDiff(carlSnap.carAng, rsCar.angVel));
                    err[phase].carRot = std::max(err[phase].carRot, rotDiff(carlSnap.carRot, rsCar.rotMat));
                    err[phase].ballPos = std::max(err[phase].ballPos, vecDiff(carlSnap.ballPos, rsBall.pos));
                    err[phase].ballVel = std::max(err[phase].ballVel, vecDiff(carlSnap.ballVel, rsBall.vel));
                    err[phase].ballAng = std::max(err[phase].ballAng, vecDiff(carlSnap.ballAng, rsBall.angVel));

                    if (s.resetTick > 0 && tick + groupTicks == s.resetTick)
                    {
                        resetStateKernel<<<1, 1, 0, env.getStream()>>>(
                            env.getDeviceState(), env.getDeviceWorkspace(), carlSnap,
                            false, -1);
                        CUDA_CHECK(cudaGetLastError());
                        CUDA_CHECK(cudaStreamSynchronize(env.getStream()));

                        RocketSim::CarState rsResetCar = {};
                        rsResetCar.pos = toRocketVec(carlSnap.carPos);
                        rsResetCar.rotMat = quatToRotMat(carlSnap.carRot);
                        rsResetCar.vel = toRocketVec(carlSnap.carVel);
                        rsResetCar.angVel = toRocketVec(carlSnap.carAng);
                        rsResetCar.isOnGround = carlSnap.carInternal.isOnGround;
                        rsResetCar.isDemoed = carlSnap.carDemoed != 0;
                        rsResetCar.demoRespawnTimer = carlSnap.carDemoed != 0
                            ? 1e6f
                            : 0.f;
                        rsResetCar.boost = carlSnap.carInternal.boost;
                        car->SetState(rsResetCar);

                        RocketSim::BallState rsResetBall = {};
                        rsResetBall.pos = toRocketVec(carlSnap.ballPos);
                        rsResetBall.vel = toRocketVec(carlSnap.ballVel);
                        rsResetBall.angVel = toRocketVec(carlSnap.ballAng);
                        ball->SetState(rsResetBall);

                        phase = 1;
                    }
                }

                if (s.resetTick > 0)
                {
                    std::printf("%-30s %4d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        (std::string(s.name) + " (pre)").c_str(),
                        s.resetTick,
                        err[0].carPos, err[0].carVel, err[0].carAng, err[0].carRot,
                        err[0].ballPos, err[0].ballVel, err[0].ballAng,
                        err[0].carlBallSpeed, err[0].rsBallSpeed,
                        err[0].carlBallAngSpeed, err[0].rsBallAngSpeed);
                    std::printf("%-30s %4d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        (std::string(s.name) + " (post)").c_str(),
                        s.resetTick,
                        err[1].carPos, err[1].carVel, err[1].carAng, err[1].carRot,
                        err[1].ballPos, err[1].ballVel, err[1].ballAng,
                        err[1].carlBallSpeed, err[1].rsBallSpeed,
                        err[1].carlBallAngSpeed, err[1].rsBallAngSpeed);
                }
                else
                {
                    std::printf("%-30s %4s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        s.name,
                        "-",
                        err[0].carPos, err[0].carVel, err[0].carAng, err[0].carRot,
                        err[0].ballPos, err[0].ballVel, err[0].ballAng,
                        err[0].carlBallSpeed, err[0].rsBallSpeed,
                        err[0].carlBallAngSpeed, err[0].rsBallAngSpeed);
                }
                }
            }

            delete arena;
        }

        CUDA_CHECK(cudaFree(dActions));
        CUDA_CHECK(cudaFree(dSnapshot));

        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "parity harness failed: %s\n", e.what());
        return 1;
    }
}
