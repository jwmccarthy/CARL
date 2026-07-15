#pragma once

#include "../RLConstants.cuh"
#include "../Cuda/Math.cuh"
#include "../Cuda/Common.cuh"
#include "../Cuda/DeviceArray.cuh"

struct CarControls
{
    float throttle, steer;
    float yaw, pitch, roll;
    int jump, boost, slide;
};

struct GoalState
{
    int blueScore;
    int orangeScore;
    int lastScorer;
    int tick;
};

struct BallState
{
    const int nBall;

    DeviceArray<Vec3> pos;
    DeviceArray<Vec3> vel;
    DeviceArray<Vec3> ang;
    DeviceArray<Vec3> imp;

    BallState(const int nBall)
        : nBall(nBall)
        , pos(nBall)
        , vel(nBall)
        , ang(nBall)
        , imp(nBall)
    {}

    void reset() { imp.fill(0); }
};

struct CarInternalState
{
    // Ground
    int   isOnGround;
    float airTime;
    float airTimeSinceJump;
    float handbrakeVal;

    // Jump
    int   hasJumped;
    int   isJumping;
    float jumpTime;
    int   lastJump;

    // Flip
    int   hasDoubleJumped;
    int   hasFlipped;
    int   isFlipping;
    float flipTime;

    // Auto-flip
    int   isAutoFlipping;
    float autoFlipTimer;
    float autoFlipTorqueScale;
    Vec3  flipRelTorque;

    // Demo respawn selection
    int respawnIdx;
    int lastRespawnDir;

    // Boost
    int   isBoosting;
    float boost;
    float boostingTime;
    float timeSinceBoosted;
};

struct CarState
{
    DeviceArray<Vec3>  pos;
    DeviceArray<Vec3>  vel;
    DeviceArray<Vec3>  ang;
    DeviceArray<Vec3>  cen;
    DeviceArray<Vec3>  imp;
    DeviceArray<Quat>  rot;
    DeviceArray<int>   isDemoed;
    DeviceArray<float> demoRespawnTimer;
    DeviceArray<int>   carContactIdx;       // Index of car last bumped
    DeviceArray<float> carContactCooldown;  // Ticks since ^^ occurred
    DeviceArray<int>   ballHitTick;         // Last extra impulse tick
    DeviceArray<int>   ballContactTick;     // Last live contact tick

    DeviceArray<CarControls>      controls;
    DeviceArray<CarInternalState> internal;

    CarState(int nTotalCars)
        : pos(nTotalCars)
        , vel(nTotalCars)
        , ang(nTotalCars)
        , cen(nTotalCars)
        , imp(nTotalCars)
        , rot(nTotalCars)
        , isDemoed(nTotalCars)
        , demoRespawnTimer(nTotalCars)
        , carContactIdx(nTotalCars)
        , carContactCooldown(nTotalCars)
        , ballHitTick(nTotalCars)
        , ballContactTick(nTotalCars)
        , controls(nTotalCars)
        , internal(nTotalCars)
    {
        controls.fill(0);
    }

    CARL_HD CARL_FI Vec3 getHitboxCenter(int i)
    {
        return pos[i] + rot[i].toWorld(CAR_OFFSETS);
    }

    void reset()
    {
        imp.fill(0);

        isDemoed.fill(0);
        demoRespawnTimer.fill(0);
        carContactIdx.fill(0xFF);
        carContactCooldown.fill(0);
        ballHitTick.fill(0xFF);
        ballContactTick.fill(0xFF);

        std::vector<CarInternalState> hInternal(internal.count);
        for (int i = 0; i < internal.count; i++)
        {
            hInternal[i].isOnGround = true;
            hInternal[i].boost = BOOST_SPAWN_AMOUNT;
        }

        CUDA_CHECK(cudaMemcpy(internal, hInternal.data(),
            internal.count * sizeof(CarInternalState), CARL_HToD));
    }
};

struct GameState
{
    const int nSim;
    const int nBlue;
    const int nOrange;
    const int nCars;
    const int nTotalCars;
    const int seed;

    int tickCount;

    BallState ball;
    CarState  cars;

    DeviceArray<float> boostPadCooldowns;
    DeviceArray<GoalState> goals;

    GameState(const int nSim, const int nBlue,
              const int nOrange, const int seed)
        : nSim(nSim)
        , nBlue(nBlue)
        , nOrange(nOrange)
        , nCars(nBlue + nOrange)
        , nTotalCars(nSim * nCars)
        , seed(seed)
        , tickCount(0)
        , ball(nSim)
        , cars(nTotalCars)
        , boostPadCooldowns(nSim * NUM_BOOST_PADS)
        , goals(nSim)
    {}

    void reset()
    {
        tickCount = 0;
        ball.reset();
        cars.reset();
        boostPadCooldowns.fill(0);
        goals.fill(0);
    }
};
