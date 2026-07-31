#pragma once

#include "../Cuda/Random.cuh"
#include "GameState.cuh"
#include "Workspace.cuh"
#include "../RLConstants.cuh"

CARL_D CARL_FI void resetBall(BallState* __restrict__ ball, int simIdx)
{
    ball->pos[simIdx] = { 0.f, 0.f, BALL_REST_Z };
    ball->vel[simIdx] = Vec3::zero();
    ball->ang[simIdx] = Vec3::zero();
}

CARL_D CARL_FI void resetCar(CarState* __restrict__ cars, int carIdx, int locIdx, bool invert)
{
    CarSpawn loc = KICKOFF_LOCATIONS[locIdx % 5];

    // Orange mirrors both horizontal coordinates and rotates by half a turn
    float x   = invert ? -loc.x       : loc.x;
    float y   = invert ? -loc.y       : loc.y;
    float yaw = invert ? loc.yaw + PI : loc.yaw;

    cars->pos[carIdx] = { x, y, loc.z };
    cars->vel[carIdx] = Vec3::zero();
    cars->ang[carIdx] = Vec3::zero();
    cars->rot[carIdx] = Quat::angle(yaw);
    cars->cen[carIdx] = cars->getHitboxCenter(carIdx);
    cars->imp[carIdx] = Vec3::zero();
    cars->isDemoed[carIdx] = 0;
    cars->demoRespawnTimer[carIdx] = 0.f;
    cars->carContactIdx[carIdx] = -1;
    cars->carContactCooldown[carIdx] = 0.f;
    cars->ballHitTick[carIdx] = -1;
    cars->ballContactTick[carIdx] = -1;
    cars->controls[carIdx] = {};

    CarInternalState internal{};
    internal.isOnGround = true;
    internal.boost = BOOST_SPAWN_AMOUNT;
    cars->internal[carIdx] = internal;
}

CARL_D CARL_FI void resetToKickoff(GameState* __restrict__ state, int simIdx)
{
    const int nBlue = state->nBlue;
    const int nOrange = state->nOrange;
    const int nCars = state->nCars;

    // Both teams use corresponding entries from one shuffled location order
    const uint32_t seed = hash32((uint32_t)state->seed)
                        ^ hash32((uint32_t)simIdx)
                        ^ hash32((uint32_t)state->tickCount);
    const int permIdx = randomIndex(seed, 120);
    const int* carLocs = KICKOFF_PERMUTATIONS[permIdx];

    resetBall(&state->ball, simIdx);

    #pragma unroll 2
    for (int team = 0; team < 2; team++)
    {
        const bool invert = team;
        const int teamSize = team ? nOrange : nBlue;

        for (int i = 0; i < teamSize; i++)
        {
            const int locIdx = carLocs[i];
            const int carIdx = simIdx * nCars + (team * nBlue + i);

            resetCar(&state->cars, carIdx, locIdx, invert);
        }
    }
}

CARL_D CARL_FI void resetAfterDone(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    int simIdx,
    int maxTicks,
    int overtimeTimeoutTicks,
    int noTouchTimeoutTicks)
{
    GoalState& goal = state->goals[simIdx];
    const bool scored = goal.lastScorer != 0;
    const bool noTouchTimeout = noTouchTimeoutTicks > 0
        && state->tickCount - state->lastBallTouchTicks[simIdx]
            >= noTouchTimeoutTicks;
    const bool overtimeExpired = goal.overtime
        && state->episodeTicks[simIdx] >= maxTicks + overtimeTimeoutTicks;

    if (!goal.overtime && !scored && !noTouchTimeout
        && state->episodeTicks[simIdx] >= maxTicks
        && goal.blueScore == goal.orangeScore)
    {
        goal.overtime = true;
        return;
    }

    if (!scored && !noTouchTimeout && !overtimeExpired
        && (goal.overtime || state->episodeTicks[simIdx] < maxTicks)) return;

    goal = {};

    resetToKickoff(state, simIdx);
    state->episodeTicks[simIdx] = 0;
    state->lastBallTouchTicks[simIdx] = state->tickCount;

    const int carBase = simIdx * state->nCars;
    for (int local = 0; local < state->nCars; local++)
    {
        const int carIdx = carBase + local;
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
    }

    const int pairBase = simIdx * space->ccMan.maxPairsPerSim;

    for (int pair = 0; pair < space->ccMan.maxPairsPerSim; pair++)
    {
        space->ccMan.count[pairBase + pair] = 0;
    }

    for (int pad = 0; pad < NUM_BOOST_PADS; pad++)
    {
        state->boostPadCooldowns[simIdx * NUM_BOOST_PADS + pad] = 0.f;
    }
}
