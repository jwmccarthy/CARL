#pragma once

#include "State/GameState.cuh"
#include "RLConstants.cuh"

CARL_D CARL_FI void detectGoal(
    GameState* __restrict__ state,
    int simIdx,
    const Vec3& ballPos)
{
    GoalState& goal = state->goals[simIdx];

    const float threshold = SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y + BALL_RADIUS;

    const int scorer = ballPos.y >  threshold ?  1
                     : ballPos.y < -threshold ? -1 : 0;

    if (!scorer) return;
    
    if (scorer > 0)
    {
        goal.blueScore++;
    }
    else
    {
        goal.orangeScore++;
    }

    goal.lastScorer = scorer;
    goal.tick = state->tickCount;

    state->ball.pos[simIdx] = { 0.f, 0.f, BALL_REST_Z };
    state->ball.vel[simIdx] = Vec3::zero();
    state->ball.ang[simIdx] = Vec3::zero();
    state->ball.imp[simIdx] = Vec3::zero();
}
