#pragma once

#include "State/GameState.cuh"
#include "RLConstants.cuh"
#include "State/Events.cuh"

CARL_D CARL_FI void detectGoal(
    GameState* __restrict__ state,
    int simIdx,
    const Vec3& ballPos,
    EventBuffer events = {})
{
    GoalState& goal = state->goals[simIdx];

    const float threshold = SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y + BALL_RADIUS;

    const int scorer = ballPos.y >  threshold ?  1
                     : ballPos.y < -threshold ? -1 : 0;

    if (!scorer || goal.lastScorer != 0) return;

    if (events.enabled())
    {
        events.ball(simIdx, EVENT_GOAL);
        events.goalSide[simIdx] = scorer;
        events.ballSurfaces[simIdx] = -1;
    }
    
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
}
