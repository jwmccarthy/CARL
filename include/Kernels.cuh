#pragma once

#include "State/GameState.cuh"
#include "State/Workspace.cuh"
#include "Arena/ArenaMesh.cuh"

__global__ void resetKernel(GameState* __restrict__ state);

__global__ void beginStepKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space);

__global__ void carControlsKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    const DiscreteControls* __restrict__ actions);

__global__ void carArenaBroadPhaseKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena);

__global__ void carSuspensionRaycastKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena);

__global__ void carArenaSATKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena);

__global__ void carArenaClipKernel(
    ArenaMesh* __restrict__ arena,
    Workspace* __restrict__ space);

__global__ void carArenaSolveKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena);

__global__ void carCarBallSolveKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena);

__global__ void integrateCarsKernel(GameState* __restrict__ state);

__global__ void applyImpulseCacheKernel(GameState* __restrict__ state);

__global__ void boostPadKernel(GameState* __restrict__ state);
