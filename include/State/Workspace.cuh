#pragma once

#include "../RLConstants.cuh"
#include "../Cuda/Math.cuh"
#include "../Cuda/DeviceArray.cuh"
#include "../Physics/SuspensionUtils.cuh"

struct BroadPhase
{
    DeviceArray<int> cellIdx;
    DeviceArray<int> numTris;
    DeviceArray<int> triPrefix;

    BroadPhase(int nTotalCars)
        : cellIdx(nTotalCars)
        , numTris(nTotalCars + 1)
        , triPrefix(nTotalCars + 1)
    {}

    void reset() { numTris.fill(0); }
};

struct CarTriNarrow
{
    const int maxCarTriPairs;

    DeviceArray<Vec3> pairV0;
    DeviceArray<Vec3> pairV1;
    DeviceArray<Vec3> pairV2;

    DeviceArray<int>   triIdx;
    DeviceArray<int>   axisIdx;
    DeviceArray<Vec3>  minAxis;
    DeviceArray<float> minPen;

    DeviceArray<int> conPairCount;

    CarTriNarrow(int nTotalCars)
        : maxCarTriPairs(nTotalCars * MAX_CAR_TRI_PAIRS)
        , pairV0(maxCarTriPairs)
        , pairV1(maxCarTriPairs)
        , pairV2(maxCarTriPairs)
        , triIdx(maxCarTriPairs)
        , axisIdx(maxCarTriPairs)
        , minAxis(maxCarTriPairs)
        , minPen(maxCarTriPairs)
        , conPairCount(maxCarTriPairs)
    {}

    void reset() { conPairCount.fill(0); }
};

struct CarTriContact
{
    const int maxContacts;

    DeviceArray<Vec3>  carPos;
    DeviceArray<Vec3>  arenaPos;
    DeviceArray<Vec3>  normal;
    DeviceArray<float> depth;

    CarTriContact(int maxCarTriPairs)
        : maxContacts(maxCarTriPairs * MAX_PAIR_CONTACTS)
        , carPos(maxContacts)
        , arenaPos(maxContacts)
        , normal(maxContacts)
        , depth(maxContacts)
    {}
};

struct CarTriHit
{
    DeviceArray<int> carHitStart;
    DeviceArray<int> carHitCount;

    CarTriHit(int nTotalCars)
        : carHitStart(nTotalCars)
        , carHitCount(nTotalCars)
    {}

    void reset() { carHitCount.fill(0); }
};

struct CarTriManifold
{
    const int maxManifoldPoints;

    DeviceArray<Vec3> carPos;
    DeviceArray<Vec3> worldPos;
    DeviceArray<Vec3> worldNormal;

    DeviceArray<float> dist;
    DeviceArray<float> friction;
    DeviceArray<float> restitution;
    DeviceArray<float> impulse;
    DeviceArray<float> impulseTan;

    DeviceArray<int> lifetime;
    DeviceArray<int> count;

    CarTriManifold(int nTotalCars)
        : maxManifoldPoints(nTotalCars * MAX_CAR_MANIFOLD_POINTS)
        , carPos(maxManifoldPoints)
        , worldPos(maxManifoldPoints)
        , worldNormal(maxManifoldPoints)
        , dist(maxManifoldPoints)
        , friction(maxManifoldPoints)
        , restitution(maxManifoldPoints)
        , impulse(maxManifoldPoints)
        , impulseTan(maxManifoldPoints)
        , lifetime(maxManifoldPoints)
        , count(nTotalCars)
    {}

    void reset() { count.fill(0); }
};

struct CarCarManifold
{
    const int maxPairsPerSim;
    const int totalPairs;
    const int totalPoints;

    DeviceArray<Vec3>  carPosA;
    DeviceArray<Vec3>  carPosB;
    DeviceArray<Vec3>  normal;

    DeviceArray<float> dist;
    DeviceArray<float> impulse;
    DeviceArray<float> impulseTan;

    DeviceArray<int> lifetime;
    DeviceArray<int> count;

    CarCarManifold(int nSim, int nCars)
        : maxPairsPerSim(nCars * (nCars - 1) / 2)
        , totalPairs(nSim * maxPairsPerSim)
        , totalPoints(totalPairs * MAX_CAR_MANIFOLD_POINTS)
        , carPosA(totalPoints)
        , carPosB(totalPoints)
        , normal(totalPoints)
        , dist(totalPoints)
        , impulse(totalPoints)
        , impulseTan(totalPoints)
        , lifetime(totalPoints)
        , count(totalPairs)
    {}

    void reset() { count.fill(0); }
};

struct CarSuspension
{
    const int totalWheels;

    DeviceArray<float> rayDist;
    DeviceArray<Vec3>  rayNormal;

    DeviceArray<float> brakeFactor;
    DeviceArray<float> brakeFactorPrev;
    DeviceArray<float> throttleVal;
    DeviceArray<float> rawThrottle;
    DeviceArray<float> handbrakeVal;
    DeviceArray<float> steerAngle;
    DeviceArray<float> latFrictionPrev;
    DeviceArray<float> lonFrictionPrev;

    DeviceArray<Vec3>  jumpImpulse;
    DeviceArray<float> engineDrive;
    DeviceArray<float> engineDrivePrev;

    CarSuspension(int nTotalCars)
        : totalWheels(nTotalCars * NUM_WHEELS)
        , rayDist(totalWheels)
        , rayNormal(totalWheels)
        , brakeFactor(nTotalCars)
        , brakeFactorPrev(nTotalCars)
        , throttleVal(nTotalCars)
        , rawThrottle(nTotalCars)
        , handbrakeVal(nTotalCars)
        , steerAngle(nTotalCars)
        , latFrictionPrev(totalWheels)
        , lonFrictionPrev(totalWheels)
        , jumpImpulse(nTotalCars)
        , engineDrive(nTotalCars)
        , engineDrivePrev(nTotalCars)
    {}

    void reset()
    {
        latFrictionPrev.fill(0);
        lonFrictionPrev.fill(0);

        brakeFactor.fill(0);
        brakeFactorPrev.fill(0);
        throttleVal.fill(0);
        rawThrottle.fill(0);
        handbrakeVal.fill(0);
        steerAngle.fill(0);

        jumpImpulse.fill(0);
        engineDrive.fill(0);
        engineDrivePrev.fill(0);
    }
};

struct Workspace
{    
    BroadPhase     bp;
    CarTriNarrow   ctNrw;
    CarTriContact  ctCon;
    CarTriHit      ctHit;
    CarTriManifold ctMan;
    CarCarManifold ccMan;
    CarSuspension  susp;

    Workspace(int nSim, int nCars, int nTotalCars)
        : bp(nTotalCars)
        , ctNrw(nTotalCars)
        , ctCon(ctNrw.maxCarTriPairs)
        , ctHit(nTotalCars)
        , ctMan(nTotalCars)
        , ccMan(nSim, nCars)
        , susp(nTotalCars)
    {}

    void reset()
    {
        bp.reset();
        ctNrw.reset();
        ctHit.reset();
        ctMan.reset();
        ccMan.reset();
        susp.reset();
    }
};
