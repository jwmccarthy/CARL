#pragma once

#include "../RLConstants.cuh"
#include "../State/Workspace.cuh"
#include "CarArena/SolverBody.cuh"
#include "SuspensionUtils.cuh"

CARL_D CARL_FI float evalNonStickyFriction(const float normZ)
{
    if (normZ <= 0.f)    return 0.1f;
    if (normZ >= 1.f)    return 1.f;
    if (normZ < 0.7075f) return 0.1f + (normZ / 0.7075f) * 0.4f;

    return 0.5f + ((normZ - 0.7075f) / (1.f - 0.7075f)) * 0.5f;
}

CARL_D CARL_FI float evalLatFrictionCurve(const float input)
{
    return 1.f - 0.8f * clampf(input, 0.f, 1.f);
}

CARL_D CARL_FI float evalHandbrakeLongFrictionFactor(const float input)
{
    return 0.5f + 0.4f * clampf(input, 0.f, 1.f);
}

CARL_D CARL_FI float coastingBrakeFactor(const float absForwardSpeedUU)
{
    return absForwardSpeedUU < 25.f ? 1.f : 0.15f;
}

struct SuspensionWheelContribution
{
    Vec3 directVel;
    Vec3 directAng;
    Vec3 deltaVel;
    Vec3 contactNorm;
    bool hasContact;
};

struct SuspFrame
{
    Vec3 cenOffset;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    Vec3 vel;
    Vec3 ang;
    Vec3 angCrossUp;

    float brakeTorque;
    float pushbackDt;
    int carIdx;
};

struct SuspContactSummary
{
    Vec3 normalSum;
    int count;
};

CARL_D CARL_FI SuspensionWheelContribution computeSuspWheel(
    const SolverBody& body,
    CarSuspension& susp,
    const SuspFrame& frame,
    int wheel)
{
    SuspensionWheelContribution contribution{};

    const int wheelIdx = frame.carIdx * NUM_WHEELS + wheel;
    const float dist = __ldg(&susp.rayDist[wheelIdx]);
    if (dist > suspRayLength(wheel)) return contribution;

    const float radius = suspRadius(wheel);
    const float restLen = suspRestLength(wheel);
    const float suspLen = fminf(
        fmaxf(dist - radius, restLen - MAX_SUSPENSION_TRAVEL),
        restLen + MAX_SUSPENSION_TRAVEL);

    const Vec3 normal = Vec3::ldg(susp.rayNormal[wheelIdx]);
    const Vec3 hardPointRel =
        frame.cenOffset + body.rot.toWorld(suspConnection(wheel));
    const Vec3 fullRelPos = hardPointRel - frame.up * dist;
    const Vec3 pointVel = frame.vel + frame.ang.cross(fullRelPos);
    const float projVel = normal.dot(pointVel);
    const float denom = normal.dot(frame.up);
    const float clipped = denom > 0.1f ? 1.f / denom : 10.f;
    const float relVel = projVel * clipped;
    const float damping = relVel < 0.f
        ? SUSP_DAMPING_COMPRESSION
        : SUSP_DAMPING_RELAXATION;

    // The damping term receives the suspension clip factor exactly once
    float suspForce = ((restLen - suspLen) * SUSP_STIFFNESS - damping * projVel)
                    * clipped * suspForceScale(wheel);
    if (suspForce <= 0.f) suspForce = 0.f;

    // Spring impulse plus Bullet's extra wheel-penetration pushback
    float suspImpulse = 0.f;

    if (suspForce != 0.f)
    {
        suspImpulse = suspForce * PHYS_DT;

        const float pushbackThreshold = restLen + radius 
                                      - SUSPENSION_SUBTRACTION;

        if (dist < pushbackThreshold)
        {
            const float penetration = pushbackThreshold - dist;
            const Vec3 torqueAxis = fullRelPos.cross(normal);
            const Vec3 invInertiaTorque = applyInvInertiaWorld(body, torqueAxis);

            const float jacobian = CAR_INV_MASS 
                                 + invInertiaTorque.dot(torqueAxis);
            const float jacInv = 1.f / fmaxf(jacobian, 1e-8f);

            const float posError = SUSP_PUSHBACK_ERP * penetration 
                                 / frame.pushbackDt;
            const float velError = -projVel;

            float pushback = (posError + velError) * jacInv;
            if (pushback < 0.f) pushback = 0.f;

            suspImpulse += pushback * 0.25f;
        }
    }

    contribution.deltaVel = normal * (suspImpulse * CAR_INV_MASS);
    contribution.directAng = applyInvInertiaWorld(body, fullRelPos.cross(normal))
                           * suspImpulse;

    Vec3 lateralAxle = frame.right;
    if (wheel < 2)
    {
        const float steerAngle = __ldg(&susp.steerAngle[frame.carIdx]);
        if (fabsf(steerAngle) > 1e-6f)
        {
            float sinAngle, cosAngle;
            __sincosf(steerAngle, &sinAngle, &cosAngle);
            lateralAxle = frame.right * cosAngle 
                        + frame.up.cross(frame.right) * sinAngle;
        }
    }

    Vec3 axleDir = lateralAxle - normal * lateralAxle.dot(normal);
    const float axleLenSq = axleDir.lenSq();

    contribution.contactNorm = normal;
    contribution.hasContact = true;
    if (axleLenSq <= 1e-10f) return contribution;

    axleDir = axleDir * rsqrtf(axleLenSq);

    const Vec3 lateralRelPos =
        fullRelPos - frame.up * frame.up.dot(fullRelPos);
    const Vec3 hardPointVel = pointVel + frame.angCrossUp * dist;
    const Vec3 curveLongDir = lateralAxle.cross(normal);
    const Vec3 forwardDir = normal.cross(axleDir);

    float frictionCurveInput = 0.f;
    const float baseFriction = fabsf(hardPointVel.dot(lateralAxle));
    if (baseFriction > 5.f)
    {
        frictionCurveInput = baseFriction /
            (fabsf(hardPointVel.dot(curveLongDir)) + baseFriction);
    }

    const float handbrake = __ldg(&susp.handbrakeVal[frame.carIdx]);
    float lateralBase = evalLatFrictionCurve(frictionCurveInput);
    float longitudinalBase = 1.f;
    if (handbrake > 0.f)
    {
        lateralBase *= (0.1f - 1.f) * handbrake + 1.f;
        longitudinalBase *= (evalHandbrakeLongFrictionFactor(frictionCurveInput) - 1.f)
                          * handbrake + 1.f;
    }

    const float nonStickyScale = evalNonStickyFriction(normal.z);
    const bool sticky = __ldg(&susp.throttleVal[frame.carIdx]) != 0.f;
    const float frictionScale = sticky ? 1.f : nonStickyScale;
    const float lateralFriction = lateralBase * frictionScale;
    const float longitudinalFriction = longitudinalBase * frictionScale;

    // Friction coefficients are consumed one tick after they are calculated
    const float lateralFrictionUsed = __ldg(&susp.latFrictionPrev[wheelIdx]);
    const float longitudinalFrictionUsed = __ldg(&susp.lonFrictionPrev[wheelIdx]);
    susp.latFrictionPrev[wheelIdx] = lateralFriction;
    susp.lonFrictionPrev[wheelIdx] = longitudinalFriction;

    const Vec3 torqueAxis = fullRelPos.cross(axleDir);
    const Vec3 angularLateral = applyInvInertiaWorld(body, torqueAxis);
    const float jacobian = CAR_INV_MASS 
                         + axleDir.dot(angularLateral.cross(fullRelPos));
    const float latJacInv = 1.f / fmaxf(jacobian, 1e-8f);
    const float lateralVel = axleDir.dot(pointVel);

    // Bullet removes only a fraction of lateral slip per tick
    constexpr float CONTACT_DAMPING = 0.2f;
    const float sideImpulse =
        -CONTACT_DAMPING * lateralVel * latJacInv;

    const float longitudinalVel = forwardDir.dot(pointVel);

    // Rolling-friction calibration scale.
    constexpr float ROLLING_FRICTION_SCALE_MAGIC = 113.73963f;
    const float engineDrive = __ldg(&susp.engineDrivePrev[frame.carIdx]);
    const float rollingFriction = engineDrive != 0.f
        ? -0.75f * engineDrive
        : clampf(-longitudinalVel * ROLLING_FRICTION_SCALE_MAGIC,
                 -frame.brakeTorque, frame.brakeTorque);

    Vec3 wheelImpulse = axleDir * (sideImpulse * lateralFrictionUsed)
                      + forwardDir * (rollingFriction * longitudinalFrictionUsed);
    wheelImpulse = wheelImpulse * ((CAR_MASS / 3.f) * PHYS_DT);

    contribution.directVel = wheelImpulse * CAR_INV_MASS;
    contribution.directAng = contribution.directAng
        + applyInvInertiaWorld(body, lateralRelPos.cross(wheelImpulse));

    return contribution;
}

CARL_D CARL_FI void applySuspAggregate(
    SolverBody& body,
    CarSuspension& susp,
    const CarTriManifold& manifold,
    const SuspFrame& frame,
    const SuspContactSummary& contacts)
{
    const float throttle = __ldg(&susp.throttleVal[frame.carIdx]);
    const bool hasRawThrottle =
        __ldg(&susp.rawThrottle[frame.carIdx]) != 0.f;
    const int manifoldCount = __ldg(&manifold.count[frame.carIdx]);
    const bool hasWorldContact = manifoldCount != 0;

    Vec3 upwardsDir;
    bool applyAutoroll = false;

    // Apply sticky force whenever any suspension ray has contact.
    if (contacts.count > 0)
    {
        upwardsDir = contacts.normalSum
            * rsqrtf(contacts.normalSum.lenSq() + 1e-16f);

        const float forwardSpeed = fabsf(frame.vel.dot(frame.forward));
        const bool fullStick =
            throttle != 0.f || forwardSpeed > CAR_STOPPING_VEL;

        // Apply the base sticky force for the four-wheel Octane.
        float stickyScale = 0.5f;
        if (fullStick) stickyScale += 1.f - fabsf(upwardsDir.z);

        body.extVel = body.extVel
            + upwardsDir * (stickyScale * WORLD_GRAVITY.z * PHYS_DT);

        applyAutoroll =
            hasRawThrottle && (contacts.count < NUM_WHEELS || hasWorldContact);
    }
    else if (hasWorldContact)
    {
        upwardsDir = Vec3::ldg(
            manifold.worldNormal[frame.carIdx * MAX_CAR_MANIFOLD_POINTS]);
        applyAutoroll = hasRawThrottle;
    }
    else
    {
        return;
    }

    if (!applyAutoroll) return;

    // Align the car's up direction with the aggregate contact normal
    const Vec3 groundDownDir = upwardsDir.neg();
    const Vec3 crossRightDir = upwardsDir.cross(frame.forward);
    const Vec3 crossForwardDir = groundDownDir.cross(crossRightDir);
    const float rightTorqueFactor =
        1.f - clampf(frame.right.dot(crossRightDir), 0.f, 1.f);
    const float forwardTorqueFactor =
        1.f - clampf(frame.forward.dot(crossForwardDir), 0.f, 1.f);
    const Vec3 rightTorqueDir = frame.forward
        * (frame.right.dot(upwardsDir) >= 0.f ? -1.f : 1.f);
    const Vec3 forwardTorqueDir = frame.right
        * (frame.forward.dot(upwardsDir) >= 0.f ? 1.f : -1.f);

    body.deltaVel = body.deltaVel
        + groundDownDir * (CAR_AUTOROLL_FORCE * PHYS_DT);
    body.deltaAng = body.deltaAng
        + (forwardTorqueDir * forwardTorqueFactor
            + rightTorqueDir * rightTorqueFactor)
        * (CAR_AUTOROLL_TORQUE * PHYS_DT);
}

CARL_D __noinline__ void applyCarSuspension(
    SolverBody& body,
    Workspace* __restrict__ space,
    const int carIdx,
    const int tickCount)
{
    const float brakeFactor = __ldg(&space->susp.brakeFactorPrev[carIdx]);
    const float brakeTorque = brakeFactor * (CAR_MASS * (14.25f + 1.f / 3.f));

    const Vec3 up = body.rot.toWorld(WORLD_Z);
    const SuspFrame frame = {
        body.cen - body.pos,
        body.rot.toWorld(WORLD_X),
        body.rot.toWorld(WORLD_Y),
        up,
        body.vel,
        body.ang,
        body.ang.cross(up),
        brakeTorque,
        tickCount <= 1 ? SUSP_PUSHBACK_DT_FIRST_TICK : PHYS_DT,
        carIdx
    };

    SuspContactSummary contacts{};
    CarSuspension& susp = space->susp;

    for (int wheel = 0; wheel < NUM_WHEELS; wheel++)
    {
        const SuspensionWheelContribution contribution =
            computeSuspWheel(body, susp, frame, wheel);

        body.vel = body.vel + contribution.directVel;
        body.ang = body.ang + contribution.directAng;
        body.deltaVel = body.deltaVel + contribution.deltaVel;
        contacts.normalSum = contacts.normalSum + contribution.contactNorm;
        contacts.count += contribution.hasContact;
    }

    applySuspAggregate(body, susp, space->ctMan, frame, contacts);

    const Vec3 jumpImpulse = Vec3::ldg(space->susp.jumpImpulse[carIdx]);
    body.extVel = body.extVel + jumpImpulse;
    space->susp.jumpImpulse[carIdx] = Vec3::zero();
}
