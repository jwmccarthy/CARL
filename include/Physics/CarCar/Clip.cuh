#pragma once

#include "Physics/Collision/Clip.cuh"
#include "SAT.cuh"

struct FaceRef
{
    const CarBox* ref = nullptr;
    const CarBox* inc = nullptr;
    int axis = 0;

    CARL_D CARL_FI bool valid() const
    {
        return ref != nullptr;
    }
};

struct IncidentFace
{
    Vec3 normal;
    int axis;
};

CARL_D CARL_FI FaceRef selectFaceRef(
    const SATResult& sat,
    const CarBox& boxA,
    const CarBox& boxB)
{
    if (sat.axisIdx < 3) return { &boxA, &boxB, sat.axisIdx };
    if (sat.axisIdx < 6) return { &boxB, &boxA, sat.axisIdx - 3 };
    return {};
}

CARL_D CARL_FI IncidentFace selectIncidentFace(
    const CarBox& box,
    const Vec3& refNormal)
{
    IncidentFace face{};
    float bestAlignment = -1.f;

    #pragma unroll
    for (int axis = 0; axis < 3; axis++)
    {
        const Vec3 normal = box.axis(axis);
        const float dot = normal.dot(refNormal);
        const float alignment = fabsf(dot);
        if (alignment <= bestAlignment) continue;

        bestAlignment = alignment;
        face.axis = axis;
        face.normal = dot > 0.f ? normal.neg() : normal;
    }

    return face;
}

CARL_D CARL_FI void buildIncidentQuad(
    const CarBox& box,
    const Vec3& refNormal,
    Vec3* quad)
{
    const IncidentFace face = selectIncidentFace(box, refNormal);
    const int axisU = (face.axis + 1) % 3;
    const int axisV = (face.axis + 2) % 3;
    const Vec3 extentU = box.axis(axisU) * CAR_HALF_EX[axisU];
    const Vec3 extentV = box.axis(axisV) * CAR_HALF_EX[axisV];
    const Vec3 center = box.cen + face.normal * CAR_HALF_EX[face.axis];

    quad[0] = center - extentU - extentV;
    quad[1] = center + extentU - extentV;
    quad[2] = center + extentU + extentV;
    quad[3] = center - extentU + extentV;
}

CARL_D CARL_FI int clipAndSwap(
    Vec3*& input,
    Vec3*& output,
    int count,
    const Vec3& normal,
    float dist)
{
    count = clipPlane(input, count, output, normal, dist);

    Vec3* swap = input;
    input = output;
    output = swap;

    return count;
}

CARL_D CARL_FI int clipIncidentQuad(
    const FaceRef& face,
    Vec3*& input,
    Vec3*& output)
{
    int count = 4;

    for (int axis = 0; axis < 3 && count > 0; axis++)
    {
        if (axis == face.axis) continue;

        const Vec3 sideAxis = face.ref->axis(axis);
        const float extent = CAR_HALF_EX[axis];
        const float maxDist = face.ref->cen.dot(sideAxis) + extent;
        const float minDist = face.ref->cen.dot(sideAxis.neg()) + extent;

        count = clipAndSwap(input, output, count, sideAxis, maxDist);
        if (count == 0) break;

        count = clipAndSwap(input, output, count, sideAxis.neg(), minDist);
    }

    return count;
}

CARL_D CARL_FI int supportContact(
    const SATResult& sat,
    const CarBox& boxA,
    const CarBox& boxB,
    CarCarContact* contacts)
{
    const Vec3 pointA = boxA.support(sat.minAxis.neg());
    const Vec3 pointB = boxB.support(sat.minAxis);

    contacts[0] = {
        boxA.rot.toLocal(pointA - boxA.cen),
        boxB.rot.toLocal(pointB - boxB.cen),
        sat.minAxis,
        sat.minPen
    };

    return 1;
}

CARL_D CARL_FI int faceContacts(
    const SATResult& sat,
    const CarBox& boxA,
    const CarBox& boxB,
    const FaceRef& face,
    CarCarContact* contacts)
{
    Vec3 refNormal = face.ref->axis(face.axis);
    if (refNormal.dot(face.inc->cen - face.ref->cen) < 0.f)
    {
        refNormal = refNormal.neg();
    }

    Vec3 polyA[8];
    Vec3 polyB[8];
    buildIncidentQuad(*face.inc, refNormal, polyA);

    Vec3* input = polyA;
    Vec3* output = polyB;

    const int polyCount = clipIncidentQuad(face, input, output);
    const Vec3 refCenter = face.ref->cen + refNormal * CAR_HALF_EX[face.axis];

    int count = 0;
    
    for (int i = 0; i < polyCount && count < MAX_CAR_MANIFOLD_POINTS; i++)
    {
        const float depth = (refCenter - input[i]).dot(refNormal);
        if (depth < -CAR_CONTACT_BREAK) continue;

        contacts[count++] = {
            boxA.rot.toLocal(input[i] - boxA.cen),
            boxB.rot.toLocal(input[i] - boxB.cen),
            sat.minAxis,
            fmaxf(depth, 0.f)
        };
    }

    if (count > 0) return count;

    contacts[0] = {
        boxA.rot.toLocal(refCenter - boxA.cen),
        boxB.rot.toLocal(refCenter - boxB.cen),
        sat.minAxis,
        sat.minPen
    };
    return 1;
}

CARL_D CARL_FI int carCarContacts(
    const SATResult& sat,
    const CarBox& boxA,
    const CarBox& boxB,
    CarCarContact* contacts)
{
    const FaceRef face = selectFaceRef(sat, boxA, boxB);
    if (!face.valid()) return supportContact(sat, boxA, boxB, contacts);

    return faceContacts(sat, boxA, boxB, face, contacts);
}
