#pragma once

#include "./Common.cuh"

#define CARL_VECTOR_OP(type, op) \
    constexpr CARL_HD CARL_FI type operator op(const type& v) const \
    { return { x op v.x, y op v.y, z op v.z }; }

#define CARL_SCALAR_OP(type, scalar, op) \
    constexpr CARL_HD CARL_FI type operator op(scalar s) const \
    { return { x op s, y op s, z op s }; }

#define CARL_CHAIN_VECTOR_OPS(type) \
    CARL_HD CARL_FI type add(const type& v) const { return *this + v; } \
    CARL_HD CARL_FI type sub(const type& v) const { return *this - v; } \
    CARL_HD CARL_FI type mul(const type& v) const { return *this * v; } \
    CARL_HD CARL_FI type div(const type& v) const { return *this / v; }

#define CARL_CHAIN_SCALAR_OPS(type, scalar) \
    CARL_HD CARL_FI type add(scalar s) const { return *this + s; } \
    CARL_HD CARL_FI type sub(scalar s) const { return *this - s; } \
    CARL_HD CARL_FI type mul(scalar s) const { return *this * s; } \
    CARL_HD CARL_FI type div(scalar s) const { return *this / s; }

struct Int3;

CARL_HD CARL_FI float clampf(float value, float minValue, float maxValue)
{
    return fmaxf(minValue, fminf(maxValue, value));
}

struct __align__(16) Vec3
{
    float x, y, z, _w;
    
    static CARL_D CARL_FI Vec3 ldg(const Vec3& v)
    {
        return v;
    }

    static constexpr CARL_HD CARL_FI Vec3 zero()
    {
        return { 0.f, 0.f, 0.f };
    }

    static constexpr CARL_HD CARL_FI Vec3 fill(const float s)
    {
        return { s, s, s, 0.f };
    }

    CARL_VECTOR_OP(Vec3, +)
    CARL_VECTOR_OP(Vec3, -)
    CARL_VECTOR_OP(Vec3, *)
    CARL_VECTOR_OP(Vec3, /)

    CARL_SCALAR_OP(Vec3, float, +)
    CARL_SCALAR_OP(Vec3, float, -)
    CARL_SCALAR_OP(Vec3, float, *)
    CARL_SCALAR_OP(Vec3, float, /)

    CARL_CHAIN_VECTOR_OPS(Vec3)
    CARL_CHAIN_SCALAR_OPS(Vec3, float)

    CARL_HD CARL_FI float& operator[](int i)       { return (&x)[i]; }
    CARL_HD CARL_FI float  operator[](int i) const { return (&x)[i]; }

    CARL_HD CARL_FI Vec3 neg() const
    {
        return { -x, -y, -z };
    }

    CARL_HD CARL_FI Vec3 abs() const
    {
        return { fabsf(x), fabsf(y), fabsf(z) };
    }

    CARL_HD CARL_FI Vec3 sign() const
    {
        return {
            x >= 0.f ? 1.f : -1.f,
            y >= 0.f ? 1.f : -1.f,
            z >= 0.f ? 1.f : -1.f
        };
    }

    CARL_HD CARL_FI int argMax() const
    {
        return y > x ? (y > z ? 1 : 2) : (x > z ? 0 : 2);
    }

    CARL_HD CARL_FI float prod() const
    {
        return x * y * z;
    }

    CARL_HD CARL_FI Vec3 cross(const Vec3& v) const
    {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }

    CARL_HD CARL_FI float dot(const Vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    CARL_HD CARL_FI float lenSq() const
    {
        return dot(*this);
    }

    CARL_HD CARL_FI float len() const
    {
        return sqrtf(lenSq());
    }

    CARL_HD CARL_FI Vec3 norm() const
    {
        return *this / len();
    }

    CARL_HD CARL_FI bool gte(float s) const
    {
        return x >= s && y >= s && z >= s;
    }

    CARL_HD CARL_FI Vec3 max(const Vec3& v) const
    {
        return { fmaxf(x, v.x), fmaxf(y, v.y), fmaxf(z, v.z) };
    }

    CARL_HD CARL_FI Vec3 max(float s) const
    {
        return { fmaxf(x, s), fmaxf(y, s), fmaxf(z, s) };
    }

    CARL_HD CARL_FI Vec3 min(const Vec3& v) const
    {
        return { fminf(x, v.x), fminf(y, v.y), fminf(z, v.z) };
    }

    CARL_HD CARL_FI Vec3 min(float s) const
    {
        return { fminf(x, s), fminf(y, s), fminf(z, s) };
    }

    CARL_HD CARL_FI Vec3 clamp(const Vec3& lo, const Vec3& hi) const
    {
        return max(lo).min(hi);
    }

    CARL_HD CARL_FI Vec3 clamp(float lo, float hi) const
    {
        return max(lo).min(hi);
    }

    CARL_HD CARL_FI Int3 toInt3() const;
};

struct __align__(16) Int3
{
    int x, y, z, _w;

    static CARL_D CARL_FI Int3 ldg(const Int3& i)
    {
        int4 tmp = __ldg(reinterpret_cast<const int4*>(&i));
        return { tmp.x, tmp.y, tmp.z, tmp.w };
    }

    CARL_VECTOR_OP(Int3, +)
    CARL_VECTOR_OP(Int3, -)
    CARL_VECTOR_OP(Int3, *)
    CARL_VECTOR_OP(Int3, /)

    CARL_SCALAR_OP(Int3, int, +)
    CARL_SCALAR_OP(Int3, int, -)
    CARL_SCALAR_OP(Int3, int, *)
    CARL_SCALAR_OP(Int3, int, /)

    CARL_CHAIN_VECTOR_OPS(Int3)
    CARL_CHAIN_SCALAR_OPS(Int3, int)

    CARL_HD CARL_FI int& operator[](int i)
    {
        return (&x)[i];
    }

    CARL_HD CARL_FI int operator[](int i) const
    {
        return (&x)[i];
    }

    CARL_HD CARL_FI Int3 max(const Int3& v) const
    {
        return { x > v.x ? x : v.x, y > v.y ? y : v.y, z > v.z ? z : v.z };
    }

    CARL_HD CARL_FI Int3 max(int s) const
    {
        return { x > s ? x : s, y > s ? y : s, z > s ? z : s };
    }

    CARL_HD CARL_FI Int3 min(const Int3& v) const
    {
        return { x < v.x ? x : v.x, y < v.y ? y : v.y, z < v.z ? z : v.z };
    }

    CARL_HD CARL_FI Int3 min(int s) const
    {
        return { x < s ? x : s, y < s ? y : s, z < s ? z : s };
    }

    CARL_HD CARL_FI Vec3 toVec3() const
    {
        return { (float)x, (float)y, (float)z };
    }
};


CARL_HD CARL_FI Int3 Vec3::toInt3() const
{
    return { (int)x, (int)y, (int)z };
}

struct __align__(16) Quat
{
    float x, y, z, w;

    static CARL_D CARL_FI Quat ldg(const Quat& i)
    {
        return i;
    }

    static constexpr CARL_HD CARL_FI Quat angle(float a)
    {
        return { 0.f, 0.f, sinf(a / 2), cosf(a / 2) };
    }

    static CARL_HD CARL_FI Quat fromAxes(const Vec3 &f, const Vec3 &r, const Vec3 &u)
    {
        const float m[3][3] = {{f.x, r.x, u.x}, {f.y, r.y, u.y}, {f.z, r.z, u.z}};
        const float trace = m[0][0] + m[1][1] + m[2][2];
        float q[4];
        if (trace > 0.f)
        {
            const float x = trace + 1.f, s = .5f / sqrtf(x);
            q[3] = x * s;
            q[0] = (m[2][1] - m[1][2]) * s;
            q[1] = (m[0][2] - m[2][0]) * s;
            q[2] = (m[1][0] - m[0][1]) * s;
        }
        else
        {
            const int i =
                m[0][0] < m[1][1] ? (m[1][1] < m[2][2] ? 2 : 1) : (m[0][0] < m[2][2] ? 2 : 0);
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            const float x = m[i][i] - m[j][j] - m[k][k] + 1.f, s = .5f / sqrtf(x);
            q[i] = x * s;
            q[3] = (m[k][j] - m[j][k]) * s;
            q[j] = (m[j][i] + m[i][j]) * s;
            q[k] = (m[k][i] + m[i][k]) * s;
        }
        return {q[0], q[1], q[2], q[3]};
    }

    CARL_SCALAR_OP(Quat, float, +)
    CARL_SCALAR_OP(Quat, float, -)
    CARL_SCALAR_OP(Quat, float, *)
    CARL_SCALAR_OP(Quat, float, /)

    CARL_HD CARL_FI Vec3 toVec3() const
    {
        return { x, y, z, 0 };
    }

    CARL_HD CARL_FI float dot(const Quat& q) const
    {
        return (x * q.x + z * q.z) + (y * q.y + w * q.w);
    }

    CARL_HD CARL_FI float lenSq() const
    {
        return dot(*this);
    }

    CARL_HD CARL_FI float len() const
    {
        return sqrtf(lenSq());
    }

    CARL_HD CARL_FI Quat normalize() const
    {
        const float inv = 1.f / len();
        return { x * inv, y * inv, z * inv, w * inv };
    }

    CARL_HD CARL_FI Quat norm() const
    {
        return normalize();
    }

    CARL_HD CARL_FI Quat conj() const
    {
        return { -x, -y, -z, w };
    }

    CARL_HD CARL_FI Quat comp(const Quat& q) const
    {
        return {
            (w * q.x - z * q.y) + (x * q.w + y * q.z), (w * q.y - x * q.z) + (y * q.w + z * q.x),
            (w * q.z - y * q.x) + (z * q.w + x * q.y), (w * q.w - z * q.z) - (x * q.x + y * q.y)};
    }

    CARL_HD CARL_FI Vec3 toWorld(const Vec3& v) const
    {
        const float s = 2.f / lenSq();
        const float wx = w * x, wy = w * y, wz = w * z;
        const float xx = x * x, xy = x * y, xz = x * z, yy = y * y, yz = y * z, zz = z * z;
        return {(1.f - (yy + zz) * s) * v.x + ((xy - wz) * s) * v.y + ((xz + wy) * s) * v.z,
                ((xy + wz) * s) * v.x + (1.f - (xx + zz) * s) * v.y + ((yz - wx) * s) * v.z,
                ((xz - wy) * s) * v.x + ((yz + wx) * s) * v.y + (1.f - (xx + yy) * s) * v.z};
    }

    CARL_HD CARL_FI Vec3 toLocal(const Vec3& v) const
    {
        return conj().toWorld(v);
    }
};
