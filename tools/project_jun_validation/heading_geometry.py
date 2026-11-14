"""Planar equivariance primitives for xyzw root quaternions, independent of Isaac Lab."""

import math


def yaw_xyzw(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError("invalid quaternion")
    x, y, z, w = q
    norm = sum(value * value for value in q)
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("quaternion must be normalized")
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotate_xy(v, angle):
    if len(v) != 2 or not all(math.isfinite(x) for x in (*v, angle)):
        raise ValueError("invalid vector or heading")
    c, s = math.cos(angle), math.sin(angle)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def relative_heading(target_yaw, robot_yaw):
    if not math.isfinite(target_yaw) or not math.isfinite(robot_yaw):
        raise ValueError("invalid heading")
    return math.atan2(math.sin(target_yaw - robot_yaw), math.cos(target_yaw - robot_yaw))
