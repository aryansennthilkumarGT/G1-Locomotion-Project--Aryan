"""Mathematical checks; these do not demonstrate simulator or actor equivalence."""

import math
import unittest

from heading_geometry import relative_heading, rotate_xy, yaw_xyzw


class HeadingGeometryTests(unittest.TestCase):
    def test_yaw_and_sign_invariance(self):
        for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            q = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
            self.assertAlmostEqual(relative_heading(yaw_xyzw(q), yaw), 0.0)
            self.assertAlmostEqual(relative_heading(yaw_xyzw([-x for x in q]), yaw), 0.0)

    def test_rotation_equivariance_and_relative_heading(self):
        for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            body = (0.4, 0.1)
            world = rotate_xy(body, yaw)
            back = rotate_xy(world, -yaw)
            self.assertAlmostEqual(back[0], body[0])
            self.assertAlmostEqual(back[1], body[1])
            self.assertAlmostEqual(relative_heading(yaw + 0.3, yaw), 0.3)

    def test_invalid_quaternion_fails(self):
        with self.assertRaises(ValueError):
            yaw_xyzw([0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            yaw_xyzw([0.0, 0.0, float("nan"), 1.0])


if __name__ == "__main__":
    unittest.main()
