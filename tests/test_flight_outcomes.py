import unittest

from drone_physic import DroneParameters, ground_contact_height
from flight_outcome import assess_landing


def landed_state(**overrides):
    state = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.04,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "wx": 0.0,
        "wy": 0.0,
        "wz": 0.0,
    }
    state.update(overrides)
    return state


class LandingOutcomeTests(unittest.TestCase):
    def test_flat_stable_contact_is_success(self):
        result = assess_landing(landed_state())

        self.assertTrue(result["touchdown"])
        self.assertTrue(result["landing_success"])
        self.assertEqual(result["landing_reason"], "stable_landing")

    def test_contact_and_success_are_distinct(self):
        result = assess_landing(landed_state(vx=0.2))

        self.assertTrue(result["touchdown"])
        self.assertFalse(result["landing_success"])
        self.assertEqual(result["landing_reason"], "horizontal_speed_exceeded")

    def test_vertical_and_angular_speeds_prevent_success(self):
        cases = (
            ({"vz": 0.2}, "vertical_speed_exceeded"),
            ({"wz": 0.2}, "angular_speed_exceeded"),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                result = assess_landing(landed_state(**overrides))

                self.assertTrue(result["touchdown"])
                self.assertFalse(result["landing_success"])
                self.assertEqual(result["landing_reason"], reason)

    def test_contact_height_accounts_for_tilt(self):
        parameters = DroneParameters()
        orientation = (0.2, -0.1, 0.0)
        height = ground_contact_height(orientation, parameters)
        result = assess_landing(landed_state(
            z=height,
            roll=orientation[0],
            pitch=orientation[1],
        ), parameters)

        self.assertTrue(result["touchdown"])
        self.assertFalse(result["landing_success"])
        self.assertEqual(result["landing_reason"], "tilt_exceeded")

    def test_contact_is_false_above_the_oriented_ground_height(self):
        result = assess_landing(landed_state(z=0.2))

        self.assertFalse(result["touchdown"])
        self.assertFalse(result["landing_success"])
        self.assertEqual(result["landing_reason"], "no_ground_contact")


if __name__ == "__main__":
    unittest.main()