import unittest

import numpy as np

from drone_physic import DroneParameters, DroneState, compute_forces_and_moments
from drone_simulation import DroneEnv


class DroneSimulationTests(unittest.TestCase):
    def test_motor_action_changes_attitude_through_moment(self):
        env = DroneEnv(gui=False)
        initial = env.get_state()
        state = env.step({"motors": [1.0, 0.0, 0.0, 0.0]})
        self.assertGreater(abs(state["wx"]), 0.0)
        self.assertNotEqual(state["roll"], initial["roll"])
        env.close()

    def test_legacy_throttle_maps_to_four_equal_motors(self):
        env = DroneEnv(gui=False)
        state = env.step({"throttle": 0.8, "roll": 0.0, "pitch": 0.0, "yaw_rate": 0.0})
        self.assertEqual(state["motor_1"], 0.8)
        self.assertEqual(state["motor_4"], 0.8)
        self.assertGreater(state["z"], 0.15)
        env.close()

    def test_rotated_body_thrust_has_horizontal_component(self):
        parameters = DroneParameters()
        env = DroneEnv(gui=False)
        state = env.state
        state.orientation = np.array([0.2, 0.0, 0.0])
        forces = compute_forces_and_moments(state, [0.5] * 4, parameters)
        self.assertGreater(abs(forces["thrust_force"][1]), 1e-6)
        env.close()

    def test_angular_damping_opposes_angular_velocity(self):
        parameters = DroneParameters()
        env = DroneEnv(gui=False)
        env.state.angular_velocity = np.array([1.0, -2.0, 0.5])
        forces = compute_forces_and_moments(env.state, [0.0] * 4, parameters)
        self.assertTrue(np.all(forces["damping_moment"] * env.state.angular_velocity <= 0.0))
        env.close()

    def test_state_contains_learning_and_environment_fields(self):
        env = DroneEnv(gui=False)
        state = env.get_state()
        expected_fields = [
            "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "wx", "wy", "wz",
            "motor_1", "motor_2", "motor_3", "motor_4", "total_thrust", "mass",
            "wind_x", "wind_y", "wind_z",
        ]
        for field in expected_fields:
            self.assertIn(field, state)
        self.assertNotIn("ground_effect", state)
        self.assertNotIn("wing_area", state)
        env.close()


if __name__ == "__main__":
    unittest.main()
