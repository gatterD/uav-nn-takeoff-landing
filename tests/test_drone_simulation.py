import unittest

from drone_simulation import DroneEnv


class DroneSimulationTests(unittest.TestCase):
    def test_simple_takeoff_and_landing_state(self):
        env = DroneEnv(gui=False)

        initial = env.get_state()
        self.assertGreaterEqual(initial["z"], 0.0)
        self.assertEqual(initial["vx"], 0.0)
        self.assertEqual(initial["vy"], 0.0)
        self.assertEqual(initial["vz"], 0.0)

        takeoff_action = {
            "throttle": 0.8,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw_rate": 0.0,
        }
        after_takeoff = env.step(takeoff_action)
        self.assertGreater(after_takeoff["z"], initial["z"])

        landing_action = {
            "throttle": 0.2,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw_rate": 0.0,
        }
        after_landing = env.step(landing_action)
        self.assertLessEqual(after_landing["z"], after_takeoff["z"] + 1e-4)

        env.close()

    def test_physical_state_includes_aerodynamic_parameters(self):
        env = DroneEnv(gui=False)
        state = env.get_state()

        expected_fields = [
            "mass",
            "air_speed",
            "angle_of_attack",
            "sideslip_angle",
            "flight_path_angle",
            "path_angle",
            "roll_angle",
            "thrust",
            "aero_force_x",
            "aero_force_y",
            "aero_force_z",
            "inertia_x",
            "inertia_y",
            "inertia_z",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "aero_moment_x",
            "aero_moment_y",
            "aero_moment_z",
            "air_density",
            "wing_area",
            "wind_x",
            "wind_y",
            "wind_z",
            "turbulence",
            "wind_shear",
            "ground_effect",
        ]

        for field in expected_fields:
            self.assertIn(field, state)

        self.assertGreaterEqual(state["air_density"], 0.0)
        self.assertGreater(state["wing_area"], 0.0)
        self.assertGreaterEqual(state["mass"], 0.0)

        env.close()


if __name__ == "__main__":
    unittest.main()
