import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from drone_physic import (
    DroneParameters,
    DroneState,
    compute_forces_and_moments,
    ground_contact_height,
    integrate_state,
    mix_controls,
    motor_forces_and_moments,
    rotation_matrix,
)
from drone_simulation import DroneEnv
from drone_controller import run_flight_simulation
from flight_data_logger import (
    BASELINE_SCENARIO,
    generate_random_scenarios,
    validate_scenario_records,
)
from flight_log_validation import REQUIRED_NUMERIC_COLUMNS, validate_flight_log


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
        state = env.step({"throttle": 0.8})
        self.assertEqual(state["motor_1"], 0.8)
        self.assertEqual(state["motor_4"], 0.8)
        self.assertGreater(state["z"], 0.15)
        env.close()

    def test_attitude_commands_are_rejected_instead_of_ignored(self):
        env = DroneEnv(gui=False)
        try:
            with self.assertRaisesRegex(ValueError, "Attitude controls are not supported"):
                env.step({"throttle": 0.5, "roll": 0.0})
        finally:
            env.close()

    def test_seeded_scenarios_repeat_and_baseline_flight_validates(self):
        self.assertEqual(
            generate_random_scenarios(seed=42),
            generate_random_scenarios(seed=42),
        )

        records = [
            dict(record)
            for record in run_flight_simulation(
                gui=False,
                env_params=BASELINE_SCENARIO,
                scenario_name="baseline_test",
                target_altitude=1.0,
                hover_duration=0.1,
            )
        ]

        validate_scenario_records(
            records, target_altitude=1.0, hover_duration=0.1
        )
        self.assertEqual(records[-1]["phase"], "landing")
        self.assertEqual(records[-1]["episode_status"], "success")
        self.assertEqual(records[-1]["termination_reason"], "stable_landing")

        wrong_order = [dict(record) for record in records]
        first_hover = next(
            index for index, record in enumerate(wrong_order)
            if record["phase"] == "hover"
        )
        wrong_order[first_hover]["phase"] = "landing"
        with self.assertRaisesRegex(ValueError, "phases must follow"):
            validate_scenario_records(
                wrong_order, target_altitude=1.0, hover_duration=0.1
            )

        short_hover = [dict(record) for record in records]
        first_landing = next(
            index for index, record in enumerate(short_hover)
            if record["phase"] == "landing"
        )
        short_hover[first_landing - 1]["phase"] = "landing"
        with self.assertRaisesRegex(ValueError, "required duration"):
            validate_scenario_records(
                short_hover, target_altitude=1.0, hover_duration=0.1
            )

    def test_seeded_environment_and_difficulty_combinations_are_classified(self):
        scenarios = generate_random_scenarios(seed=42)
        self.assertEqual(len(scenarios), 9)

        max_steps = 10000
        for scenario_name, env_params in scenarios.items():
            with self.subTest(scenario=scenario_name):
                records = [
                    dict(record)
                    for record in run_flight_simulation(
                        gui=False,
                        env_params=env_params,
                        scenario_name=scenario_name,
                        target_altitude=1.0,
                        hover_duration=0.1,
                        max_steps=max_steps,
                    )
                ]

                self.assertGreater(len(records), 0)
                self.assertLessEqual(len(records), max_steps)
                validated = validate_flight_log(pd.DataFrame(records))
                self.assertTrue(
                    np.isfinite(
                        validated[list(REQUIRED_NUMERIC_COLUMNS)].to_numpy()
                    ).all()
                )
                self.assertGreaterEqual(validated["z"].max(), 0.9)
                self.assertIn("hover", validated["phase"].tolist())

                if records[-1]["episode_status"] == "success":
                    self.assertEqual(records[-1]["phase"], "landing")
                    self.assertEqual(
                        records[-1]["termination_reason"], "stable_landing"
                    )
                    validate_scenario_records(
                        records, target_altitude=1.0, hover_duration=0.1
                    )
                else:
                    self.assertIn(
                        records[-1]["episode_status"], ("failure", "timeout")
                    )
                    self.assertTrue(records[-1]["termination_reason"])
                    with self.assertRaisesRegex(
                        ValueError, "stable ground contact"
                    ):
                        validate_scenario_records(
                            records, target_altitude=1.0, hover_duration=0.1
                        )

    def test_flight_generator_marks_step_limit_as_timeout(self):
        records = [
            dict(record)
            for record in run_flight_simulation(gui=False, max_steps=1)
        ]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[-1]["episode_status"], "timeout")
        self.assertTrue(records[-1]["termination_reason"].startswith("max_steps_reached:"))

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

    def test_body_rates_update_orientation_as_composed_rotations(self):
        parameters = DroneParameters()
        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([0.3, -0.25, 0.4]),
            angular_velocity=np.array([0.8, -0.4, 0.6]),
        )

        next_state, _ = integrate_state(state, [0.5] * 4, parameters)
        expected = (
            Rotation.from_euler("xyz", state.orientation)
            * Rotation.from_rotvec(next_state.angular_velocity * parameters.time_step)
        )

        np.testing.assert_allclose(
            Rotation.from_euler("xyz", state.orientation).as_matrix(),
            rotation_matrix(*state.orientation),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", next_state.orientation).as_matrix(),
            expected.as_matrix(),
            atol=1e-12,
        )

    def test_control_mixer_matches_motor_moment_convention(self):
        parameters = DroneParameters()
        expected_moments = np.array([0.1, 0.12, 0.04])
        motors = mix_controls(12.0, *expected_moments, parameters)
        actual_moments = motor_forces_and_moments(motors, parameters)[2]

        np.testing.assert_allclose(actual_moments, expected_moments, atol=1e-12)

    def test_integrator_resolves_ground_contact_without_pybullet(self):
        parameters = DroneParameters()
        state = DroneState(
            position=np.array([0.0, 0.0, 0.01]),
            velocity=np.array([0.0, 0.0, -1.0]),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

        next_state, _ = integrate_state(state, [0.0] * 4, parameters)

        self.assertEqual(next_state.position[2], parameters.ground_clearance)
        self.assertEqual(next_state.velocity[2], 0.0)

    def test_ground_contact_supports_body_and_damps_landing_motion(self):
        parameters = DroneParameters()
        state = DroneState(
            position=np.array([0.0, 0.0, parameters.ground_clearance]),
            velocity=np.array([1.0, 0.0, 0.0]),
            orientation=np.array([0.2, 0.1, 0.0]),
            angular_velocity=np.array([0.5, -0.3, 1.0]),
        )

        next_state, _ = integrate_state(state, [0.0] * 4, parameters)

        self.assertAlmostEqual(
            next_state.position[2],
            ground_contact_height(next_state.orientation, parameters),
        )
        self.assertLess(next_state.velocity[0], state.velocity[0])
        self.assertLess(
            np.linalg.norm(next_state.angular_velocity),
            np.linalg.norm(state.angular_velocity),
        )

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
