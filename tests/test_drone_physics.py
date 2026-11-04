import unittest

import numpy as np

from drone_physic import (
    DroneParameters,
    DroneState,
    compute_forces_and_moments,
    ground_contact_height,
    integrate_state,
    mix_controls,
    motor_forces_and_moments,
    motor_positions,
    motor_thrusts,
)


def make_state(z=1.0, orientation=None, velocity=None, angular_velocity=None):
    return DroneState(
        position=np.array([0.0, 0.0, z]),
        velocity=np.zeros(3) if velocity is None else np.array(velocity, dtype=float),
        orientation=np.zeros(3) if orientation is None else np.array(orientation, dtype=float),
        angular_velocity=(
            np.zeros(3)
            if angular_velocity is None
            else np.array(angular_velocity, dtype=float)
        ),
    )


class DronePhysicsTests(unittest.TestCase):
    def test_motor_positions_are_diagonal_with_arm_length_as_radius(self):
        parameters = DroneParameters(arm_length=0.3)
        positions = motor_positions(parameters)

        np.testing.assert_allclose(np.linalg.norm(positions, axis=1), 0.3)
        np.testing.assert_allclose(positions[:, 2], 0.0)
        np.testing.assert_allclose(
            positions[:, :2],
            0.3 / np.sqrt(2.0) * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]),
        )

    def test_free_fall_acceleration_is_gravity(self):
        parameters = DroneParameters()
        forces = compute_forces_and_moments(make_state(), [0.0] * 4, parameters)

        self.assertAlmostEqual(forces["force"][2] / parameters.mass, -parameters.gravity)

    def test_hover_thrust_cancels_gravity_for_multiple_coefficients(self):
        for coefficient in (1.0, 1.6):
            with self.subTest(coefficient=coefficient):
                parameters = DroneParameters(motor_thrust_coefficient=coefficient)
                command = (
                    parameters.mass * parameters.gravity
                    / (4.0 * parameters.max_motor_thrust * coefficient)
                )
                forces = compute_forces_and_moments(
                    make_state(), [command] * 4, parameters
                )

                self.assertAlmostEqual(forces["force"][2] / parameters.mass, 0.0)

    def test_symmetric_motors_create_no_body_moment(self):
        for coefficient in (1.0, 1.8):
            parameters = DroneParameters(motor_thrust_coefficient=coefficient)
            np.testing.assert_allclose(
                motor_forces_and_moments([0.4] * 4, parameters)[2],
                np.zeros(3),
                atol=1e-12,
            )

    def test_single_motor_moments_follow_position_and_spin_signs(self):
        parameters = DroneParameters()
        expected_signs = np.array([
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ])

        for motor_index, signs in enumerate(expected_signs):
            motors = np.zeros(4)
            motors[motor_index] = 1.0
            moments = motor_forces_and_moments(motors, parameters)[2]
            np.testing.assert_array_equal(np.sign(moments), signs)

    def test_mixer_recovers_reachable_wrench_for_thrust_coefficients(self):
        requested_thrust = 18.0
        requested_moments = np.array([0.08, -0.06, 0.02])
        for coefficient in (1.0, 1.7):
            with self.subTest(coefficient=coefficient):
                parameters = DroneParameters(motor_thrust_coefficient=coefficient)
                motors = mix_controls(requested_thrust, *requested_moments, parameters)
                _, actual_thrust, actual_moments = motor_forces_and_moments(
                    motors, parameters
                )

                self.assertTrue(np.all((motors >= 0.0) & (motors <= 1.0)))
                self.assertAlmostEqual(actual_thrust, requested_thrust)
                np.testing.assert_allclose(actual_moments, requested_moments, atol=1e-12)

    def test_mixer_saturation_stays_bounded(self):
        parameters = DroneParameters(motor_thrust_coefficient=1.5)
        motors = mix_controls(1e6, 1e5, -1e5, 1e5, parameters)
        thrusts = motor_thrusts(motors, parameters)

        self.assertTrue(np.all(np.isfinite(motors)))
        self.assertTrue(np.all((motors >= 0.0) & (motors <= 1.0)))
        self.assertLessEqual(
            thrusts.sum(),
            4.0 * parameters.max_motor_thrust * parameters.motor_thrust_coefficient,
        )

    def test_ground_contact_supports_body_without_penetration(self):
        parameters = DroneParameters()
        orientation = np.array([0.2, -0.1, 0.0])
        height = ground_contact_height(orientation, parameters)
        state = make_state(
            z=height,
            orientation=orientation,
            velocity=[0.0, 0.0, -0.5],
        )

        next_state, _ = integrate_state(state, [0.0] * 4, parameters)

        self.assertGreaterEqual(
            next_state.position[2],
            ground_contact_height(next_state.orientation, parameters),
        )

    def test_identical_conditions_produce_identical_trajectories(self):
        parameters = DroneParameters()
        trajectories = []
        for _ in range(2):
            state = make_state(z=2.0, angular_velocity=[0.1, -0.05, 0.02])
            rows = []
            for _ in range(300):
                state, _ = integrate_state(state, [0.5, 0.48, 0.52, 0.5], parameters)
                rows.append(np.concatenate((state.position, state.velocity, state.orientation)))
            trajectories.append(np.asarray(rows))

        np.testing.assert_array_equal(trajectories[0], trajectories[1])

    def test_long_run_has_no_non_finite_state(self):
        parameters = DroneParameters()
        state = make_state(z=1.0)
        for _ in range(5000):
            state, _ = integrate_state(state, [0.35, 0.36, 0.34, 0.35], parameters)
            values = np.concatenate((state.position, state.velocity, state.orientation, state.angular_velocity))
            self.assertTrue(np.isfinite(values).all())

    def test_reset_restores_initial_state_and_motor_commands(self):
        from drone_simulation import DroneEnv

        env = DroneEnv(gui=False)
        try:
            env.step({"motors": [1.0, 0.0, 0.0, 1.0]})
            reset_state = env.reset()

            np.testing.assert_array_equal(env.state.position, [0.0, 0.0, 0.15])
            np.testing.assert_array_equal(env.state.velocity, np.zeros(3))
            np.testing.assert_array_equal(env.state.orientation, np.zeros(3))
            np.testing.assert_array_equal(env.motor_commands, np.zeros(4))
            self.assertEqual(reset_state["z"], 0.15)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()