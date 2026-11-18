import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from manual_takeoff import run_manual_flight, save_trajectory_plot


class ManualTakeoffTests(unittest.TestCase):
    def test_saves_3d_trajectory_plot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "trajectory.png"

            saved_path = save_trajectory_plot(
                [(0.0, 0.0, 0.1), (1.0, 0.5, 2.0)],
                output_path=output_path,
            )

            self.assertEqual(saved_path, output_path)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_collects_and_saves_coordinates_after_flight(self):
        class FakeEnvironment:
            TIME_STEP = 2.0
            parameters = object()

            def __init__(self):
                self.states = [
                    {"x": 1.0, "y": 2.0, "z": 5.0, "vz": 0.0},
                    {"x": 2.0, "y": 3.0, "z": 5.0, "vz": 0.0},
                    {"x": 3.0, "y": 4.0, "z": 0.04, "vz": 0.0},
                ]
                self.current_state = {
                    "x": 0.0, "y": 0.0, "z": 0.15, "vz": 0.0,
                }
                self.closed = False

            def set_wind(self, **_kwargs):
                pass

            def get_state(self):
                return self.current_state

            def step(self, _action):
                self.current_state = self.states.pop(0)
                return self.current_state

            def close(self):
                self.closed = True

        environment = FakeEnvironment()
        with (
            patch("manual_takeoff.DroneEnv", return_value=environment),
            patch("manual_takeoff.assess_landing", return_value={
                "landing_success": True,
            }),
            patch("manual_takeoff.save_trajectory_plot") as save_plot,
        ):
            run_manual_flight(gui=False)

        self.assertTrue(environment.closed)
        save_plot.assert_called_once_with([
            (1.0, 2.0, 5.0),
            (2.0, 3.0, 5.0),
            (3.0, 4.0, 0.04),
        ])

    def test_rejects_empty_trajectory(self):
        with self.assertRaisesRegex(ValueError, "empty flight trajectory"):
            save_trajectory_plot([])


if __name__ == "__main__":
    unittest.main()
