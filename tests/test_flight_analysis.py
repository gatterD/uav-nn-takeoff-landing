import unittest

import pandas as pd

from data_analysis import compute_metrics
from flight_analysis import analyze_flight_metrics, analyze_altitude_tracking, summarize_run
from flight_log_validation import REQUIRED_NUMERIC_COLUMNS


def make_log():
    row_count = 3
    data = {column: [0.0] * row_count for column in REQUIRED_NUMERIC_COLUMNS}
    data.update({
        "time": [0.0, 1.0, 2.0],
        "phase": ["takeoff", "landing", "landing"],
        "target_altitude": [1.0, 0.0, 0.0],
        "altitude": [0.0, 0.04, 0.04],
        "altitude_error": [1.0, 99.0, 99.0],
        "z": [0.04, 0.04, 0.04],
        "x": [0.0, 1.0, 0.0],
        "episode_status": ["in_progress", "in_progress", "success"],
        "termination_reason": ["", "", "stable_landing"],
    })
    return pd.DataFrame(data)


class FlightAnalysisTests(unittest.TestCase):
    def test_path_length_is_not_endpoint_displacement(self):
        metrics = compute_metrics(make_log())

        self.assertAlmostEqual(metrics["path_length"], 2.0)
        self.assertAlmostEqual(metrics["displacement"], 0.0)
        self.assertNotIn("distance_3d", metrics)

    def test_tracking_error_uses_each_rows_target(self):
        dataframe = make_log()

        tracking = analyze_altitude_tracking(dataframe)
        metrics = compute_metrics(dataframe)

        self.assertAlmostEqual(tracking["mae"], 0.36)
        self.assertAlmostEqual(metrics["mean_altitude_error"], 0.36)

    def test_throttle_integral_is_not_reported_as_energy(self):
        dataframe = make_log()
        dataframe["throttle"] = [0.0, 1.0, 0.0]

        metrics = compute_metrics(dataframe)

        self.assertAlmostEqual(metrics["throttle_integral"], 1.0)
        self.assertNotIn("energy_proxy", metrics)

    def test_flight_analysis_uses_shared_success_and_episode_status(self):
        dataframe = make_log()

        metrics = analyze_flight_metrics(dataframe)
        summary = summarize_run(dataframe, "test")

        self.assertTrue(metrics["touchdown"])
        self.assertTrue(metrics["landing_success"])
        self.assertEqual(metrics["episode_status"], "success")
        self.assertEqual(summary["flight_metrics"]["landing_reason"], "stable_landing")

    def test_horizontal_errors_are_measured_from_each_target_position(self):
        dataframe = make_log()
        dataframe["x"] = [2.0, 3.0, 4.0]
        dataframe["y"] = [0.0, 0.0, 0.0]
        dataframe["target_x"] = [1.0, 1.0, 2.0]
        dataframe["target_y"] = [0.0, 1.0, 0.0]

        metrics = analyze_flight_metrics(dataframe)

        self.assertAlmostEqual(metrics["landing_position_error"], 2.0)
        self.assertAlmostEqual(metrics["trajectory_deviation"], 5.0 ** 0.5)

    def test_touchdown_is_not_success_when_horizontal_speed_is_high(self):
        dataframe = make_log()
        dataframe.loc[2, "vx"] = 0.2

        metrics = analyze_flight_metrics(dataframe)

        self.assertTrue(metrics["touchdown"])
        self.assertFalse(metrics["landing_success"])
        self.assertEqual(metrics["landing_reason"], "horizontal_speed_exceeded")

    def test_both_analysis_entry_points_reject_nonmonotonic_time(self):
        dataframe = make_log()
        dataframe["time"] = [0.0, 2.0, 1.0]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            compute_metrics(dataframe)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            summarize_run(dataframe, "invalid")

    def test_short_logs_are_rejected_before_metrics(self):
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            compute_metrics(make_log().iloc[:1])
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            analyze_flight_metrics(make_log().iloc[:1])


if __name__ == "__main__":
    unittest.main()
