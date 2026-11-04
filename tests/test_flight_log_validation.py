import unittest

import pandas as pd

from flight_log_validation import REQUIRED_NUMERIC_COLUMNS, validate_flight_log


def valid_log(rows=3):
    values = {column: [0.0] * rows for column in REQUIRED_NUMERIC_COLUMNS}
    values.update({
        "time": [float(index) for index in range(rows)],
        "phase": ["takeoff"] * rows,
    })
    return pd.DataFrame(values)


class FlightLogValidationTests(unittest.TestCase):
    def test_accepts_valid_log_and_changing_target_altitude(self):
        dataframe = valid_log()
        dataframe["target_altitude"] = [2.0, 2.0, 0.0]

        result = validate_flight_log(dataframe)

        self.assertEqual(result["target_altitude"].tolist(), [2.0, 2.0, 0.0])

    def test_rejects_empty_and_short_logs(self):
        with self.assertRaisesRegex(ValueError, "no samples"):
            validate_flight_log(valid_log(0))
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            validate_flight_log(valid_log(1))

    def test_rejects_missing_required_column(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_flight_log(valid_log().drop(columns=["wx"]))

    def test_rejects_nan_and_infinity_in_physical_columns(self):
        for invalid in (float("nan"), float("inf")):
            dataframe = valid_log()
            dataframe.loc[1, "vx"] = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "vx"):
                validate_flight_log(dataframe)

    def test_rejects_repeated_or_decreasing_time(self):
        for times in ([0.0, 0.0, 1.0], [0.0, -0.5, 1.0]):
            dataframe = valid_log()
            dataframe["time"] = times
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "strictly increasing"):
                validate_flight_log(dataframe)


if __name__ == "__main__":
    unittest.main()