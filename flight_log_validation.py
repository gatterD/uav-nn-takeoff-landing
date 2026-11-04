"""Validation shared by the flight-log analysis entry points."""

import numpy as np
import pandas as pd


REQUIRED_NUMERIC_COLUMNS = (
    "time",
    "target_altitude",
    "altitude",
    "altitude_error",
    "vertical_velocity",
    "throttle",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "wx",
    "wy",
    "wz",
    "motor_1",
    "motor_2",
    "motor_3",
    "motor_4",
)
REQUIRED_COLUMNS = ("phase", *REQUIRED_NUMERIC_COLUMNS)


def validate_flight_log(dataframe, minimum_rows=2):
    """Return a numeric copy or raise a clear error for an invalid trajectory."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"Flight log is missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("Flight log contains no samples")
    if len(dataframe) < minimum_rows:
        raise ValueError(f"Flight log requires at least {minimum_rows} samples")
    if dataframe["phase"].isna().any() or dataframe["phase"].astype(str).str.strip().eq("").any():
        raise ValueError("Flight log column 'phase' contains missing values")

    validated = dataframe.copy()
    for column in REQUIRED_NUMERIC_COLUMNS:
        values = pd.to_numeric(validated[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Flight log column '{column}' contains missing or non-finite values")
        validated[column] = values

    if (validated["time"].diff().dropna() <= 0.0).any():
        raise ValueError("Flight log time must be strictly increasing")

    return validated