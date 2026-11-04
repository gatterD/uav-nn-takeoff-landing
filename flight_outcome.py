"""Shared touchdown and stable-landing criteria for simulation and analysis."""

from dataclasses import dataclass
import math

import numpy as np

from drone_physic import DroneParameters, ground_contact_height


@dataclass(frozen=True)
class LandingCriteria:
    contact_tolerance: float = 0.02
    max_vertical_speed: float = 0.15
    max_horizontal_speed: float = 0.1
    max_angular_speed: float = 0.1
    max_tilt: float = math.radians(10.0)


DEFAULT_LANDING_CRITERIA = LandingCriteria()


def assess_landing(state, parameters=None, criteria=DEFAULT_LANDING_CRITERIA):
    """Return contact and stable-landing status using shared physical thresholds."""
    parameters = parameters or DroneParameters()
    roll = float(state.get("roll", math.nan))
    pitch = float(state.get("pitch", math.nan))
    yaw = float(state.get("yaw", math.nan))
    z = float(state.get("z", math.nan))
    vx = float(state.get("vx", math.nan))
    vy = float(state.get("vy", math.nan))
    vz = float(state.get("vz", math.nan))
    wx = float(state.get("wx", math.nan))
    wy = float(state.get("wy", math.nan))
    wz = float(state.get("wz", math.nan))
    values = (roll, pitch, yaw, z, vx, vy, vz, wx, wy, wz)

    if not all(math.isfinite(value) for value in values):
        return {
            "ground_contact_height": math.nan,
            "touchdown": False,
            "landing_success": False,
            "landing_reason": "non_finite_state",
        }

    stored_height = state.get("ground_contact_height")
    if stored_height is None or not math.isfinite(float(stored_height)):
        contact_height = ground_contact_height((roll, pitch, yaw), parameters)
    else:
        contact_height = float(stored_height)

    touchdown = abs(z - contact_height) <= criteria.contact_tolerance
    horizontal_speed = math.hypot(vx, vy)
    angular_speed = math.sqrt(wx * wx + wy * wy + wz * wz)
    tilt = math.hypot(roll, pitch)

    if not touchdown:
        reason = "no_ground_contact"
    elif abs(vz) > criteria.max_vertical_speed:
        reason = "vertical_speed_exceeded"
    elif horizontal_speed > criteria.max_horizontal_speed:
        reason = "horizontal_speed_exceeded"
    elif angular_speed > criteria.max_angular_speed:
        reason = "angular_speed_exceeded"
    elif tilt > criteria.max_tilt:
        reason = "tilt_exceeded"
    else:
        reason = "stable_landing"

    return {
        "ground_contact_height": contact_height,
        "touchdown": touchdown,
        "landing_success": reason == "stable_landing",
        "landing_reason": reason,
        "horizontal_speed": horizontal_speed,
        "angular_speed": angular_speed,
        "tilt": tilt,
    }