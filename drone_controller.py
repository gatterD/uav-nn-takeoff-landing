"""Контроллеры формируют action и не изменяют состояние DroneEnv напрямую."""

import numpy as np

from drone_physic import DroneParameters, mix_controls
from drone_simulation import DroneEnv


def altitude_controller(target_altitude, current_altitude, velocity_z, integral=0.0):
    """Простой PID-подобный регулятор высоты, возвращающий нормированную тягу."""
    error = float(target_altitude) - float(current_altitude)
    command = 0.41 + 0.22 * error - 0.10 * float(velocity_z) + 0.01 * float(integral)
    return float(np.clip(command, 0.0, 1.0))


def motor_action(total_thrust, roll=0.0, pitch=0.0, yaw=0.0, parameters=None):
    """Преобразует команды регулятора в четыре нормированные команды моторов."""
    if parameters is None:
        parameters = DroneParameters()
    motors = mix_controls(total_thrust, roll, pitch, yaw, parameters)
    return {
        "motors": motors,
        "motor_1": float(motors[0]), "motor_2": float(motors[1]),
        "motor_3": float(motors[2]), "motor_4": float(motors[3]),
        "total_thrust": float(total_thrust),
    }


def run_flight_simulation(gui=True, env_params=None, scenario_name="default"):
    """Генерирует воспроизводимый takeoff-hover-landing датасет."""
    env = DroneEnv(gui=gui)
    if env_params:
        env.set_environment(
            wind=env_params.get("wind"),
            mass=env_params.get("mass"),
            max_motor_thrust=env_params.get("max_motor_thrust"),
        )
    target_altitude = 5.0
    landing = False
    integral = 0.0

    try:
        for step in range(20000):
            state_before = env.get_state()
            if step > 10000:
                landing = True
                target_altitude = 0.0
            error = target_altitude - state_before["z"]
            integral = float(np.clip(integral + error * env.TIME_STEP, -5.0, 5.0))
            throttle = altitude_controller(target_altitude, state_before["z"], state_before["vz"], integral)
            action = motor_action(
                throttle * 4.0 * env.parameters.max_motor_thrust,
                parameters=env.parameters,
            )
            state = env.step(action)
            phase = "landing" if landing else ("hover" if abs(error) < 0.1 else "takeoff")
            yield {
                "scenario": scenario_name,
                "time": step * env.TIME_STEP,
                "phase": phase,
                "target_altitude": target_altitude,
                "altitude": state["z"],
                "altitude_error": target_altitude - state["z"],
                "vertical_velocity": state["vz"],
                "throttle": throttle,
                "x": state["x"], "y": state["y"], "z": state["z"],
                "vx": state["vx"], "vy": state["vy"], "vz": state["vz"],
                "roll": state["roll"], "pitch": state["pitch"], "yaw": state["yaw"],
                "wx": state["wx"], "wy": state["wy"], "wz": state["wz"],
                "motor_1": state["motor_1"], "motor_2": state["motor_2"],
                "motor_3": state["motor_3"], "motor_4": state["motor_4"],
                "total_thrust": state["total_thrust"], "thrust": state["thrust"],
                "target_x": 0.0, "target_y": 0.0,
                "target_z": target_altitude,
                "target_roll": 0.0, "target_pitch": 0.0, "target_yaw": 0.0,
                "mass": state["mass"], "wind_x": state["wind_x"],
                "max_motor_thrust": state["max_motor_thrust"],
                "wind_y": state["wind_y"], "wind_z": state["wind_z"],
                "drag_force_x": state["linear_drag_x"],
                "drag_force_y": state["linear_drag_y"],
                "drag_force_z": state["linear_drag_z"],
            }
            if (landing and state["z"] <= env.parameters.ground_clearance + 0.02
                    and abs(state["vz"]) < 0.15):
                break
    finally:
        env.close()
