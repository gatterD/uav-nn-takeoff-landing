"""Упрощенная воспроизводимая 6-DOF модель квадрокоптера."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DroneParameters:
    """Централизованные параметры объекта и численного интегрирования."""

    mass: float = 1.0
    gravity: float = 9.81
    arm_length: float = 0.2
    Ixx: float = 0.01
    Iyy: float = 0.01
    Izz: float = 0.02
    motor_thrust_coefficient: float = 1.0
    motor_torque_coefficient: float = 0.04
    max_motor_thrust: float = 6.0
    time_step: float = 1.0 / 240.0
    linear_drag_coefficient: float = 0.08
    angular_drag_coefficient: float = 0.04
    ground_clearance: float = 0.04

    @property
    def inertia(self):
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    @property
    def inverse_inertia(self):
        return np.diag([1.0 / self.Ixx, 1.0 / self.Iyy, 1.0 / self.Izz])


@dataclass
class DroneState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray


def rotation_matrix(roll, pitch, yaw):
    """Возвращает матрицу Z-Y-X из корпуса в мировую систему координат."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def clamp_motors(motors):
    motors = np.asarray(motors, dtype=float).reshape(4)
    return np.clip(motors, 0.0, 1.0)


def motor_thrusts(motors, parameters):
    """Преобразует нормированные команды четырех моторов в тяги."""
    return parameters.max_motor_thrust * parameters.motor_thrust_coefficient * clamp_motors(motors)


def motor_forces_and_moments(motors, parameters):
    """Схема: M1 перед-слева, M2 перед-справа, M3 зад-справа, M4 зад-слева."""
    thrusts = motor_thrusts(motors, parameters)
    total_thrust = float(np.sum(thrusts))
    roll_torque = parameters.arm_length * (thrusts[0] - thrusts[1] - thrusts[2] + thrusts[3])
    pitch_torque = parameters.arm_length * (-thrusts[0] - thrusts[1] + thrusts[2] + thrusts[3])
    yaw_torque = parameters.motor_torque_coefficient * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
    return thrusts, total_thrust, np.array([roll_torque, pitch_torque, yaw_torque])


def compute_forces_and_moments(state, motors, parameters, wind=None):
    """Вычисляет силы и моменты в мировых и корпусных координатах."""
    wind = np.zeros(3, dtype=float) if wind is None else np.asarray(wind, dtype=float)
    thrusts, total_thrust, motor_moment = motor_forces_and_moments(motors, parameters)
    rotation = rotation_matrix(*state.orientation)
    thrust_force = rotation @ np.array([0.0, 0.0, total_thrust])
    gravity_force = np.array([0.0, 0.0, -parameters.mass * parameters.gravity])
    relative_velocity = state.velocity - wind
    drag_force = -parameters.linear_drag_coefficient * relative_velocity
    total_force = thrust_force + gravity_force + drag_force
    damping_moment = -parameters.angular_drag_coefficient * state.angular_velocity
    return {
        "motor_thrusts": thrusts,
        "total_thrust": total_thrust,
        "thrust_force": thrust_force,
        "gravity_force": gravity_force,
        "drag_force": drag_force,
        "force": total_force,
        "motor_moment": motor_moment,
        "damping_moment": damping_moment,
        "moment": motor_moment + damping_moment,
    }


def integrate_state(state, motors, parameters, wind=None):
    """Один явный Euler-шаг поступательной и вращательной динамики."""
    forces = compute_forces_and_moments(state, motors, parameters, wind)
    acceleration = forces["force"] / parameters.mass
    inertia_omega = parameters.inertia @ state.angular_velocity
    gyroscopic = np.cross(state.angular_velocity, inertia_omega)
    angular_acceleration = parameters.inverse_inertia @ (forces["moment"] - gyroscopic)
    dt = parameters.time_step

    new_velocity = state.velocity + acceleration * dt
    new_position = state.position + new_velocity * dt
    new_angular_velocity = state.angular_velocity + angular_acceleration * dt
    orientation = Rotation.from_euler("xyz", state.orientation)
    body_rotation = Rotation.from_rotvec(new_angular_velocity * dt)
    new_orientation = (orientation * body_rotation).as_euler("xyz")

    if new_position[2] < parameters.ground_clearance:
        new_position[2] = parameters.ground_clearance
        if new_velocity[2] < 0.0:
            new_velocity[2] = 0.0

    return DroneState(new_position, new_velocity, new_orientation, new_angular_velocity), forces


def mix_controls(total_thrust, roll, pitch, yaw, parameters):
    """Смешивает команды тяги и моментов в четыре команды моторов."""
    hover = float(total_thrust) / (4.0 * parameters.max_motor_thrust)
    roll_term = float(roll) / (4.0 * parameters.arm_length * parameters.max_motor_thrust)
    pitch_term = float(pitch) / (4.0 * parameters.arm_length * parameters.max_motor_thrust)
    yaw_term = float(yaw) / (4.0 * parameters.motor_torque_coefficient * parameters.max_motor_thrust)
    return clamp_motors([
        hover + roll_term - pitch_term + yaw_term,
        hover - roll_term - pitch_term - yaw_term,
        hover - roll_term + pitch_term + yaw_term,
        hover + roll_term + pitch_term - yaw_term,
    ])


def calculate_air_density(altitude):
    """Простая стандартная модель плотности воздуха для внешних экспериментов."""
    rho_0 = 1.225
    temperature = max(216.65, 288.15 - 0.0065 * float(altitude))
    exponent = -9.81 / (287.0 * 0.0065)
    return rho_0 * (temperature / 288.15) ** exponent
