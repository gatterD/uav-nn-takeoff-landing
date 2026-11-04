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
    body_half_length: float = 0.12
    body_half_width: float = 0.08
    ground_friction_coefficient: float = 0.7
    ground_angular_damping: float = 8.0

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


def ground_contact_height(orientation, parameters):
    """Высота центра для опоры ориентированного прямоугольного корпуса."""
    world_up_in_body_axes = rotation_matrix(*orientation)[2]
    half_extents = np.array([
        parameters.body_half_length,
        parameters.body_half_width,
        parameters.ground_clearance,
    ])
    return float(np.dot(np.abs(world_up_in_body_axes), half_extents))


def clamp_motors(motors):
    motors = np.asarray(motors, dtype=float).reshape(4)
    return np.clip(motors, 0.0, 1.0)


def motor_thrusts(motors, parameters):
    """Преобразует нормированные команды четырех моторов в тяги."""
    return parameters.max_motor_thrust * parameters.motor_thrust_coefficient * clamp_motors(motors)


def motor_positions(parameters):
    """X layout, with arm_length measured radially from the center of mass.

    The body frame is X forward, Y left, and Z up. Motor order is front-left,
    front-right, rear-right, rear-left.
    """
    offset = parameters.arm_length / np.sqrt(2.0)
    return np.array([
        [offset, offset, 0.0],
        [offset, -offset, 0.0],
        [-offset, -offset, 0.0],
        [-offset, offset, 0.0],
    ])


def motor_forces_and_moments(motors, parameters):
    """Compute r x F moments; M1/M3 spin CW and M2/M4 CCW viewed from +Z."""
    thrusts = motor_thrusts(motors, parameters)
    total_thrust = float(np.sum(thrusts))
    thrust_vectors = thrusts[:, None] * np.array([0.0, 0.0, 1.0])
    moments = np.cross(motor_positions(parameters), thrust_vectors).sum(axis=0)
    moments[2] += parameters.motor_torque_coefficient * np.dot(
        [1.0, -1.0, 1.0, -1.0], thrusts
    )
    return thrusts, total_thrust, moments


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

    contact_height = ground_contact_height(new_orientation, parameters)
    if new_position[2] <= contact_height:
        impact_speed = max(0.0, -new_velocity[2])
        new_position[2] = contact_height
        if new_velocity[2] < 0.0:
            new_velocity[2] = 0.0

        horizontal_speed = np.linalg.norm(new_velocity[:2])
        friction_delta = parameters.ground_friction_coefficient * (
            parameters.gravity * dt + impact_speed
        )
        if horizontal_speed > 0.0:
            new_velocity[:2] *= max(0.0, 1.0 - friction_delta / horizontal_speed)
        new_angular_velocity *= np.exp(-parameters.ground_angular_damping * dt)

    return DroneState(new_position, new_velocity, new_orientation, new_angular_velocity), forces


def mix_controls(total_thrust, roll, pitch, yaw, parameters):
    """Смешивает команды тяги и моментов в четыре команды моторов."""
    thrust_gain = parameters.max_motor_thrust * parameters.motor_thrust_coefficient
    arm_projection = parameters.arm_length / np.sqrt(2.0)
    hover = float(total_thrust) / (4.0 * thrust_gain)
    roll_term = float(roll) / (4.0 * arm_projection * thrust_gain)
    pitch_term = float(pitch) / (4.0 * arm_projection * thrust_gain)
    yaw_term = float(yaw) / (
        4.0 * parameters.motor_torque_coefficient * thrust_gain
    )
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
