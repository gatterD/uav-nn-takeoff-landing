"""Среда симуляции: состояние считает drone_physic, PyBullet только визуализирует."""

from dataclasses import replace

import numpy as np
import pybullet as p
import pybullet_data

from drone_physic import DroneParameters, DroneState, integrate_state


class DroneEnv:
    def __init__(self, gui=True, parameters=None):
        self.gui = gui
        self.parameters = parameters or DroneParameters()
        self.TIME_STEP = self.parameters.time_step
        self.wind = np.zeros(3, dtype=float)
        self.motor_commands = np.zeros(4, dtype=float)
        self._reset_state()
        self._init_simulation()

    def set_environment(self, wind=None, mass=None, max_motor_thrust=None,
                        linear_drag_coefficient=None, angular_drag_coefficient=None,
                        **_ignored):
        if wind is not None:
            self.wind = np.asarray(wind, dtype=float)
        changes = {}
        for name, value in {
            "mass": mass,
            "max_motor_thrust": max_motor_thrust,
            "linear_drag_coefficient": linear_drag_coefficient,
            "angular_drag_coefficient": angular_drag_coefficient,
        }.items():
            if value is not None:
                changes[name] = float(value)
        if changes:
            self.parameters = replace(self.parameters, **changes)
            self.TIME_STEP = self.parameters.time_step
            if hasattr(self, "drone_id"):
                p.changeDynamics(self.drone_id, -1, mass=self.parameters.mass)

    def set_wind(self, wind_x=0.0, wind_y=0.0, wind_z=0.0):
        self.wind = np.array([wind_x, wind_y, wind_z], dtype=float)

    def _reset_state(self):
        self.state = DroneState(
            position=np.array([0.0, 0.0, 0.15], dtype=float),
            velocity=np.zeros(3, dtype=float),
            orientation=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        )
        self.last_forces = {
            "motor_thrusts": np.zeros(4),
            "total_thrust": 0.0,
            "thrust_force": np.zeros(3),
            "gravity_force": np.zeros(3),
            "drag_force": np.zeros(3),
            "force": np.zeros(3),
            "motor_moment": np.zeros(3),
            "damping_moment": np.zeros(3),
            "moment": np.zeros(3),
        }

    def _init_simulation(self):
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setTimeStep(self.TIME_STEP)
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_drone()
        self._sync_visual_state()
        if self.gui:
            p.resetDebugVisualizerCamera(4.0, 45, -30, [0, 0, 1.5])

    def _create_drone(self):
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.parameters.body_half_length,
                self.parameters.body_half_width,
                self.parameters.ground_clearance,
            ],
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.parameters.body_half_length,
                self.parameters.body_half_width,
                self.parameters.ground_clearance,
            ],
            rgbaColor=[0.0, 1.0, 0.0, 1.0],
        )
        self.drone_id = p.createMultiBody(
            baseMass=self.parameters.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.state.position.tolist(),
        )
        p.changeDynamics(self.drone_id, -1, linearDamping=0.0, angularDamping=0.0)

    def _sync_visual_state(self):
        quaternion = p.getQuaternionFromEuler(self.state.orientation.tolist())
        p.resetBasePositionAndOrientation(self.drone_id, self.state.position.tolist(), quaternion)
        p.resetBaseVelocity(
            self.drone_id,
            self.state.velocity.tolist(),
            self.state.angular_velocity.tolist(),
        )

    def _action_to_motors(self, action):
        unsupported_controls = {"roll", "pitch", "yaw", "yaw_rate"}.intersection(action)
        if unsupported_controls:
            controls = ", ".join(sorted(unsupported_controls))
            raise ValueError(f"Attitude controls are not supported; provide motor commands instead: {controls}")
        if "motors" in action:
            return np.asarray(action["motors"], dtype=float).reshape(4)
        motor_keys = ["motor_1", "motor_2", "motor_3", "motor_4"]
        if all(key in action for key in motor_keys):
            return np.array([action[key] for key in motor_keys], dtype=float)
        throttle = float(action.get("throttle", 0.0))
        return np.full(4, throttle, dtype=float)

    def get_state(self):
        state = {
            "x": float(self.state.position[0]), "y": float(self.state.position[1]), "z": float(self.state.position[2]),
            "vx": float(self.state.velocity[0]), "vy": float(self.state.velocity[1]), "vz": float(self.state.velocity[2]),
            "roll": float(self.state.orientation[0]), "pitch": float(self.state.orientation[1]), "yaw": float(self.state.orientation[2]),
            "wx": float(self.state.angular_velocity[0]), "wy": float(self.state.angular_velocity[1]), "wz": float(self.state.angular_velocity[2]),
            "mass": float(self.parameters.mass), "max_motor_thrust": float(self.parameters.max_motor_thrust),
            "thrust": float(self.last_forces["total_thrust"]),
            "motor_1": float(self.motor_commands[0]), "motor_2": float(self.motor_commands[1]),
            "motor_3": float(self.motor_commands[2]), "motor_4": float(self.motor_commands[3]),
            "total_thrust": float(self.last_forces["total_thrust"]),
            "wind_x": float(self.wind[0]), "wind_y": float(self.wind[1]), "wind_z": float(self.wind[2]),
            "linear_drag_x": float(self.last_forces["drag_force"][0]),
            "linear_drag_y": float(self.last_forces["drag_force"][1]),
            "linear_drag_z": float(self.last_forces["drag_force"][2]),
            "aero_force_x": float(self.last_forces["drag_force"][0]),
            "aero_force_y": float(self.last_forces["drag_force"][1]),
            "aero_force_z": float(self.last_forces["drag_force"][2]),
            "aero_moment_x": float(self.last_forces["damping_moment"][0]),
            "aero_moment_y": float(self.last_forces["damping_moment"][1]),
            "aero_moment_z": float(self.last_forces["damping_moment"][2]),
            "inertia_x": float(self.parameters.Ixx), "inertia_y": float(self.parameters.Iyy), "inertia_z": float(self.parameters.Izz),
            "angular_velocity_x": float(self.state.angular_velocity[0]),
            "angular_velocity_y": float(self.state.angular_velocity[1]),
            "angular_velocity_z": float(self.state.angular_velocity[2]),
        }
        return state

    def stabilize(self, throttle):
        return np.full(4, np.clip(float(throttle), 0.0, 1.0))

    def step(self, action):
        """Advances one step using four motor commands or a shared throttle."""
        self.motor_commands = np.clip(self._action_to_motors(action), 0.0, 1.0)
        self.state, self.last_forces = integrate_state(
            self.state, self.motor_commands, self.parameters, self.wind
        )
        self._sync_visual_state()
        p.stepSimulation(physicsClientId=self.physics_client)
        return self.get_state()

    def reset(self):
        self._reset_state()
        self.motor_commands = np.zeros(4, dtype=float)
        self._sync_visual_state()
        return self.get_state()

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
        return None
