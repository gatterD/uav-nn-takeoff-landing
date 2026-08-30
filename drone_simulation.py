import numpy as np
import pybullet as p
import pybullet_data


class DroneEnv:

    def __init__(self, gui=True):
        self.gui = gui
        self.TIME_STEP = 1 / 240

        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw_rate = 0.0

        self.mass = 1.0
        self.wing_area = 0.2
        self.air_density = 1.225
        self.inertia = np.array([0.01, 0.01, 0.02], dtype=float)
        self.wind = np.array([0.0, 0.0, 0.0], dtype=float)
        self.turbulence_amplitude = 0.03
        self.wind_shear_factor = 0.02
        self.turbulence = 0.0
        self.wind_shear = 0.0
        self.ground_effect = 0.0

        self._reset_state()
        self._init_simulation()

    def set_environment(self, wind=None, air_density=None,
                        turbulence_amplitude=None,
                        wind_shear_factor=None):
        if wind is not None:
            self.wind = np.array(wind, dtype=float)
        if air_density is not None:
            self.air_density = float(air_density)
        if turbulence_amplitude is not None:
            self.turbulence_amplitude = float(turbulence_amplitude)
        if wind_shear_factor is not None:
            self.wind_shear_factor = float(wind_shear_factor)

    def set_wind(self, wind_x=0.0, wind_y=0.0, wind_z=0.0):
        self.wind = np.array([wind_x, wind_y, wind_z], dtype=float)

    def set_air_density(self, air_density):
        self.air_density = float(air_density)

    def set_turbulence_amplitude(self, turbulence_amplitude):
        self.turbulence_amplitude = float(turbulence_amplitude)

    def set_wind_shear_factor(self, wind_shear_factor):
        self.wind_shear_factor = float(wind_shear_factor)

    def _reset_state(self):
        self.position = np.array([0.0, 0.0, 0.15], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.thrust = 0.0
        self.aero_force = np.array([0.0, 0.0, 0.0], dtype=float)
        self.aero_moment = np.array([0.0, 0.0, 0.0], dtype=float)

    def _init_simulation(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0.0)
        p.setTimeStep(self.TIME_STEP)

        self.plane_id = p.loadURDF("plane.urdf")
        self._create_drone()
        self._sync_visual_state()

        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1.5],
        )

    def _create_drone(self):
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.12, 0.08, 0.04],
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.12, 0.08, 0.04],
            rgbaColor=[0.0, 1.0, 0.0, 1.0],
        )
        self.drone_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.0, 0.0, 0.15],
        )
        p.changeDynamics(self.drone_id, -1, linearDamping=0.0, angularDamping=0.0)

    def _sync_visual_state(self):
        quaternion = p.getQuaternionFromEuler([self.roll, self.pitch, self.yaw])
        p.resetBasePositionAndOrientation(self.drone_id, self.position.tolist(), quaternion)
        p.resetBaseVelocity(self.drone_id, self.velocity.tolist(), self.angular_velocity.tolist())

    def _compute_aerodynamics(self):
        airspeed_vector = self.velocity + self.wind
        airspeed = np.linalg.norm(airspeed_vector)
        if airspeed < 1e-6:
            airspeed_vector = np.array([1e-6, 0.0, 0.0])
            airspeed = 1e-6

        angle_of_attack = np.arctan2(airspeed_vector[2], airspeed_vector[0])
        sideslip_angle = np.arctan2(airspeed_vector[1], airspeed_vector[0])
        flight_path_angle = np.arctan2(self.velocity[2], np.linalg.norm(self.velocity[:2]))
        path_angle = np.arctan2(self.velocity[1], self.velocity[0])
        roll_angle = self.roll

        self.turbulence = self.turbulence_amplitude * np.sin(self.position[0] * 0.3 + self.position[2] * 0.7)
        self.wind_shear = self.wind_shear_factor * max(0.0, 0.5 - self.position[2])
        self.ground_effect = 0.35 if self.position[2] < 0.3 else 0.0

        dynamic_pressure = 0.5 * self.air_density * airspeed**2
        lift = dynamic_pressure * self.wing_area * 0.8 * np.sin(angle_of_attack)
        drag = dynamic_pressure * self.wing_area * 0.2 * np.cos(angle_of_attack)
        lateral = dynamic_pressure * self.wing_area * 0.05 * np.sin(sideslip_angle)

        aero_force = np.array([
            -drag,
            lateral,
            lift,
        ], dtype=float)

        aero_force += np.array([
            self.wind_shear * 0.02,
            self.turbulence * 0.01,
            self.turbulence * 0.02,
        ], dtype=float)

        aero_moment = np.array([
            self.angular_velocity[0] * 0.02,
            self.angular_velocity[1] * 0.015,
            self.angular_velocity[2] * 0.01,
        ], dtype=float)

        self.aero_force = aero_force
        self.aero_moment = aero_moment
        self.air_speed = airspeed
        self.angle_of_attack = angle_of_attack
        self.sideslip_angle = sideslip_angle
        self.flight_path_angle = flight_path_angle
        self.path_angle = path_angle
        self.roll_angle = roll_angle

    def get_state(self):
        self._compute_aerodynamics()
        return {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "z": float(self.position[2]),
            "vx": float(self.velocity[0]),
            "vy": float(self.velocity[1]),
            "vz": float(self.velocity[2]),
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "yaw": float(self.yaw),
            "wx": float(self.angular_velocity[0]),
            "wy": float(self.angular_velocity[1]),
            "wz": float(self.angular_velocity[2]),
            "mass": float(self.mass),
            "air_speed": float(self.air_speed),
            "angle_of_attack": float(self.angle_of_attack),
            "sideslip_angle": float(self.sideslip_angle),
            "flight_path_angle": float(self.flight_path_angle),
            "path_angle": float(self.path_angle),
            "roll_angle": float(self.roll_angle),
            "thrust": float(self.thrust),
            "aero_force_x": float(self.aero_force[0]),
            "aero_force_y": float(self.aero_force[1]),
            "aero_force_z": float(self.aero_force[2]),
            "inertia_x": float(self.inertia[0]),
            "inertia_y": float(self.inertia[1]),
            "inertia_z": float(self.inertia[2]),
            "angular_velocity_x": float(self.angular_velocity[0]),
            "angular_velocity_y": float(self.angular_velocity[1]),
            "angular_velocity_z": float(self.angular_velocity[2]),
            "aero_moment_x": float(self.aero_moment[0]),
            "aero_moment_y": float(self.aero_moment[1]),
            "aero_moment_z": float(self.aero_moment[2]),
            "air_density": float(self.air_density),
            "wing_area": float(self.wing_area),
            "wind_x": float(self.wind[0]),
            "wind_y": float(self.wind[1]),
            "wind_z": float(self.wind[2]),
            "turbulence": float(self.turbulence),
            "wind_shear": float(self.wind_shear),
            "ground_effect": float(self.ground_effect),
        }

    def stabilize(self, throttle):
        return np.clip(
            np.array([throttle, throttle, throttle, throttle], dtype=float),
            0.0,
            1.0,
        )

    def step(self, action):
        throttle = float(action["throttle"])
        self.target_roll = float(action["roll"])
        self.target_pitch = float(action["pitch"])
        self.target_yaw_rate = float(action["yaw_rate"])

        motors = self.stabilize(throttle)
        self.thrust = np.mean(motors) * 20.0

        self._compute_aerodynamics()

        acceleration = (self.aero_force + np.array([0.0, 0.0, self.thrust])) / self.mass
        acceleration[2] -= 9.81
        angular_acceleration = self.aero_moment / self.inertia

        self.velocity += acceleration * self.TIME_STEP
        self.angular_velocity += angular_acceleration * self.TIME_STEP

        self.velocity[2] *= 0.98

        self.position[2] += self.velocity[2] * self.TIME_STEP
        if self.position[2] < 0.0:
            self.position[2] = 0.0
            self.velocity[2] = 0.0

        self.velocity[:2] *= 0.96
        self.position[:2] += self.velocity[:2] * self.TIME_STEP

        self.roll += (self.target_roll - self.roll) * 0.2
        self.pitch += (self.target_pitch - self.pitch) * 0.2
        self.yaw += self.target_yaw_rate * self.TIME_STEP
        self.angular_velocity *= 0.9

        self._sync_visual_state()
        p.stepSimulation()

        return self.get_state()

    def reset(self):
        self._reset_state()
        self._sync_visual_state()
        return self.get_state()

    def close(self):
        p.disconnect()
        return None