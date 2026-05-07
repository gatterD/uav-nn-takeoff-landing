import pybullet as p
import pybullet_data
import numpy as np


class DroneEnv:

    def __init__(self, gui=True):

        self.gui = gui

        # =========================================================
        # ФИЗИКА
        # =========================================================

        self.GRAVITY = -9.81
        self.TIME_STEP = 1 / 240

        self.DRONE_MASS = 1.5

        # Размеры корпуса
        self.DRONE_SIZE = [0.25, 0.25, 0.08]

        # Максимальная тяга одного мотора
        self.MAX_MOTOR_THRUST = 15.0

        # Расстояние от центра до мотора
        self.ARM_LENGTH = 0.25

        # Коэффициенты аэродинамики
        self.LINEAR_DRAG = 0.04
        self.QUADRATIC_DRAG = 0.02
        self.ANGULAR_DRAG = 0.05

        # Коэффициенты моментов
        self.ROLL_TORQUE_COEFF = 1.5
        self.PITCH_TORQUE_COEFF = 1.5
        self.YAW_TORQUE_COEFF = 0.4

        # Ветер
        self.WIND_FORCE = np.array([0.3, 0.0, 0.0])

        # Турбулентность
        self.TURBULENCE = 0.15

        # Шум сенсоров
        self.SENSOR_NOISE = 0.002

        # =========================================================
        # ПОДКЛЮЧЕНИЕ
        # =========================================================

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, self.GRAVITY)
        p.setTimeStep(self.TIME_STEP)

        # Улучшение стабильности физики
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.TIME_STEP,
            numSolverIterations=100,
        )

        # =========================================================
        # МИР
        # =========================================================

        self.plane_id = p.loadURDF("plane.urdf")

        self._create_drone()

    # =============================================================
    # СОЗДАНИЕ ДРОНА
    # =============================================================

    def _create_drone(self):

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=self.DRONE_SIZE
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self.DRONE_SIZE,
            rgbaColor=[0, 1, 0, 1]
        )

        self.drone_id = p.createMultiBody(
            baseMass=self.DRONE_MASS,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0.15]
        )

        # Уменьшение скольжения
        p.changeDynamics(
            self.drone_id,
            -1,
            lateralFriction=1.0,
            angularDamping=0.01,
            linearDamping=0.01
        )

    # =============================================================
    # СОСТОЯНИЕ
    # =============================================================

    def get_state(self):

        pos, orn = p.getBasePositionAndOrientation(self.drone_id)

        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)

        euler = p.getEulerFromQuaternion(orn)

        # Добавление шума сенсоров
        noise = lambda: np.random.normal(0, self.SENSOR_NOISE)

        return {

            # Позиция
            "x": pos[0] + noise(),
            "y": pos[1] + noise(),
            "z": pos[2] + noise(),

            # Скорости
            "vx": lin_vel[0] + noise(),
            "vy": lin_vel[1] + noise(),
            "vz": lin_vel[2] + noise(),

            # Углы
            "roll": euler[0] + noise(),
            "pitch": euler[1] + noise(),
            "yaw": euler[2] + noise(),

            # Угловые скорости
            "wx": ang_vel[0] + noise(),
            "wy": ang_vel[1] + noise(),
            "wz": ang_vel[2] + noise(),
        }

    # =============================================================
    # МОТОРЫ
    # =============================================================

    def apply_motor_forces(
            self,
            front_left,
            front_right,
            rear_left,
            rear_right
    ):

        motors = np.clip(
            [
                front_left,
                front_right,
                rear_left,
                rear_right
            ],
            0.0,
            1.0
        )

        thrusts = motors * self.MAX_MOTOR_THRUST

        total_thrust = np.sum(thrusts)

        # =========================================================
        # ЛОКАЛЬНАЯ ТЯГА
        # =========================================================

        pos, orn = p.getBasePositionAndOrientation(self.drone_id)

        rot_matrix = np.array(
            p.getMatrixFromQuaternion(orn)
        ).reshape(3, 3)

        local_z = rot_matrix[:, 2]

        thrust_force = local_z * total_thrust

        p.applyExternalForce(
            self.drone_id,
            -1,
            thrust_force.tolist(),
            pos,
            p.WORLD_FRAME
        )

        # =========================================================
        # МОМЕНТЫ
        # =========================================================

        roll_torque = (
                (front_right + rear_right)
                - (front_left + rear_left)
        ) * self.ROLL_TORQUE_COEFF

        pitch_torque = (
                (rear_left + rear_right)
                - (front_left + front_right)
        ) * self.PITCH_TORQUE_COEFF

        yaw_torque = (
                (front_left + rear_right)
                - (front_right + rear_left)
        ) * self.YAW_TORQUE_COEFF

        torque = [
            roll_torque,
            pitch_torque,
            yaw_torque
        ]

        p.applyExternalTorque(
            self.drone_id,
            -1,
            torque,
            p.WORLD_FRAME
        )

    # =============================================================
    # АЭРОДИНАМИКА
    # =============================================================

    def apply_aerodynamics(self):

        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)

        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)

        # =========================================================
        # ЛИНЕЙНОЕ СОПРОТИВЛЕНИЕ
        # =========================================================

        drag_linear = -self.LINEAR_DRAG * lin_vel

        # =========================================================
        # КВАДРАТИЧНОЕ СОПРОТИВЛЕНИЕ
        # =========================================================

        drag_quadratic = (
                -self.QUADRATIC_DRAG
                * np.linalg.norm(lin_vel)
                * lin_vel
        )

        total_drag = drag_linear + drag_quadratic

        # =========================================================
        # ВЕТЕР
        # =========================================================

        wind = self.WIND_FORCE.copy()

        # =========================================================
        # ТУРБУЛЕНТНОСТЬ
        # =========================================================

        turbulence = np.random.normal(
            0,
            self.TURBULENCE,
            3
        )

        aerodynamic_force = (
                total_drag
                + wind
                + turbulence
        )

        p.applyExternalForce(
            self.drone_id,
            -1,
            aerodynamic_force.tolist(),
            [0, 0, 0],
            p.WORLD_FRAME
        )

        # =========================================================
        # УГЛОВОЕ СОПРОТИВЛЕНИЕ
        # =========================================================

        angular_drag = (
                -self.ANGULAR_DRAG
                * ang_vel
        )

        p.applyExternalTorque(
            self.drone_id,
            -1,
            angular_drag.tolist(),
            p.WORLD_FRAME
        )

    # =============================================================
    # STEP
    # =============================================================

    def step(self, action):

        """
        action = [
            front_left,
            front_right,
            rear_left,
            rear_right
        ]

        диапазон:
        0..1
        """

        self.apply_motor_forces(*action)

        self.apply_aerodynamics()

        p.stepSimulation()

        return self.get_state()

    # =============================================================
    # RESET
    # =============================================================

    def reset(self):

        p.resetSimulation()

        p.setGravity(0, 0, self.GRAVITY)

        self.plane_id = p.loadURDF("plane.urdf")

        self._create_drone()

        return self.get_state()

    # =============================================================
    # CLOSE
    # =============================================================

    def close(self):
        p.disconnect()