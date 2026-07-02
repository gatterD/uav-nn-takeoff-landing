import numpy as np
import pybullet as p
import pybullet_data
import time

class DroneEnvironment:
    def __init__(self, gui=True):
        self.gui = gui
        
        # Инициализация PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # Настройка симуляции
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # шаг за шагом
        
        # Загрузка окружения
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Параметры дрона
        self.drone_id = None
        self.visual_shape_id = None
        self.collision_shape_id = None
        
        # Состояние дрона из физического движка
        self.drone_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': np.array([0, 0, 0, 1]),
            'angular_velocity': np.zeros(3)
        }
        
        self.step_count = 0
        self.dt = 0.01  # шаг симуляции (сек)
        
        # Создание дрона
        self.create_drone()
        
    def create_drone(self):
        """Создание визуальной модели дрона в PyBullet"""
        # Создаем тело дрона как куб с массой
        half_extents = [0.2, 0.2, 0.05]
        mass = 1.0
        
        # Создаем коллизионную форму
        self.collision_shape_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )
        
        # Создаем визуальную форму
        self.visual_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.6, 0.9, 1.0]
        )
        
        # Создаем мультибоди
        base_position = [0, 0, 0.2]
        base_orientation = [0, 0, 0, 1]
        
        self.drone_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=self.collision_shape_id,
            baseVisualShapeIndex=self.visual_shape_id,
            basePosition=base_position,
            baseOrientation=base_orientation
        )
        
        # Добавляем инерцию для вращения
        p.changeDynamics(
            self.drone_id,
            -1,
            lateralFriction=0.3,
            spinningFriction=0.1,
            rollingFriction=0.1,
            restitution=0.2
        )
        
        # Добавляем моторы как визуальные элементы
        self.motor_indicators = []
        motor_positions = [
            [0.3, 0.3, 0.05],
            [-0.3, 0.3, 0.05],
            [-0.3, -0.3, 0.05],
            [0.3, -0.3, 0.05]
        ]
        motor_colors = [
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0.5],
            [1, 1, 0, 0.5]
        ]
        
        for i, (pos, color) in enumerate(zip(motor_positions, motor_colors)):
            motor_id = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.04,
                length=0.02,
                rgbaColor=color
            )
            
            # Привязываем мотор к дрону
            motor_body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=motor_id,
                basePosition=[pos[0], pos[1], pos[2]]
            )
            
            # Создаем ограничение, чтобы мотор был привязан к дрону
            p.createConstraint(
                parentBodyUniqueId=self.drone_id,
                parentLinkIndex=-1,
                childBodyUniqueId=motor_body,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 1],
                parentFramePosition=[pos[0], pos[1], pos[2]],
                childFramePosition=[0, 0, 0]
            )
            
            self.motor_indicators.append(motor_body)
            
    def apply_motor_forces(self, motor_speeds):
        """Применение сил от моторов к дрону"""
        # Получаем текущую позицию и ориентацию
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        
        # Создаем вектор тяги в локальных координатах
        thrust = np.sum(motor_speeds) * 10.0  # максимальная тяга 10N на мотор
        
        # Применяем силу в направлении вверх в локальных координатах
        # Преобразуем локальное направление в глобальное
        local_up = [0, 0, 1]
        global_up = p.multiplyTransforms([0, 0, 0], orn, local_up, [0, 0, 0, 1])[0]
        
        # Применяем силу в центре масс
        force = [thrust * global_up[0], thrust * global_up[1], thrust * global_up[2]]
        p.applyExternalForce(
            self.drone_id,
            -1,
            force,
            pos,
            p.LINK_FRAME
        )
        
        # Применяем моменты для вращения
        # Упрощенно: создаем моменты на основе разницы скоростей моторов
        torque = np.array([
            (motor_speeds[1] - motor_speeds[3]) * 0.1,  # roll
            (motor_speeds[0] - motor_speeds[2]) * 0.1,  # pitch
            (motor_speeds[0] - motor_speeds[1] + motor_speeds[2] - motor_speeds[3]) * 0.05  # yaw
        ])
        
        # Применяем момент к телу
        p.applyExternalTorque(
            self.drone_id,
            -1,
            torque.tolist(),
            p.LINK_FRAME
        )
        
    def step(self, motor_speeds=None):
        """Шаг симуляции"""
        # Применяем управление, если задано
        if motor_speeds is not None:
            self.apply_motor_forces(motor_speeds)
            
        # Шаг симуляции
        p.stepSimulation()
        self.step_count += 1
        
        # Обновляем состояние
        self.update_state()
        
        # Если есть GUI, даем время на отрисовку
        if self.gui:
            time.sleep(self.dt)
            
        return self.drone_state.copy()
    
    def update_state(self):
        """Обновление состояния дрона из PyBullet"""
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        self.drone_state['position'] = np.array(pos)
        self.drone_state['velocity'] = np.array(vel)
        self.drone_state['orientation'] = np.array(orn)
        self.drone_state['angular_velocity'] = np.array(ang_vel)
        
    def reset(self):
        """Сброс симуляции"""
        p.resetBasePositionAndOrientation(self.drone_id, [0, 0, 0.2], [0, 0, 0, 1])
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
        self.step_count = 0
        self.update_state()
        
    def close(self):
        """Закрытие симуляции"""
        p.disconnect()
        
    def get_state(self):
        """Получение текущего состояния"""
        return self.drone_state.copy()