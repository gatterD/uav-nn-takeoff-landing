import numpy as np

class DronePhysics:
    def __init__(self, mass=1.0, arm_length=0.25, max_thrust=10.0, 
                 moment_inertia=np.diag([0.01, 0.01, 0.02])):
        """
        Физическая модель квадрокоптера
        
        Параметры:
        mass - масса дрона (кг)
        arm_length - длина плеча (м)
        max_thrust - максимальная тяга одного мотора (Н)
        moment_inertia - тензор инерции (кг*м^2)
        """
        self.mass = mass
        self.arm_length = arm_length
        self.max_thrust = max_thrust
        self.moment_inertia = moment_inertia
        self.gravity = 9.81
        
        # Параметры для смешивания тяги и моментов
        self.thrust_coef = 1.0  # коэффициент тяги
        self.torque_coef = 0.1  # коэффициент момента рыскания
        
        # Состояние дрона (используем float64 для всех массивов)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.orientation = np.array([0, 0, 0, 1], dtype=np.float64)  # кватернион (x, y, z, w)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        
        # Силы и моменты
        self.forces = np.zeros(3, dtype=np.float64)
        self.moments = np.zeros(3, dtype=np.float64)
        
        # Выходные данные моторов (скорости)
        self.motor_speeds = np.zeros(4, dtype=np.float64)
        
    def set_motor_speeds(self, speeds):
        """Установка скоростей моторов"""
        self.motor_speeds = np.clip(speeds, 0, 1).astype(np.float64)  # нормализованные значения 0-1
        
    def compute_forces_and_moments(self):
        """Вычисление сил и моментов от моторов"""
        # Тяга каждого мотора пропорциональна скорости
        thrusts = self.motor_speeds * self.max_thrust
        
        # Суммарная тяга (вдоль оси Z дрона)
        total_thrust = np.sum(thrusts)
        
        # Моменты (вращение вокруг осей)
        # X - крен (roll)
        moment_x = self.arm_length * (thrusts[1] - thrusts[3])
        # Y - тангаж (pitch)
        moment_y = self.arm_length * (thrusts[0] - thrusts[2])
        # Z - рыскание (yaw)
        moment_z = self.torque_coef * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
        
        # Применяем гравитацию (в мировых координатах)
        gravity_force = np.array([0, 0, -self.mass * self.gravity], dtype=np.float64)
        
        # Поворачиваем тягу в мировые координаты
        # Кватернионное вращение вектора (0, 0, total_thrust)
        rotated_thrust = self.rotate_vector_by_quaternion(
            np.array([0, 0, total_thrust], dtype=np.float64), self.orientation
        )
        
        # Суммарная сила
        self.forces = gravity_force + rotated_thrust
        
        # Моменты
        self.moments = np.array([moment_x, moment_y, moment_z], dtype=np.float64)
        
    def rotate_vector_by_quaternion(self, vector, quaternion):
        """Поворот вектора кватернионом"""
        x, y, z, w = quaternion
        vx, vy, vz = vector
        
        # Формула поворота: q * v * q^(-1)
        # Вычисляем произведение кватернионов
        qx, qy, qz, qw = x, y, z, w
        
        # q * v (где v - вектор как кватернион)
        tx = qw * vx + qy * vz - qz * vy
        ty = qw * vy + qz * vx - qx * vz
        tz = qw * vz + qx * vy - qy * vx
        tw = -qx * vx - qy * vy - qz * vz
        
        # (q * v) * q^(-1)
        result_x = tw * (-qx) + tx * qw + ty * (-qz) - tz * (-qy)
        result_y = tw * (-qy) + ty * qw + tz * (-qx) - tx * (-qz)
        result_z = tw * (-qz) + tz * qw + tx * (-qy) - ty * (-qx)
        
        return np.array([result_x, result_y, result_z], dtype=np.float64)
    
    def update(self, dt):
        """Обновление состояния дрона"""
        # Вычисляем силы и моменты
        self.compute_forces_and_moments()
        
        # Обновление линейной скорости
        acceleration = self.forces / self.mass
        self.velocity += acceleration * dt
        
        # Обновление позиции
        self.position += self.velocity * dt
        
        # Обновление угловой скорости
        try:
            # Решаем систему для получения углового ускорения
            angular_acceleration = np.linalg.solve(self.moment_inertia, self.moments)
        except np.linalg.LinAlgError:
            # Если матрица инерции вырожденная, используем псевдообратную
            angular_acceleration = np.linalg.lstsq(self.moment_inertia, self.moments, rcond=None)[0]
            
        self.angular_velocity += angular_acceleration * dt
        
        # Обновление ориентации (кватернион)
        # Используем метод полу-шага для большей точности
        omega = self.angular_velocity
        q = self.orientation
        
        # Создаем кватернион скорости вращения
        # q_dot = 0.5 * omega * q
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]  # w, x, y, z
        wx, wy, wz = omega
        
        q_dot = 0.5 * np.array([
            wx * qw + wz * qy - wy * qz,
            wy * qw + wx * qz - wz * qx,
            wz * qw + wy * qx - wx * qy,
            -wx * qx - wy * qy - wz * qz
        ], dtype=np.float64)
        
        # Обновляем кватернион
        self.orientation += q_dot * dt
        
        # Нормализуем кватернион
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation = self.orientation / norm
        
        # Ограничиваем значения, чтобы избежать численной нестабильности
        self.orientation = np.clip(self.orientation, -1.0, 1.0)