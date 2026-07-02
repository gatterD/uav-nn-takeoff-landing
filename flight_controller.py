import numpy as np

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.integral_limit = integral_limit
        
    def update(self, error, dt):
        """Обновление контроллера"""
        # Пропорциональная часть
        p = self.kp * error
        
        # Интегральная часть
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral
        
        # Дифференциальная часть
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        d = self.kd * derivative
        
        self.previous_error = error
        return p + i + d
    
    def reset(self):
        """Сброс контроллера"""
        self.integral = 0.0
        self.previous_error = 0.0


class FlightController:
    def __init__(self):
        # Параметры дрона
        self.mass = 1.0
        self.gravity = 9.81
        self.max_thrust_per_motor = 10.0
        
        # PID контроллеры - более мягкие настройки для плавного полета
        # Высота - настроены для плавного взлета
        self.altitude_pid = PIDController(kp=3.0, ki=0.3, kd=1.0, integral_limit=2.0)
        self.velocity_z_pid = PIDController(kp=2.0, ki=0.2, kd=0.5)
        
        # Крен (roll)
        self.roll_pid = PIDController(kp=4.0, ki=0.1, kd=0.8, integral_limit=1.0)
        self.roll_rate_pid = PIDController(kp=2.0, ki=0.0, kd=0.2)
        
        # Тангаж (pitch)
        self.pitch_pid = PIDController(kp=4.0, ki=0.1, kd=0.8, integral_limit=1.0)
        self.pitch_rate_pid = PIDController(kp=2.0, ki=0.0, kd=0.2)
        
        # Рыскание (yaw)
        self.yaw_pid = PIDController(kp=1.5, ki=0.05, kd=0.3, integral_limit=0.5)
        self.yaw_rate_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)
        
        # Целевые значения
        self.target_position = np.array([0, 0, 0.2])  # начальная цель - земля
        self.target_heading = 0.0
        
        # Состояние
        self.arm_thrust = self.mass * self.gravity / 4.0
        
    def compute_control(self, drone_state, dt):
        """
        Вычисление управляющих сигналов для моторов
        """
        # Извлекаем состояние
        position = drone_state['position']
        velocity = drone_state['velocity']
        orientation = drone_state['orientation']
        angular_velocity = drone_state['angular_velocity']
        
        # Преобразуем кватернион в углы Эйлера
        roll, pitch, yaw = self.quaternion_to_euler(orientation)
        
        # --- Управление высотой ---
        altitude_error = self.target_position[2] - position[2]
        velocity_z_target = self.altitude_pid.update(altitude_error, dt)
        velocity_z_error = velocity_z_target - velocity[2]
        throttle = self.velocity_z_pid.update(velocity_z_error, dt)
        
        # Базовая тяга + управление с ограничением
        total_thrust = self.mass * self.gravity + throttle
        total_thrust = np.clip(total_thrust, 0, 4 * self.max_thrust_per_motor)
        
        # --- Управление по горизонтали ---
        target_roll = 0.0
        target_pitch = 0.0
        
        # PID для стабилизации крена
        roll_error = target_roll - roll
        roll_rate_target = self.roll_pid.update(roll_error, dt)
        roll_rate_error = roll_rate_target - angular_velocity[0]
        roll_control = self.roll_rate_pid.update(roll_rate_error, dt)
        
        # PID для стабилизации тангажа
        pitch_error = target_pitch - pitch
        pitch_rate_target = self.pitch_pid.update(pitch_error, dt)
        pitch_rate_error = pitch_rate_target - angular_velocity[1]
        pitch_control = self.pitch_rate_pid.update(pitch_rate_error, dt)
        
        # PID для рыскания
        yaw_error = self.target_heading - yaw
        yaw_rate_target = self.yaw_pid.update(yaw_error, dt)
        yaw_rate_error = yaw_rate_target - angular_velocity[2]
        yaw_control = self.yaw_rate_pid.update(yaw_rate_error, dt)
        
        # Смешивание для получения скоростей моторов
        base_thrust = total_thrust / 4.0
        thrust_mix = np.array([
            base_thrust + roll_control + pitch_control + yaw_control,
            base_thrust - roll_control + pitch_control - yaw_control,
            base_thrust - roll_control - pitch_control + yaw_control,
            base_thrust + roll_control - pitch_control - yaw_control
        ])
        
        # Нормализация к скоростям моторов (0-1)
        motor_speeds = np.clip(thrust_mix / self.max_thrust_per_motor, 0, 1)
        
        return motor_speeds
    
    def quaternion_to_euler(self, quaternion):
        """Преобразование кватерниона в углы Эйлера"""
        x, y, z, w = quaternion
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def set_target_position(self, position):
        """Установка целевой позиции"""
        self.target_position = np.array(position)
    
    def set_target_heading(self, heading):
        """Установка целевого курса"""
        self.target_heading = heading