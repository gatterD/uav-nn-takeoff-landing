import numpy as np


# =============================================================
# АЭРОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ
# =============================================================

class DroneAerodynamics:
    """
    Класс для моделирования аэродинамических эффектов дрона.
    Включает сопротивление воздуха, демпфирование и боковое смещение.
    """
    
    def __init__(self):
        # Коэффициенты сопротивления (drag coefficients)
        self.linear_drag_coef = 0.05      # Линейное сопротивление (низкие скорости)
        self.quadratic_drag_coef = 0.02   # Квадратичное сопротивление (высокие скорости)
        
        # Демпфирование угловых скоростей
        self.angular_damping = 0.1        # Демпфирование раскручивания
        
        # Боковой дрейф (небольшой эффект асимметрии)
        self.side_drift_coef = 0.02       # Коэффициент бокового дрейфа
        
        # Воздушные потоки
        self.wind_effect_x = 0.0          # Ветер по оси X
        self.wind_effect_y = 0.0          # Ветер по оси Y
        self.wind_effect_z = 0.0          # Ветер по оси Z (восходящие потоки)
        
        # Инерция (влияние на ускорение)
        self.inertia_factor = 0.95        # Факт инерции (1.0 = без инерции, 0.5 = половина)
    
    
    def calculate_drag_force(self, velocity):
        """
        Рассчитывает силу сопротивления воздуха.
        F_drag = -linear_drag * v - quadratic_drag * v^2
        
        Args:
            velocity: np.array([vx, vy, vz]) - вектор скорости
        
        Returns:
            np.array: вектор силы сопротивления
        """
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        
        # Линейное + квадратичное сопротивление
        drag_magnitude = (
            self.linear_drag_coef * speed +
            self.quadratic_drag_coef * speed ** 2
        )
        
        # Направление противоположно скорости
        drag_force = -drag_magnitude * (velocity / speed)
        
        return drag_force
    
    
    def calculate_angular_damping(self, angular_velocity):
        """
        Рассчитывает момент демпфирования для угловых скоростей.
        Препятствует быстрому раскручиванию дрона.
        
        Args:
            angular_velocity: np.array([roll_rate, pitch_rate, yaw_rate])
        
        Returns:
            np.array: вектор момента демпфирования
        """
        damping_moment = -self.angular_damping * angular_velocity
        return damping_moment
    
    
    def apply_wind_effect(self, velocity):
        """
        Применяет эффект ветра к скорости.
        
        Args:
            velocity: np.array([vx, vy, vz])
        
        Returns:
            np.array: скорость с эффектом ветра
        """
        wind = np.array([
            self.wind_effect_x,
            self.wind_effect_y,
            self.wind_effect_z
        ])
        
        return velocity + wind
    
    
    def apply_aerodynamic_effects(self, state_dict, time_step=0.01):
        """
        Применяет все аэродинамические эффекты к состоянию дрона.
        
        Args:
            state_dict: dict - словарь состояния дрона с полями:
                - vx, vy, vz: линейные скорости
                - roll_rate, pitch_rate, yaw_rate: угловые скорости
            time_step: float - временной шаг симуляции
        
        Returns:
            dict: измененное состояние с учетом аэродинамики
        """
        # Копируем состояние
        modified_state = state_dict.copy()
        
        # Текущие скорости
        velocity = np.array([
            state_dict.get("vx", 0.0),
            state_dict.get("vy", 0.0),
            state_dict.get("vz", 0.0)
        ])
        
        angular_velocity = np.array([
            state_dict.get("roll_rate", 0.0),
            state_dict.get("pitch_rate", 0.0),
            state_dict.get("yaw_rate", 0.0)
        ])
        
        # Рассчитываем силы сопротивления
        drag_force = self.calculate_drag_force(velocity)
        
        # Рассчитываем демпфирование
        angular_damping = self.calculate_angular_damping(angular_velocity)
        
        # Применяем эффект ветра
        wind_velocity = self.apply_wind_effect(velocity)
        
        # Обновляем состояние с учетом аэродинамики
        # Сопротивление немного замедляет дрон
        modified_state["vx"] = velocity[0] + drag_force[0] * time_step * self.inertia_factor
        modified_state["vy"] = velocity[1] + drag_force[1] * time_step * self.inertia_factor
        modified_state["vz"] = velocity[2] + drag_force[2] * time_step * self.inertia_factor
        
        # Демпфирование уменьшает угловые скорости
        modified_state["roll_rate"] = angular_velocity[0] + angular_damping[0] * time_step
        modified_state["pitch_rate"] = angular_velocity[1] + angular_damping[1] * time_step
        modified_state["yaw_rate"] = angular_velocity[2] + angular_damping[2] * time_step
        
        return modified_state
    
    
    def set_wind(self, wind_x=0.0, wind_y=0.0, wind_z=0.0):
        """
        Устанавливает параметры ветра.
        
        Args:
            wind_x: Ветер по оси X (м/с)
            wind_y: Ветер по оси Y (м/с)
            wind_z: Ветер по оси Z (м/с) - восходящие потоки
        """
        self.wind_effect_x = wind_x
        self.wind_effect_y = wind_y
        self.wind_effect_z = wind_z
    
    
    def reset(self):
        """Сбрасывает ветер на нулевые значения."""
        self.set_wind(0.0, 0.0, 0.0)


# =============================================================
# ФИЗИЧЕСКИЕ МОДЕЛИ
# =============================================================

def calculate_air_density(altitude):
    """
    Рассчитывает плотность воздуха на заданной высоте.
    Использует барометрическую формулу.
    
    Args:
        altitude: высота в метрах
    
    Returns:
        float: плотность воздуха (кг/м³)
    """
    # Стандартные условия на уровне моря
    rho_0 = 1.225  # кг/м³
    
    # Температурный градиент тропосферы
    temp_gradient = 0.0065  # К/м
    temp_0 = 288.15  # К (15°C)
    
    # Высота масштаба
    H = 8435  # м
    
    # Барометрическая формула
    temperature = temp_0 - temp_gradient * altitude
    if temperature <= 0:
        temperature = 216.65  # Минимальная температура
    
    exponent = -9.81 / (287 * temp_gradient)
    air_density = rho_0 * (temperature / temp_0) ** exponent
    
    return air_density


def calculate_magnus_effect(velocity, angular_velocity, characteristic_length=0.1):
    """
    Рассчитывает эффект Магнуса (подъемная сила при вращении).
    Очень слабый эффект для более реалистичной симуляции.
    
    Args:
        velocity: np.array([vx, vy, vz]) - линейная скорость
        angular_velocity: np.array([roll_rate, pitch_rate, yaw_rate]) - угловая скорость
        characteristic_length: характерный размер дрона
    
    Returns:
        np.array: вектор силы Магнуса
    """
    # Очень слабый коэффициент Магнуса для дрона
    magnus_coef = 0.001
    
    speed = np.linalg.norm(velocity)
    
    if speed < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    
    # Сила Магнуса = magnus_coef * (omega x v)
    magnus_force = magnus_coef * np.cross(angular_velocity, velocity)
    
    return magnus_force
