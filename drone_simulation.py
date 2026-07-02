import numpy as np
import time
from drone_physics import DronePhysics
from flight_controller import FlightController
from drone_env import DroneEnvironment

class DroneSimulation:
    def __init__(self, use_pybullet=True, gui=True):
        """
        Основной класс симуляции дрона
        
        Параметры:
        use_pybullet - использовать ли PyBullet для визуализации
        gui - показывать ли графический интерфейс
        """
        self.use_pybullet = use_pybullet
        self.gui = gui
        self.dt = 0.01  # шаг симуляции
        self.time = 0.0
        
        # Создание компонентов
        self.physics = DronePhysics()
        self.controller = FlightController()
        
        if use_pybullet:
            try:
                self.environment = DroneEnvironment(gui=gui)
            except Exception as e:
                print(f"Ошибка при создании окружения PyBullet: {e}")
                print("Переключение на режим без визуализации")
                self.use_pybullet = False
                self.environment = None
        else:
            self.environment = None
            
        # Логгер данных
        self.logger = None
        
        # Флаг симуляции
        self.running = False
        
        # Режим полета
        self.flight_mode = 'takeoff'  # 'takeoff', 'hover', 'landing', 'landed'
        
        # Параметры полета
        self.target_height = 15.0  # высота зависания (метры)
        self.hover_duration = 5.0  # время зависания (секунды)
        self.hover_start_time = 0.0  # время начала зависания
        self.landing_start_time = 0.0  # время начала посадки
        
        # Ограничения скорости
        self.max_vertical_speed = 3.0  # максимальная вертикальная скорость (м/с)
        self.max_horizontal_speed = 2.0  # максимальная горизонтальная скорость (м/с)
        
        # Состояние дрона
        self.is_on_ground = True
        
    def initialize(self):
        """Инициализация симуляции"""
        if self.use_pybullet and self.environment:
            self.environment.reset()
            
        # Сброс физической модели (дрон на земле)
        self.physics.position = np.array([0, 0, 0.2], dtype=np.float64)
        self.physics.velocity = np.zeros(3, dtype=np.float64)
        self.physics.orientation = np.array([0, 0, 0, 1], dtype=np.float64)
        self.physics.angular_velocity = np.zeros(3, dtype=np.float64)
        
        # Сброс контроллера
        self.controller.altitude_pid.reset()
        self.controller.velocity_z_pid.reset()
        self.controller.roll_pid.reset()
        self.controller.roll_rate_pid.reset()
        self.controller.pitch_pid.reset()
        self.controller.pitch_rate_pid.reset()
        self.controller.yaw_pid.reset()
        self.controller.yaw_rate_pid.reset()
        
        # Начальная цель - земля
        self.controller.set_target_position([0, 0, 0.2])
        
        self.time = 0.0
        self.running = True
        self.flight_mode = 'takeoff'
        self.is_on_ground = True
        self.takeoff_start_time = 0.0
        
        print(f"Дрон на земле. Начинаем взлет на {self.target_height} метров...")
        
    def run_simulation(self, duration=60.0):
        """Запуск симуляции на определенное время"""
        self.initialize()
        
        print("Запуск симуляции дрона...")
        print("Цикл полета: Взлет → Зависание → Посадка")
        print(f"Высота зависания: {self.target_height} метров")
        print(f"Время зависания: {self.hover_duration} секунд")
        print(f"Максимальная скорость: {self.max_vertical_speed} м/с")
        print("Нажмите Ctrl+C для остановки")
        print("-" * 60)
        
        try:
            steps = int(duration / self.dt)
            
            for step in range(steps):
                if not self.running:
                    break
                    
                # Шаг симуляции
                try:
                    self.step()
                except Exception as e:
                    print(f"Ошибка на шаге {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.running = False
                    break
                
                # Обновление информации в консоли (каждые 100 шагов)
                if step % 100 == 0:
                    pos = self.physics.position
                    vel = np.linalg.norm(self.physics.velocity)
                    target_z = self.controller.target_position[2]
                    
                    # Определяем статус
                    if self.flight_mode == 'takeoff':
                        status = "ВЗЛЕТ"
                    elif self.flight_mode == 'hover':
                        remaining = self.hover_duration - (self.time - self.hover_start_time)
                        status = f"ЗАВИСАНИЕ (осталось: {remaining:.1f}с)"
                    elif self.flight_mode == 'landing':
                        status = "ПОСАДКА"
                    elif self.flight_mode == 'landed':
                        status = "ПРИЗЕМЛИЛСЯ"
                    else:
                        status = "НЕИЗВЕСТНО"
                    
                    print(f"[{status:30}] Время: {self.time:.1f}с, "
                          f"Высота: {pos[2]:.2f}м, "
                          f"Цель: {target_z:.2f}м, "
                          f"Скорость: {vel:.2f}м/с")
                    
        except KeyboardInterrupt:
            print("\nСимуляция остановлена пользователем")
            
        finally:
            self.stop()
            
    def step(self):
        """Один шаг симуляции"""
        if not self.running:
            return
            
        # Обновление режима полета
        self.update_flight_mode()
        
        # Получение состояния
        state = self.get_state()
        
        # Вычисление управления
        motor_speeds = self.controller.compute_control(state, self.dt)
        
        # Применение управления к физике
        self.physics.set_motor_speeds(motor_speeds)
        self.physics.update(self.dt)
        
        # Ограничение скорости
        self.limit_velocity()
        
        # Синхронизация с PyBullet
        if self.use_pybullet and self.environment:
            try:
                # Обновляем позицию дрона в PyBullet
                pos = self.physics.position
                orn = self.physics.orientation
                vel = self.physics.velocity
                ang_vel = self.physics.angular_velocity
                
                # Используем глобальные функции PyBullet
                import pybullet as p
                
                # Проверяем, что дрон существует
                if self.environment.drone_id is not None:
                    p.resetBasePositionAndOrientation(
                        self.environment.drone_id,
                        pos.tolist(),
                        orn.tolist()
                    )
                    
                    p.resetBaseVelocity(
                        self.environment.drone_id,
                        vel.tolist(),
                        ang_vel.tolist()
                    )
                
                # Шаг симуляции через среду
                self.environment.step()
                
                if self.gui:
                    time.sleep(self.dt * 0.5)
                    
            except Exception as e:
                print(f"Ошибка при синхронизации с PyBullet: {e}")
                self.use_pybullet = False
                
        # Обновление времени
        self.time += self.dt
        
        # Проверка статуса полета
        self.check_flight_status()
        
    def limit_velocity(self):
        """Ограничение скорости дрона"""
        # Ограничение вертикальной скорости
        if abs(self.physics.velocity[2]) > self.max_vertical_speed:
            self.physics.velocity[2] = np.sign(self.physics.velocity[2]) * self.max_vertical_speed
        
        # Ограничение горизонтальной скорости
        horizontal_speed = np.linalg.norm(self.physics.velocity[:2])
        if horizontal_speed > self.max_horizontal_speed:
            scale = self.max_horizontal_speed / horizontal_speed
            self.physics.velocity[0] *= scale
            self.physics.velocity[1] *= scale
        
        # Ограничение угловой скорости
        max_angular_speed = 5.0  # рад/с
        if np.linalg.norm(self.physics.angular_velocity) > max_angular_speed:
            self.physics.angular_velocity = self.physics.angular_velocity / np.linalg.norm(self.physics.angular_velocity) * max_angular_speed
        
    def update_flight_mode(self):
        """Обновление режима полета"""
        current_height = self.physics.position[2]
        
        if self.flight_mode == 'takeoff':
            # Устанавливаем цель - взлететь на target_height с ограничением скорости
            # Используем плавное изменение цели
            if current_height < self.target_height * 0.9:
                # Набираем высоту с ограничением скорости
                target_z = min(current_height + self.max_vertical_speed * self.dt * 20, self.target_height)
                self.controller.set_target_position([0, 0, target_z])
            else:
                # Переключаемся в режим зависания
                self.controller.set_target_position([0, 0, self.target_height])
                
                # Проверяем, достиг ли дрон целевой высоты
                if current_height >= self.target_height * 0.98:
                    self.flight_mode = 'hover'
                    self.hover_start_time = self.time
                    print(f"\nДрон достиг высоты {self.target_height:.1f}м! Начинаем зависание...")
                
        elif self.flight_mode == 'hover':
            # Удерживаем высоту
            self.controller.set_target_position([0, 0, self.target_height])
            
            # Проверяем, прошло ли время зависания
            if self.time - self.hover_start_time >= self.hover_duration:
                self.flight_mode = 'landing'
                self.landing_start_time = self.time
                print(f"\nНачинаем посадку...")
                
        elif self.flight_mode == 'landing':
            # Плавно снижаемся с ограничением скорости
            elapsed = self.time - self.landing_start_time
            
            # Вычисляем целевую высоту с учетом ограничения скорости
            descent_rate = 1.5  # скорость снижения (м/с) - меньше для плавности
            target_z = max(self.target_height - elapsed * descent_rate, 0.2)
            
            # Плавно уменьшаем цель
            self.controller.set_target_position([0, 0, target_z])
            
            # Проверяем, достигли ли земли
            if current_height < 0.5:
                self.flight_mode = 'landed'
                self.controller.set_target_position([0, 0, 0.2])
                print(f"\nДрон приземлился!")
                
        elif self.flight_mode == 'landed':
            # Дрон на земле
            self.controller.set_target_position([0, 0, 0.2])
            # Завершаем симуляцию через небольшую задержку
            if self.time - self.landing_start_time > 2.0:
                self.running = False
                print("\nМиссия завершена: взлет → зависание → посадка")
            
    def get_state(self):
        """Получение текущего состояния"""
        if self.use_pybullet and self.environment:
            return self.environment.get_state()
        else:
            return {
                'position': self.physics.position.copy(),
                'velocity': self.physics.velocity.copy(),
                'orientation': self.physics.orientation.copy(),
                'angular_velocity': self.physics.angular_velocity.copy()
            }
            
    def check_flight_status(self):
        """Проверка статуса полета"""
        pos = self.physics.position
        
        # Проверка на падение
        if pos[2] < 0.05 and self.flight_mode not in ['landed']:
            print(f"Дрон упал! Высота: {pos[2]:.2f}м")
            self.running = False
            
        # Проверка на уход слишком высоко
        if pos[2] > self.target_height + 5.0:
            print(f"Дрон поднялся слишком высоко: {pos[2]:.2f}м")
            self.running = False
            
    def stop(self):
        """Остановка симуляции"""
        self.running = False
        if self.use_pybullet and self.environment:
            try:
                self.environment.close()
            except:
                pass
        print("\nСимуляция завершена")
        
    def set_flight_params(self, target_height=15.0, hover_duration=5.0, max_speed=3.0):
        """Установка параметров полета"""
        self.target_height = target_height
        self.hover_duration = hover_duration
        self.max_vertical_speed = max_speed

# Основная функция для запуска
def main():
    try:
        # Создание симуляции
        sim = DroneSimulation(use_pybullet=True, gui=True)
        
        # Настройка параметров полета
        sim.set_flight_params(
            target_height=15.0,    # высота зависания 15 метров
            hover_duration=5.0,    # зависание на 5 секунд
            max_speed=3.0         # максимальная скорость 3 м/с
        )
        
        # Запуск симуляции на 60 секунд
        sim.run_simulation(duration=60.0)
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()