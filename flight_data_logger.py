import csv
import json
import numpy as np
from datetime import datetime
import os

class FlightDataLogger:
    def __init__(self, log_directory="flight_logs"):
        """Инициализация логгера"""
        self.log_directory = log_directory
        self.data = []
        self.headers = []
        self.flight_start_time = None
        
        # Создание директории для логов
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            
        # Создание имени файла с временной меткой
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"flight_{self.timestamp}"
        
    def start_logging(self):
        """Начало логирования"""
        self.flight_start_time = datetime.now()
        self.data = []
        self.headers = [
            'time', 'step',
            'position_x', 'position_y', 'position_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
            'motor_1', 'motor_2', 'motor_3', 'motor_4',
            'target_x', 'target_y', 'target_z',
            'roll', 'pitch', 'yaw',
            'altitude_error', 'total_thrust'
        ]
        
    def log_state(self, state, motor_speeds, target_position, dt):
        """Логирование состояния"""
        if self.flight_start_time is None:
            self.start_logging()
            
        # Вычисление углов Эйлера
        roll, pitch, yaw = self.quaternion_to_euler(state['orientation'])
        
        # Данные для записи
        entry = {
            'time': (datetime.now() - self.flight_start_time).total_seconds(),
            'step': len(self.data),
            'position_x': state['position'][0],
            'position_y': state['position'][1],
            'position_z': state['position'][2],
            'velocity_x': state['velocity'][0],
            'velocity_y': state['velocity'][1],
            'velocity_z': state['velocity'][2],
            'orientation_x': state['orientation'][0],
            'orientation_y': state['orientation'][1],
            'orientation_z': state['orientation'][2],
            'orientation_w': state['orientation'][3],
            'angular_velocity_x': state['angular_velocity'][0],
            'angular_velocity_y': state['angular_velocity'][1],
            'angular_velocity_z': state['angular_velocity'][2],
            'motor_1': motor_speeds[0],
            'motor_2': motor_speeds[1],
            'motor_3': motor_speeds[2],
            'motor_4': motor_speeds[3],
            'target_x': target_position[0],
            'target_y': target_position[1],
            'target_z': target_position[2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'altitude_error': target_position[2] - state['position'][2],
            'total_thrust': np.sum(motor_speeds) * 10.0  # примерная тяга
        }
        
        self.data.append(entry)
        
    def quaternion_to_euler(self, quaternion):
        """Преобразование кватерниона в углы Эйлера"""
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
        
    def save_to_csv(self):
        """Сохранение данных в CSV файл"""
        if not self.data:
            print("Нет данных для сохранения")
            return
            
        csv_path = os.path.join(self.log_directory, f"{self.filename}.csv")
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writeheader()
                
                for entry in self.data:
                    # Убедимся, что все ключи присутствуют
                    row = {key: entry.get(key, None) for key in self.headers}
                    writer.writerow(row)
                    
            print(f"Данные сохранены в {csv_path}")
            
        except Exception as e:
            print(f"Ошибка при сохранении CSV: {e}")
            
    def save_to_json(self):
        """Сохранение данных в JSON файл"""
        if not self.data:
            print("Нет данных для сохранения")
            return
            
        json_path = os.path.join(self.log_directory, f"{self.filename}.json")
        
        try:
            # Подготовка данных для JSON
            json_data = {
                'metadata': {
                    'start_time': self.flight_start_time.isoformat() if self.flight_start_time else None,
                    'total_steps': len(self.data),
                    'duration': self.data[-1]['time'] - self.data[0]['time'] if len(self.data) > 1 else 0
                },
                'data': self.data
            }
            
            with open(json_path, 'w') as jsonfile:
                json.dump(json_data, jsonfile, indent=2, default=str)
                
            print(f"Данные сохранены в {json_path}")
            
        except Exception as e:
            print(f"Ошибка при сохранении JSON: {e}")
            
    def get_data_summary(self):
        """Получение сводки данных"""
        if not self.data:
            return "Нет данных для анализа"
            
        try:
            positions = np.array([[d['position_x'], d['position_y'], d['position_z']] for d in self.data])
            velocities = np.array([[d['velocity_x'], d['velocity_y'], d['velocity_z']] for d in self.data])
            motor_speeds = np.array([[d['motor_1'], d['motor_2'], d['motor_3'], d['motor_4']] for d in self.data])
            
            summary = {
                'total_steps': len(self.data),
                'duration': self.data[-1]['time'] - self.data[0]['time'],
                'max_position_z': np.max(positions[:, 2]),
                'min_position_z': np.min(positions[:, 2]),
                'max_velocity': np.max(np.linalg.norm(velocities, axis=1)),
                'max_motor_speed': np.max(motor_speeds),
                'avg_motor_speed': np.mean(motor_speeds)
            }
            
            return summary
            
        except Exception as e:
            return f"Ошибка при анализе данных: {e}"
            
    def save_summary(self):
        """Сохранение сводки в файл"""
        summary = self.get_data_summary()
        
        if isinstance(summary, dict):
            summary_path = os.path.join(self.log_directory, f"{self.filename}_summary.json")
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            print(f"Сводка сохранена в {summary_path}")
            
            # Вывод сводки в консоль
            print("\n=== Сводка полета ===")
            for key, value in summary.items():
                print(f"{key}: {value}")
        else:
            print(summary)