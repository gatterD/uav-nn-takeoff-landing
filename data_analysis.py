import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import csv
import os

class FlightDataAnalyzer:
    def __init__(self, log_directory="flight_logs"):
        """Инициализация анализатора"""
        self.log_directory = log_directory
        self.data = None
        
    def load_data(self, filepath):
        """Загрузка данных из файла"""
        if filepath.endswith('.csv'):
            self.load_csv(filepath)
        elif filepath.endswith('.json'):
            self.load_json(filepath)
        else:
            raise ValueError("Неподдерживаемый формат файла")
            
    def load_csv(self, filepath):
        """Загрузка данных из CSV"""
        data = []
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Преобразование строк в числа
                    for key, value in row.items():
                        try:
                            row[key] = float(value)
                        except ValueError:
                            pass
                    data.append(row)
            self.data = data
            print(f"Загружено {len(data)} записей из {filepath}")
            
        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            
    def load_json(self, filepath):
        """Загрузка данных из JSON"""
        try:
            with open(filepath, 'r') as jsonfile:
                json_data = json.load(jsonfile)
                
            if 'data' in json_data:
                self.data = json_data['data']
                print(f"Загружено {len(self.data)} записей из {filepath}")
            else:
                self.data = json_data
                print(f"Загружено {len(self.data)} записей из {filepath}")
                
        except Exception as e:
            print(f"Ошибка при загрузке JSON: {e}")
            
    def plot_position_over_time(self, save_path=None):
        """График позиции во времени"""
        if not self.data:
            print("Нет данных для отображения")
            return
            
        times = [d['time'] for d in self.data]
        pos_x = [d['position_x'] for d in self.data]
        pos_y = [d['position_y'] for d in self.data]
        pos_z = [d['position_z'] for d in self.data]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        axes[0].plot(times, pos_x, 'b-', label='X')
        axes[0].set_ylabel('Позиция X (м)')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(times, pos_y, 'g-', label='Y')
        axes[1].set_ylabel('Позиция Y (м)')
        axes[1].grid(True)
        axes[1].legend()
        
        axes[2].plot(times, pos_z, 'r-', label='Z')
        axes[2].set_xlabel('Время (с)')
        axes[2].set_ylabel('Позиция Z (м)')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.suptitle('Позиция дрона во времени')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        else:
            plt.show()
            
    def plot_velocity_over_time(self, save_path=None):
        """График скорости во времени"""
        if not self.data:
            print("Нет данных для отображения")
            return
            
        times = [d['time'] for d in self.data]
        vel_x = [d['velocity_x'] for d in self.data]
        vel_y = [d['velocity_y'] for d in self.data]
        vel_z = [d['velocity_z'] for d in self.data]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        axes[0].plot(times, vel_x, 'b-', label='Vx')
        axes[0].set_ylabel('Скорость X (м/с)')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(times, vel_y, 'g-', label='Vy')
        axes[1].set_ylabel('Скорость Y (м/с)')
        axes[1].grid(True)
        axes[1].legend()
        
        axes[2].plot(times, vel_z, 'r-', label='Vz')
        axes[2].set_xlabel('Время (с)')
        axes[2].set_ylabel('Скорость Z (м/с)')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.suptitle('Скорость дрона во времени')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        else:
            plt.show()
            
    def plot_orientation_over_time(self, save_path=None):
        """График ориентации во времени"""
        if not self.data:
            print("Нет данных для отображения")
            return
            
        times = [d['time'] for d in self.data]
        
        # Используем углы Эйлера, если они есть
        if 'roll' in self.data[0] and 'pitch' in self.data[0] and 'yaw' in self.data[0]:
            roll = [d['roll'] for d in self.data]
            pitch = [d['pitch'] for d in self.data]
            yaw = [d['yaw'] for d in self.data]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            
            axes[0].plot(times, roll, 'b-', label='Roll')
            axes[0].set_ylabel('Крен (рад)')
            axes[0].grid(True)
            axes[0].legend()
            
            axes[1].plot(times, pitch, 'g-', label='Pitch')
            axes[1].set_ylabel('Тангаж (рад)')
            axes[1].grid(True)
            axes[1].legend()
            
            axes[2].plot(times, yaw, 'r-', label='Yaw')
            axes[2].set_xlabel('Время (с)')
            axes[2].set_ylabel('Рыскание (рад)')
            axes[2].grid(True)
            axes[2].legend()
            
            plt.suptitle('Ориентация дрона во времени')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"График сохранен в {save_path}")
            else:
                plt.show()
        else:
            print("Данные об ориентации отсутствуют")
            
    def plot_3d_trajectory(self, save_path=None):
        """3D траектория полета"""
        if not self.data:
            print("Нет данных для отображения")
            return
            
        pos_x = [d['position_x'] for d in self.data]
        pos_y = [d['position_y'] for d in self.data]
        pos_z = [d['position_z'] for d in self.data]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Траектория
        ax.plot(pos_x, pos_y, pos_z, 'b-', linewidth=2, label='Траектория')
        
        # Начальная и конечная точки
        ax.scatter(pos_x[0], pos_y[0], pos_z[0], color='green', s=100, label='Старт')
        ax.scatter(pos_x[-1], pos_y[-1], pos_z[-1], color='red', s=100, label='Финиш')
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title('3D траектория полета дрона')
        ax.legend()
        
        # Установка равных масштабов
        max_range = max([
            max(pos_x) - min(pos_x),
            max(pos_y) - min(pos_y),
            max(pos_z) - min(pos_z)
        ]) / 2.0
        
        mid_x = (max(pos_x) + min(pos_x)) * 0.5
        mid_y = (max(pos_y) + min(pos_y)) * 0.5
        mid_z = (max(pos_z) + min(pos_z)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        else:
            plt.show()
            
    def plot_motor_speeds(self, save_path=None):
        """График скоростей моторов"""
        if not self.data:
            print("Нет данных для отображения")
            return
            
        times = [d['time'] for d in self.data]
        motor1 = [d['motor_1'] for d in self.data]
        motor2 = [d['motor_2'] for d in self.data]
        motor3 = [d['motor_3'] for d in self.data]
        motor4 = [d['motor_4'] for d in self.data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, motor1, 'r-', label='Мотор 1')
        plt.plot(times, motor2, 'g-', label='Мотор 2')
        plt.plot(times, motor3, 'b-', label='Мотор 3')
        plt.plot(times, motor4, 'm-', label='Мотор 4')
        
        plt.xlabel('Время (с)')
        plt.ylabel('Скорость мотора (0-1)')
        plt.title('Скорости моторов во времени')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        else:
            plt.show()
            
    def generate_all_plots(self, output_dir="plots"):
        """Генерация всех графиков"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        base_name = os.path.join(output_dir, f"flight_analysis_{self.data[0]['time']:.0f}")
        
        self.plot_position_over_time(f"{base_name}_position.png")
        self.plot_velocity_over_time(f"{base_name}_velocity.png")
        self.plot_orientation_over_time(f"{base_name}_orientation.png")
        self.plot_3d_trajectory(f"{base_name}_trajectory.png")
        self.plot_motor_speeds(f"{base_name}_motors.png")
        
        print(f"Все графики сохранены в директории {output_dir}")

# Пример использования
if __name__ == "__main__":
    # Создание анализатора
    analyzer = FlightDataAnalyzer("flight_logs")
    
    # Поиск последнего файла
    log_dir = "flight_logs"
    if os.path.exists(log_dir):
        files = [f for f in os.listdir(log_dir) if f.startswith("flight_")]
        if files:
            # Сортировка по времени создания
            files.sort(reverse=True)
            latest_file = os.path.join(log_dir, files[0])
            
            print(f"Загрузка последнего файла: {latest_file}")
            analyzer.load_data(latest_file)
            
            # Генерация всех графиков
            analyzer.generate_all_plots()
        else:
            print("Нет файлов с данными в директории")
    else:
        print("Директория с логами не найдена")