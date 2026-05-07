"""
Анализатор лога полёта дрона.
Принимает CSV-файл, сгенерированный flight_data_logger.py, и вычисляет
основные показатели: максимальная высота, время полёта, ошибки стабилизации,
затраты энергии и т.д. Строит графики для визуальной оценки.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(file_path):
    """Загрузка данных из CSV с обработкой возможных ошибок."""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                         'roll', 'pitch', 'yaw', 'wx', 'wy', 'wz',
                         'throttle', 'target_roll', 'target_pitch', 'target_yaw_rate']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"В файле отсутствуют столбцы: {missing}")
        return df
    except FileNotFoundError:
        raise SystemExit(f"Файл {file_path} не найден.")
    except Exception as e:
        raise SystemExit(f"Ошибка при чтении данных: {e}")


def compute_metrics(df):
    """Расчёт основных метрик полёта."""
    metrics = {}

    # Время
    metrics['total_time'] = df['time'].iloc[-1] - df['time'].iloc[0]
    # Частота дискретизации (средняя)
    dt = np.diff(df['time']).mean() if len(df) > 1 else 0
    metrics['avg_dt'] = dt

    # Начальные и конечные позиции
    metrics['initial_x'] = df['x'].iloc[0]
    metrics['initial_y'] = df['y'].iloc[0]
    metrics['initial_z'] = df['z'].iloc[0]
    metrics['final_x'] = df['x'].iloc[-1]
    metrics['final_y'] = df['y'].iloc[-1]
    metrics['final_z'] = df['z'].iloc[-1]
    
    # Перемещение
    metrics['delta_x'] = metrics['final_x'] - metrics['initial_x']
    metrics['delta_y'] = metrics['final_y'] - metrics['initial_y']
    metrics['delta_z'] = metrics['final_z'] - metrics['initial_z']
    metrics['total_displacement'] = np.sqrt(metrics['delta_x']**2 + 
                                             metrics['delta_y']**2 + 
                                             metrics['delta_z']**2)

    # Высота
    z = df['z'].values
    metrics['max_altitude'] = np.max(z)
    idx_max_z = np.argmax(z)
    metrics['time_max_altitude'] = df['time'].iloc[idx_max_z]
    metrics['min_altitude'] = np.min(z)

    # Определение взлёта и посадки (по пересечению порога 0.1 м)
    threshold = 0.1
    above = z > threshold
    takeoff_idx = np.argmax(above) if np.any(above) else 0
    metrics['takeoff_time'] = df['time'].iloc[takeoff_idx] if takeoff_idx > 0 else 0.0
    # Момент, когда z последний раз опускается ниже порога после максимальной высоты
    landing_candidates = np.where((np.diff(above.astype(int)) == -1) & (np.arange(len(z)-1) >= idx_max_z))[0]
    if len(landing_candidates) > 0:
        landing_idx = landing_candidates[0] + 1  # следующий отсчёт после падения ниже порога
    else:
        landing_idx = len(z) - 1
    metrics['landing_time'] = df['time'].iloc[landing_idx] if landing_idx < len(z) else df['time'].iloc[-1]

    # Скорости
    metrics['max_speed_x'] = np.max(np.abs(df['vx']))
    metrics['max_speed_y'] = np.max(np.abs(df['vy']))
    metrics['max_speed_z'] = np.max(np.abs(df['vz']))
    metrics['avg_abs_speed_x'] = np.mean(np.abs(df['vx']))
    metrics['avg_abs_speed_y'] = np.mean(np.abs(df['vy']))
    metrics['avg_abs_speed_z'] = np.mean(np.abs(df['vz']))

    # Углы стабилизации
    metrics['max_roll'] = np.max(np.abs(df['roll']))
    metrics['max_pitch'] = np.max(np.abs(df['pitch']))
    metrics['max_yaw'] = np.max(np.abs(df['yaw']))
    metrics['avg_abs_roll'] = np.mean(np.abs(df['roll']))
    metrics['avg_abs_pitch'] = np.mean(np.abs(df['pitch']))
    metrics['avg_abs_yaw'] = np.mean(np.abs(df['yaw']))

    # Ошибки отслеживания целевых углов
    metrics['rms_roll_error'] = np.sqrt(np.mean((df['roll'] - df['target_roll'])**2))
    metrics['rms_pitch_error'] = np.sqrt(np.mean((df['pitch'] - df['target_pitch'])**2))
    metrics['rms_yaw_rate_error'] = np.sqrt(np.mean((df['yaw'] - df['target_yaw_rate'])**2))

    # Угловые скорости
    metrics['max_wx'] = np.max(np.abs(df['wx']))
    metrics['max_wy'] = np.max(np.abs(df['wy']))
    metrics['max_wz'] = np.max(np.abs(df['wz']))

    # Энергозатраты (приблизительно: интеграл throttle * dt)
    try:
        metrics['integrated_throttle'] = np.trapezoid(df['throttle'], df['time'])
    except AttributeError:
        metrics['integrated_throttle'] = np.trapz(df['throttle'], df['time'])
    
    metrics['avg_throttle'] = np.mean(df['throttle'])

    # Время висения (|vz| < 0.05 м/с и высота > 0.5 м)
    hovering = (np.abs(df['vz']) < 0.05) & (df['z'] > 0.5)
    metrics['hovering_duration'] = np.sum(hovering) * dt

    # Пройденный путь (приближённо)
    dist = np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2 + np.diff(df['z'])**2)
    metrics['total_distance'] = np.sum(dist)

    return metrics


def print_report(metrics):
    """Вывод отчёта в консоль и сохранение в файл."""
    report = [
        "================ РЕЗУЛЬТАТЫ АНАЛИЗА ПОЛЁТА ================",
        f"Общее время симуляции:              {metrics['total_time']:.2f} с",
        f"Средний шаг по времени:              {metrics['avg_dt']:.4f} с",
        "",
        "--- Начальная и конечная позиция ---",
        f"Начальная позиция X:                 {metrics['initial_x']:.6f} м",
        f"Начальная позиция Y:                 {metrics['initial_y']:.6f} м",
        f"Начальная позиция Z:                 {metrics['initial_z']:.6f} м",
        f"Конечная позиция X:                  {metrics['final_x']:.6f} м",
        f"Конечная позиция Y:                  {metrics['final_y']:.6f} м",
        f"Конечная позиция Z:                  {metrics['final_z']:.6f} м",
        f"Перемещение ΔX:                      {metrics['delta_x']:.6f} м",
        f"Перемещение ΔY:                      {metrics['delta_y']:.6f} м",
        f"Перемещение ΔZ:                      {metrics['delta_z']:.6f} м",
        f"Общее перемещение:                   {metrics['total_displacement']:.6f} м",
        "",
        "--- Высота ---",
        f"Максимальная высота:                 {metrics['max_altitude']:.3f} м",
        f"Время достижения макс. высоты:       {metrics['time_max_altitude']:.3f} с",
        f"Минимальная высота:                  {metrics['min_altitude']:.3f} м",
        f"Время взлёта (z > 0.1 м):            {metrics['takeoff_time']:.3f} с",
        f"Время посадки (z < 0.1 м):           {metrics['landing_time']:.3f} с",
        f"Пройденный путь:                     {metrics['total_distance']:.3f} м",
        "",
        "--- Скорости ---",
        f"Макс. |vx|:                           {metrics['max_speed_x']:.3f} м/с",
        f"Макс. |vy|:                           {metrics['max_speed_y']:.3f} м/с",
        f"Макс. |vz|:                           {metrics['max_speed_z']:.3f} м/с",
        f"Сред. |vx|:                           {metrics['avg_abs_speed_x']:.3f} м/с",
        f"Сред. |vy|:                           {metrics['avg_abs_speed_y']:.3f} м/с",
        f"Сред. |vz|:                           {metrics['avg_abs_speed_z']:.3f} м/с",
        "",
        "--- Углы ориентации ---",
        f"Макс. |roll|:                         {np.degrees(metrics['max_roll']):.1f}°",
        f"Макс. |pitch|:                        {np.degrees(metrics['max_pitch']):.1f}°",
        f"Макс. |yaw|:                          {np.degrees(metrics['max_yaw']):.1f}°",
        f"Сред. |roll|:                         {np.degrees(metrics['avg_abs_roll']):.2f}°",
        f"Сред. |pitch|:                        {np.degrees(metrics['avg_abs_pitch']):.2f}°",
        f"Сред. |yaw|:                          {np.degrees(metrics['avg_abs_yaw']):.2f}°",
        f"СКО ошибки roll:                      {np.degrees(metrics['rms_roll_error']):.2f}°",
        f"СКО ошибки pitch:                     {np.degrees(metrics['rms_pitch_error']):.2f}°",
        f"СКО ошибки yaw rate:                  {metrics['rms_yaw_rate_error']:.4f} рад/с",
        "",
        "--- Угловые скорости ---",
        f"Макс. |wx|:                           {metrics['max_wx']:.3f} рад/с",
        f"Макс. |wy|:                           {metrics['max_wy']:.3f} рад/с",
        f"Макс. |wz|:                           {metrics['max_wz']:.3f} рад/с",
        "",
        "--- Энергия ---",
        f"Интеграл throttle (пропорц. энергии): {metrics['integrated_throttle']:.2f}",
        f"Средний throttle:                     {metrics['avg_throttle']:.3f}",
        f"Время висения (|vz|<0.05 и z>0.5):    {metrics['hovering_duration']:.2f} с",
        "============================================================="
    ]
    report_str = '\n'.join(report)
    print(report_str)
    with open('flight_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_str)
    print("\nОтчёт сохранён в 'flight_analysis_report.txt'")


def plot_data(df, output_dir='.'):
    """Построение и сохранение графиков."""
    Path(output_dir).mkdir(exist_ok=True)

    # Высота от времени
    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['z'], label='z (высота)')
    plt.xlabel('Время, с')
    plt.ylabel('z, м')
    plt.title('Высота полёта')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/altitude.png', dpi=150)
    plt.close()

    # Скорости
    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['vx'], label='vx')
    plt.plot(df['time'], df['vy'], label='vy')
    plt.plot(df['time'], df['vz'], label='vz')
    plt.xlabel('Время, с')
    plt.ylabel('Скорость, м/с')
    plt.title('Скорости по осям')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/velocities.png', dpi=150)
    plt.close()

    # Углы ориентации
    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], np.degrees(df['roll']), label='roll')
    plt.plot(df['time'], np.degrees(df['pitch']), label='pitch')
    plt.plot(df['time'], np.degrees(df['yaw']), label='yaw')
    plt.xlabel('Время, с')
    plt.ylabel('Угол, град')
    plt.title('Углы ориентации')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/attitude.png', dpi=150)
    plt.close()

    # Угловые скорости
    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['wx'], label='wx')
    plt.plot(df['time'], df['wy'], label='wy')
    plt.plot(df['time'], df['wz'], label='wz')
    plt.xlabel('Время, с')
    plt.ylabel('Угловая скорость, рад/с')
    plt.title('Угловые скорости')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/angular_rates.png', dpi=150)
    plt.close()

    # Тяга (throttle)
    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['throttle'], label='throttle')
    plt.xlabel('Время, с')
    plt.ylabel('Throttle (0-1)')
    plt.title('Управляющий сигнал тяги')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/throttle.png', dpi=150)
    plt.close()

    # Траектория (вид сверху + цвет по высоте)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df['x'], df['y'], c=df['z'], cmap='viridis', s=1, alpha=0.7)
    plt.colorbar(sc, label='Высота, м')
    plt.xlabel('X, м')
    plt.ylabel('Y, м')
    plt.title('Траектория полёта (вид сверху)')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'{output_dir}/trajectory_top.png', dpi=150)
    plt.close()

    # 3D-траектория (если установлен mpl_toolkits)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df['x'], df['y'], df['z'], linewidth=0.5)
        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')
        ax.set_zlabel('Z, м')
        ax.set_title('3D траектория полёта')
        plt.savefig(f'{output_dir}/trajectory_3d.png', dpi=150)
        plt.close()
    except ImportError:
        pass

    print(f"Графики сохранены в папку '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description='Анализ лога полёта дрона')
    parser.add_argument('csv_file', nargs='?', default='flight_data.csv',
                        help='Путь к CSV файлу с логом полёта')
    parser.add_argument('--no-plot', action='store_true',
                        help='Отключить построение графиков')
    parser.add_argument('--output-dir', default='flight_analysis_plots',
                        help='Папка для сохранения графиков (по умолчанию flight_analysis_plots)')
    args = parser.parse_args()

    df = load_data(args.csv_file)
    metrics = compute_metrics(df)
    print_report(metrics)

    if not args.no_plot:
        plot_data(df, args.output_dir)


if __name__ == '__main__':
    main()