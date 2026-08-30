"""
Анализатор лога полёта дрона.
Поддерживает XLSX/CSV файлы, сформированные flight_data_logger.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    'time',
    'phase',
    'target_altitude',
    'altitude',
    'altitude_error',
    'vertical_velocity',
    'throttle',
    'x',
    'y',
    'z',
    'roll',
    'pitch',
    'yaw',
]


def load_data(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл '{file_path}' не найден.")

    if path.suffix.lower() in ['.xlsx', '.xls']:
        sheets = pd.read_excel(path, sheet_name=None)
        if not sheets:
            raise ValueError(f"Файл '{file_path}' не содержит листов.")
        return sheets

    return pd.read_csv(path)


def validate_columns(df, required_columns=REQUIRED_COLUMNS):
    missing = set(required_columns) - set(df.columns)
    return sorted(missing)


def convert_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_metrics(df):
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"В данных отсутствуют обязательные столбцы: {missing}")

    df = df.copy()
    df = convert_columns(df, REQUIRED_COLUMNS)

    if df['time'].isna().any():
        raise ValueError('Столбец time содержит некорректные значения.')

    if len(df) < 2:
        raise ValueError('Недостаточно строк данных для анализа.')

    metrics = {}
    metrics['rows'] = len(df)
    metrics['total_time'] = float(df['time'].iloc[-1] - df['time'].iloc[0])
    time_diff = df['time'].diff().dropna()
    metrics['avg_dt'] = float(time_diff.mean())

    metrics['initial_position'] = (float(df['x'].iloc[0]), float(df['y'].iloc[0]), float(df['z'].iloc[0]))
    metrics['final_position'] = (float(df['x'].iloc[-1]), float(df['y'].iloc[-1]), float(df['z'].iloc[-1]))
    metrics['delta_x'] = float(metrics['final_position'][0] - metrics['initial_position'][0])
    metrics['delta_y'] = float(metrics['final_position'][1] - metrics['initial_position'][1])
    metrics['delta_z'] = float(metrics['final_position'][2] - metrics['initial_position'][2])
    metrics['path_length'] = float(np.sqrt(metrics['delta_x']**2 + metrics['delta_y']**2 + metrics['delta_z']**2))

    metrics['max_altitude'] = float(df['altitude'].max())
    metrics['min_altitude'] = float(df['altitude'].min())
    metrics['mean_altitude'] = float(df['altitude'].mean())
    metrics['final_altitude'] = float(df['altitude'].iloc[-1])

    errors = df['altitude_error'].abs()
    metrics['mean_altitude_error'] = float(errors.mean())
    metrics['max_altitude_error'] = float(errors.max())
    metrics['error_within_0_1m'] = float((errors < 0.1).sum() / len(df) * 100)
    metrics['error_within_0_5m'] = float((errors < 0.5).sum() / len(df) * 100)

    metrics['mean_throttle'] = float(df['throttle'].mean())
    metrics['max_throttle'] = float(df['throttle'].max())
    metrics['min_throttle'] = float(df['throttle'].min())
    throttle = df['throttle'].to_numpy(dtype=float)
    time_vals = df['time'].to_numpy(dtype=float)
    if len(throttle) > 1:
        metrics['energy_proxy'] = float(np.sum((throttle[:-1] + throttle[1:]) * np.diff(time_vals) / 2.0))
    else:
        metrics['energy_proxy'] = float(throttle[0] if len(throttle) == 1 else 0.0)

    metrics['max_vertical_velocity'] = float(df['vertical_velocity'].max())
    metrics['min_vertical_velocity'] = float(df['vertical_velocity'].min())
    metrics['mean_vertical_velocity'] = float(df['vertical_velocity'].mean())
    metrics['std_vertical_velocity'] = float(df['vertical_velocity'].std())

    metrics['max_roll'] = float(np.degrees(df['roll'].abs().max()))
    metrics['max_pitch'] = float(np.degrees(df['pitch'].abs().max()))
    metrics['max_yaw'] = float(np.degrees(df['yaw'].abs().max()))
    metrics['mean_roll'] = float(np.degrees(df['roll'].abs().mean()))
    metrics['mean_pitch'] = float(np.degrees(df['pitch'].abs().mean()))
    metrics['mean_yaw'] = float(np.degrees(df['yaw'].abs().mean()))

    metrics['distance_3d'] = float(np.sqrt(
        np.diff(df['x'])**2 + np.diff(df['y'])**2 + np.diff(df['z'])**2
    ).sum())

    if 'phase' in df.columns and df['phase'].notna().any():
        takeoff_df = df[df['phase'].astype(str).str.lower() == 'takeoff']
        landing_df = df[df['phase'].astype(str).str.lower() == 'landing']
        if len(takeoff_df) >= 2:
            metrics['takeoff_duration'] = float(takeoff_df['time'].iloc[-1] - takeoff_df['time'].iloc[0])
            metrics['takeoff_height_gain'] = float(takeoff_df['altitude'].max() - takeoff_df['altitude'].iloc[0])
        else:
            metrics['takeoff_duration'] = None
            metrics['takeoff_height_gain'] = None

        if len(landing_df) >= 2:
            metrics['landing_duration'] = float(landing_df['time'].iloc[-1] - landing_df['time'].iloc[0])
            metrics['landing_height_loss'] = float(landing_df['altitude'].iloc[0] - landing_df['altitude'].iloc[-1])
        else:
            metrics['landing_duration'] = None
            metrics['landing_height_loss'] = None
    else:
        metrics['takeoff_duration'] = None
        metrics['takeoff_height_gain'] = None
        metrics['landing_duration'] = None
        metrics['landing_height_loss'] = None

    metrics['hover_time'] = float(((df['vertical_velocity'].abs() < 0.05) & (df['altitude'] > 0.5)).sum() * metrics['avg_dt'])

    return metrics


def format_value(value, fmt='{:.2f}'):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    return fmt.format(value)


def generate_report(metrics, title=None):
    header = f"=== Анализ: {title} ===" if title else "=== Анализ полёта ==="
    rows = [header, f"Строк данных: {metrics['rows']}", ""]
    rows += [
        f"Общее время:                       {format_value(metrics['total_time'], '{:.3f}')} с",
        f"Средний шаг:                       {format_value(metrics['avg_dt'], '{:.4f}')} с",
        "",
        "--- Высота ---",
        f"Макс. высота:                      {format_value(metrics['max_altitude'], '{:.3f}')} м",
        f"Мин. высота:                       {format_value(metrics['min_altitude'], '{:.3f}')} м",
        f"Фин. высота:                       {format_value(metrics['final_altitude'], '{:.3f}')} м",
        f"Сред. высота:                      {format_value(metrics['mean_altitude'], '{:.3f}')} м",
        "",
        "--- Отслеживание ---",
        f"Сред. ошибка высоты:               {format_value(metrics['mean_altitude_error'], '{:.3f}')} м",
        f"Макс. ошибка высоты:               {format_value(metrics['max_altitude_error'], '{:.3f}')} м",
        f"Время в ±0.1 м:                    {format_value(metrics['error_within_0_1m'], '{:.1f}')} %",
        f"Время в ±0.5 м:                    {format_value(metrics['error_within_0_5m'], '{:.1f}')} %",
        "",
        "--- Скорость ---",
        f"Макс. вертик. скорость:            {format_value(metrics['max_vertical_velocity'], '{:.3f}')} м/с",
        f"Мин. вертик. скорость:             {format_value(metrics['min_vertical_velocity'], '{:.3f}')} м/с",
        f"Сред. вертик. скорость:            {format_value(metrics['mean_vertical_velocity'], '{:.3f}')} м/с",
        f"Сигма вертик. скорости:            {format_value(metrics['std_vertical_velocity'], '{:.3f}')} м/с",
        "",
        "--- Тяга ---",
        f"Сред. throttle:                    {format_value(metrics['mean_throttle'], '{:.3f}')}",
        f"Макс. throttle:                    {format_value(metrics['max_throttle'], '{:.3f}')}",
        f"Мин. throttle:                    {format_value(metrics['min_throttle'], '{:.3f}')}",
        f"Интеграл throttle:                 {format_value(metrics['energy_proxy'], '{:.3f}')}",
        "",
        "--- Ориентация ---",
        f"Макс. roll:                        {format_value(metrics['max_roll'], '{:.1f}')}°",
        f"Макс. pitch:                       {format_value(metrics['max_pitch'], '{:.1f}')}°",
        f"Макс. yaw:                         {format_value(metrics['max_yaw'], '{:.1f}')}°",
        f"Сред. |roll|:                      {format_value(metrics['mean_roll'], '{:.2f}')}°",
        f"Сред. |pitch|:                     {format_value(metrics['mean_pitch'], '{:.2f}')}°",
        f"Сред. |yaw|:                       {format_value(metrics['mean_yaw'], '{:.2f}')}°",
        "",
        "--- Траектория ---",
        f"Перемещение X:                     {format_value(metrics['delta_x'], '{:.3f}')} м",
        f"Перемещение Y:                     {format_value(metrics['delta_y'], '{:.3f}')} м",
        f"Перемещение Z:                     {format_value(metrics['delta_z'], '{:.3f}')} м",
        f"Расстояние по прямой:              {format_value(metrics['path_length'], '{:.3f}')} м",
        f"Пройденное расстояние 3D:          {format_value(metrics['distance_3d'], '{:.3f}')} м",
    ]

    if metrics['takeoff_duration'] is not None or metrics['landing_duration'] is not None:
        rows += ["", "--- Фазы ---"]
        rows.append(f"Длительность взлета:               {format_value(metrics['takeoff_duration'], '{:.3f}')} с")
        rows.append(f"Набор высоты при взлете:           {format_value(metrics['takeoff_height_gain'], '{:.3f}')} м")
        rows.append(f"Длительность посадки:              {format_value(metrics['landing_duration'], '{:.3f}')} с")
        rows.append(f"Сброс высоты при посадке:          {format_value(metrics['landing_height_loss'], '{:.3f}')} м")

    rows += ["", "--- Дополнительно ---", f"Время висения:                    {format_value(metrics['hover_time'], '{:.3f}')} с"]
    rows.append("=" * 60)
    return '\n'.join(rows)


def save_report(report_text, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding='utf-8')
    print(f"Отчёт сохранён: {output_path}")


def plot_data(df, output_dir='flight_analysis_plots', title=None):
    output_path = Path(output_dir)
    if title:
        output_path = output_path / title
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['altitude'], label='altitude')
    plt.plot(df['time'], df['target_altitude'], label='target_altitude', linestyle='--')
    plt.xlabel('time, s')
    plt.ylabel('altitude, m')
    plt.title('Altitude vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / 'altitude.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['altitude_error'], label='altitude_error', color='red')
    plt.xlabel('time, s')
    plt.ylabel('error, m')
    plt.title('Altitude tracking error')
    plt.grid(True)
    plt.savefig(output_path / 'altitude_error.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['vertical_velocity'], label='vertical_velocity', color='green')
    plt.xlabel('time, s')
    plt.ylabel('vertical velocity, m/s')
    plt.title('Vertical velocity')
    plt.grid(True)
    plt.savefig(output_path / 'vertical_velocity.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df['time'], df['throttle'], label='throttle', color='orange')
    plt.xlabel('time, s')
    plt.ylabel('throttle')
    plt.title('Throttle signal')
    plt.grid(True)
    plt.savefig(output_path / 'throttle.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], c=df['z'], cmap='viridis', s=2)
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.title('Top-down trajectory')
    plt.colorbar(label='z, m')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(output_path / 'trajectory_top.png', dpi=150)
    plt.close()

    print(f"Графики сохранены в: {output_path}")


def analyze_file(file_path, report_path, output_dir, no_plot):
    data = load_data(file_path)
    sections = []
    if isinstance(data, dict):
        for sheet_name, df in data.items():
            metrics = compute_metrics(df)
            sections.append(generate_report(metrics, title=sheet_name))
            if not no_plot:
                plot_data(df, output_dir=output_dir, title=sheet_name)
    else:
        metrics = compute_metrics(data)
        sections.append(generate_report(metrics))
        if not no_plot:
            plot_data(data, output_dir=output_dir)

    report_text = '\n\n'.join(sections)
    save_report(report_text, report_path)


def main():
    parser = argparse.ArgumentParser(description='Анализ лога полёта дрона')
    parser.add_argument('file', nargs='?', default='flight_data.xlsx', help='Путь к XLSX или CSV файлу')
    parser.add_argument('--report', default='flight_analysis_report.txt', help='Путь к текстовому отчету')
    parser.add_argument('--output-dir', default='flight_analysis_plots', help='Папка для графиков')
    parser.add_argument('--no-plot', action='store_true', help='Отключить генерацию графиков')
    args = parser.parse_args()

    analyze_file(args.file, args.report, args.output_dir, args.no_plot)


if __name__ == '__main__':
    main()
