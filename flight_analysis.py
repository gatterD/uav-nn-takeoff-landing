import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================

def load_flight_data(path="flight_data.xlsx"):
    """Загружает XLSX/CSV. Для Excel возвращается словарь листов."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл '{path}' не найден.")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(file_path, sheet_name=None)
        if not sheets:
            raise ValueError(f"Файл '{path}' не содержит листов.")
        return sheets

    return {"single_run": pd.read_csv(file_path)}


# =============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================

def _as_numeric(series, field_name=None):
    values = pd.to_numeric(series, errors="coerce")
    if field_name and values.isna().all():
        return pd.Series(np.nan, index=series.index)
    return values


def _scenario_info(name):
    base_name = str(name)
    if "_" in base_name:
        parts = base_name.split("_")
        if len(parts) >= 3:
            env = "_".join(parts[:-1])
            difficulty = parts[-1]
            return env, difficulty
    return base_name, "single"


# =============================================================
# АНАЛИЗ СТАТИСТИКИ
# =============================================================

def analyze_flight_statistics(df):
    df = df.copy()
    df["time"] = _as_numeric(df.get("time", pd.Series(np.nan, index=df.index)), "time")
    df["altitude"] = _as_numeric(df.get("altitude", df.get("z", pd.Series(np.nan, index=df.index))), "altitude")
    df["vertical_velocity"] = _as_numeric(df.get("vertical_velocity", df.get("vz", pd.Series(np.nan, index=df.index))), "vertical_velocity")
    df["throttle"] = _as_numeric(df.get("throttle", pd.Series(np.nan, index=df.index)), "throttle")
    df["altitude_error"] = _as_numeric(df.get("altitude_error", pd.Series(np.nan, index=df.index)), "altitude_error")

    stats_dict = {
        "total_time": float(df["time"].iloc[-1] - df["time"].iloc[0]) if len(df) > 1 else 0.0,
        "max_altitude": float(df["altitude"].max()) if df["altitude"].notna().any() else 0.0,
        "min_altitude": float(df["altitude"].min()) if df["altitude"].notna().any() else 0.0,
        "avg_altitude": float(df["altitude"].mean()) if df["altitude"].notna().any() else 0.0,
        "std_altitude": float(df["altitude"].std()) if df["altitude"].notna().any() else 0.0,
        "max_velocity": float(df["vertical_velocity"].max()) if df["vertical_velocity"].notna().any() else 0.0,
        "min_velocity": float(df["vertical_velocity"].min()) if df["vertical_velocity"].notna().any() else 0.0,
        "avg_velocity": float(df["vertical_velocity"].mean()) if df["vertical_velocity"].notna().any() else 0.0,
        "max_throttle": float(df["throttle"].max()) if df["throttle"].notna().any() else 0.0,
        "min_throttle": float(df["throttle"].min()) if df["throttle"].notna().any() else 0.0,
        "avg_throttle": float(df["throttle"].mean()) if df["throttle"].notna().any() else 0.0,
        "final_altitude": float(df["altitude"].iloc[-1]) if df["altitude"].notna().any() else 0.0,
        "altitude_error_final": float(df["altitude_error"].iloc[-1]) if df["altitude_error"].notna().any() else 0.0,
    }
    return stats_dict


# =============================================================
# АНАЛИЗ ПО ФАЗАМ
# =============================================================

def analyze_phases(df):
    df = df.copy()
    if "phase" in df.columns:
        phase_series = df["phase"].astype(str).str.lower()
        takeoff_data = df[phase_series == "takeoff"]
        landing_data = df[phase_series == "landing"]
    else:
        takeoff_data = df.iloc[: max(1, len(df) // 5)]
        landing_data = df.iloc[-max(1, len(df) // 5):]

    takeoff_duration = 0.0
    takeoff_height_gain = 0.0
    if len(takeoff_data) > 1:
        takeoff_duration = float(takeoff_data["time"].iloc[-1] - takeoff_data["time"].iloc[0])
        takeoff_height_gain = float(takeoff_data["altitude"].max() - takeoff_data["altitude"].iloc[0])

    landing_duration = 0.0
    landing_height_loss = 0.0
    if len(landing_data) > 1:
        landing_duration = float(landing_data["time"].iloc[-1] - landing_data["time"].iloc[0])
        landing_height_loss = float(landing_data["altitude"].iloc[0] - landing_data["altitude"].iloc[-1])

    phases_analysis = {
        "takeoff": {
            "duration": takeoff_duration,
            "height_gained": takeoff_height_gain,
            "avg_throttle": float(takeoff_data["throttle"].mean()) if len(takeoff_data) else 0.0,
            "altitude_at_end": float(takeoff_data["altitude"].iloc[-1]) if len(takeoff_data) else 0.0,
            "avg_altitude_error": float(takeoff_data["altitude_error"].abs().mean()) if len(takeoff_data) else 0.0,
        },
        "landing": {
            "duration": landing_duration,
            "height_lost": landing_height_loss,
            "avg_throttle": float(landing_data["throttle"].mean()) if len(landing_data) else 0.0,
            "final_altitude": float(landing_data["altitude"].iloc[-1]) if len(landing_data) else 0.0,
            "final_velocity": float(landing_data["vertical_velocity"].iloc[-1]) if len(landing_data) else 0.0,
            "avg_altitude_error": float(landing_data["altitude_error"].abs().mean()) if len(landing_data) else 0.0,
        },
    }
    return phases_analysis


# =============================================================
# СТАБИЛЬНОСТЬ
# =============================================================

def analyze_stability(df):
    df = df.copy()
    altitude_error = df["altitude_error"].astype(float)
    vertical_velocity = df["vertical_velocity"].astype(float)
    throttle = df["throttle"].astype(float)

    altitude_stability = 1.0 / (1.0 + altitude_error.std()) if not altitude_error.empty else 0.0
    velocity_stability = 1.0 / (1.0 + vertical_velocity.std()) if not vertical_velocity.empty else 0.0
    throttle_changes = throttle.diff().abs().mean() if len(throttle) > 1 else 0.0
    throttle_smoothness = 1.0 / (1.0 + throttle_changes)
    altitude_oscillations = int((altitude_error.abs() > 0.5).sum())

    return {
        "altitude_stability": float(altitude_stability),
        "velocity_stability": float(velocity_stability),
        "throttle_smoothness": float(throttle_smoothness),
        "altitude_oscillations": altitude_oscillations,
        "overall_stability": float((altitude_stability + velocity_stability + throttle_smoothness) / 3),
    }


# =============================================================
# АНАЛИЗ ОШИБОК
# =============================================================

def analyze_altitude_tracking(df):
    df = df.copy()
    if "altitude_error" in df.columns:
        altitude_error = df["altitude_error"].astype(float).abs()
    elif "altitude" in df.columns and "target_altitude" in df.columns:
        altitude_error = (df["altitude"] - df["target_altitude"]).abs()
    else:
        altitude_error = pd.Series(np.zeros(len(df)), index=df.index)

    return {
        "mae": float(altitude_error.mean()) if not altitude_error.empty else 0.0,
        "rmse": float(np.sqrt((altitude_error ** 2).mean())) if not altitude_error.empty else 0.0,
        "max_error": float(altitude_error.max()) if not altitude_error.empty else 0.0,
        "min_error": float(altitude_error.min()) if not altitude_error.empty else 0.0,
        "error_within_0_5m": float((altitude_error < 0.5).sum() / len(df) * 100) if len(df) else 0.0,
        "error_within_0_1m": float((altitude_error < 0.1).sum() / len(df) * 100) if len(df) else 0.0,
    }


# =============================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================

def plot_flight_data(df, save_path="flight_analysis_plots/", scenario_name="flight"):
    os.makedirs(save_path, exist_ok=True)
    base_name = scenario_name.replace(" ", "_")

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(df["time"], df["altitude"], label="Высота", linewidth=2)
    if "target_altitude" in df.columns:
        plt.axhline(y=df["target_altitude"].iloc[0], color="r", linestyle="--", label="Целевая (начало)")
        plt.axhline(y=df["target_altitude"].iloc[-1], color="g", linestyle="--", label="Целевая (конец)")
    plt.xlabel("Время (с)")
    plt.ylabel("Высота (м)")
    plt.title(f"Траектория высоты — {scenario_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(df["time"], df["throttle"], label="Дроссель", linewidth=1.5, color="orange")
    plt.xlabel("Время (с)")
    plt.ylabel("Дроссель")
    plt.title("Сигнал дросселя")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(df["time"], df["altitude_error"], label="Ошибка высоты", linewidth=1.5, color="red")
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    plt.fill_between(df["time"], df["altitude_error"], 0, alpha=0.3)
    plt.xlabel("Время (с)")
    plt.ylabel("Ошибка (м)")
    plt.title("Ошибка отслеживания высоты")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}{base_name}_main_analysis.png", dpi=100)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["x"], df["y"], linewidth=2, color="purple")
    plt.scatter(df["x"].iloc[0], df["y"].iloc[0], color="green", s=100, marker="o", label="Начало")
    plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="red", s=100, marker="x", label="Конец")
    plt.xlabel("X (м)")
    plt.ylabel("Y (м)")
    plt.title(f"Горизонтальная траектория — {scenario_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.plot(df["time"], df["z"], label="Z координата", linewidth=2, color="blue")
    plt.xlabel("Время (с)")
    plt.ylabel("Z (м)")
    plt.title("Вертикальная координата")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}{base_name}_trajectory_analysis.png", dpi=100)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["time"], df["vertical_velocity"], label="Вертикальная скорость", linewidth=1.5)
    plt.xlabel("Время (с)")
    plt.ylabel("Скорость (м/с)")
    plt.title("Вертикальная скорость")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    acceleration = df["vertical_velocity"].diff() / df["time"].diff()
    plt.plot(df["time"], acceleration, label="Вертикальное ускорение", linewidth=1.5, color="green")
    plt.xlabel("Время (с)")
    plt.ylabel("Ускорение (м/с²)")
    plt.title("Вертикальное ускорение")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}{base_name}_velocity_analysis.png", dpi=100)
    plt.close()


# =============================================================
# РАЗБОР СЦЕНАРИЕВ
# =============================================================

def summarize_run(df, scenario_name):
    flight_stats = analyze_flight_statistics(df)
    phases = analyze_phases(df)
    stability = analyze_stability(df)
    tracking = analyze_altitude_tracking(df)
    env_name, difficulty = _scenario_info(scenario_name)

    summary = {
        "name": scenario_name,
        "environment": env_name,
        "difficulty": difficulty,
        "flight_stats": flight_stats,
        "phases": phases,
        "stability": stability,
        "tracking": tracking,
    }
    return summary


def _format_summary_line(label, value, unit=""):
    return f"  {label:<32}{value:.3f}{unit}"


# =============================================================
# ПОЛНЫЙ ОТЧЕТ
# =============================================================

def generate_flight_report(csv_path="flight_data.xlsx", save_path="flight_analysis_plots/"):
    """Генерирует подробный текстовый отчет по каждому сценарию и сравнение между ними."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    loaded_data = load_flight_data(csv_path)
    scenario_summaries = []

    if isinstance(loaded_data, dict):
        for scenario_name, df in loaded_data.items():
            scenario_summaries.append((scenario_name, summarize_run(df, scenario_name)))
    else:
        scenario_summaries.append(("single_run", summarize_run(loaded_data, "single_run")))

    environment_groups = {}
    for scenario_name, summary in scenario_summaries:
        env = summary["environment"]
        environment_groups.setdefault(env, []).append(summary)

    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("ДЕТАЛЬНЫЙ АНАЛИЗ ПОЛЕТОВ ПО СЦЕНАРИЯМ")
    report_lines.append("=" * 90)
    report_lines.append("")

    for scenario_name, summary in scenario_summaries:
        stats = summary["flight_stats"]
        phases = summary["phases"]
        stability = summary["stability"]
        tracking = summary["tracking"]
        env_name, difficulty = summary["environment"], summary["difficulty"]

        report_lines.append(f"=== СЦЕНАРИЙ: {scenario_name} ===")
        report_lines.append(f"Среда: {env_name}")
        report_lines.append(f"Уровень сложности: {difficulty}")
        report_lines.append(f"Общее время: {stats['total_time']:.3f} c")
        report_lines.append(f"Макс. высота: {stats['max_altitude']:.3f} м")
        report_lines.append(f"Мин. высота: {stats['min_altitude']:.3f} м")
        report_lines.append(f"Средняя высота: {stats['avg_altitude']:.3f} м")
        report_lines.append(f"Финальная высота: {stats['final_altitude']:.3f} м")
        report_lines.append(f"Средняя ошибка высоты: {tracking['mae']:.3f} м")
        report_lines.append(f"RMSE: {tracking['rmse']:.3f} м")
        report_lines.append(f"Макс. ошибка: {tracking['max_error']:.3f} м")
        report_lines.append(f"Время в пределах ±0.5 м: {tracking['error_within_0_5m']:.2f}%")
        report_lines.append(f"Время в пределах ±0.1 м: {tracking['error_within_0_1m']:.2f}%")
        report_lines.append(f"Макс. вертикальная скорость: {stats['max_velocity']:.3f} м/с")
        report_lines.append(f"Средняя вертикальная скорость: {stats['avg_velocity']:.3f} м/с")
        report_lines.append(f"Средний throttle: {stats['avg_throttle']:.3f}")
        report_lines.append(f"Макс. throttle: {stats['max_throttle']:.3f}")
        report_lines.append(f"Общая стабильность: {stability['overall_stability']:.3f}")
        report_lines.append(f"Стабильность по высоте: {stability['altitude_stability']:.3f}")
        report_lines.append(f"Стабильность по скорости: {stability['velocity_stability']:.3f}")
        report_lines.append(f"Плавность дросселя: {stability['throttle_smoothness']:.3f}")
        report_lines.append(f"Колебания высоты: {stability['altitude_oscillations']}")

        report_lines.append("  Фаза взлета:")
        report_lines.append(f"    Длительность: {phases['takeoff']['duration']:.3f} c")
        report_lines.append(f"    Набор высоты: {phases['takeoff']['height_gained']:.3f} м")
        report_lines.append(f"    Средний throttle: {phases['takeoff']['avg_throttle']:.3f}")
        report_lines.append(f"    Средняя ошибка: {phases['takeoff']['avg_altitude_error']:.3f} м")

        report_lines.append("  Фаза посадки:")
        report_lines.append(f"    Длительность: {phases['landing']['duration']:.3f} c")
        report_lines.append(f"    Снижение высоты: {phases['landing']['height_lost']:.3f} м")
        report_lines.append(f"    Final altitude: {phases['landing']['final_altitude']:.3f} м")
        report_lines.append(f"    Финальная вертикальная скорость: {phases['landing']['final_velocity']:.3f} м/с")
        report_lines.append(f"    Средняя ошибка: {phases['landing']['avg_altitude_error']:.3f} м")

        if tracking['error_within_0_5m'] > 80:
            verdict = "Контроллер удерживает высоту хорошо"
        elif tracking['error_within_0_5m'] > 60:
            verdict = "Контроллер удерживает высоту допустимо"
        else:
            verdict = "Контроллер демонстрирует заметные отклонения"
        report_lines.append(f"Вывод: {verdict}")
        report_lines.append("")

    report_lines.append("=" * 90)
    report_lines.append("СРАВНЕНИЕ ПО УРОВНЯМ СЛОЖНОСТИ ВНУТРИ КАЖДОЙ СРЕДЫ")
    report_lines.append("=" * 90)
    report_lines.append("")

    for environment, groups in sorted(environment_groups.items()):
        ordered = sorted(groups, key=lambda s: ["light", "challenging", "extreme"].index(s["difficulty"]) if s["difficulty"] in ["light", "challenging", "extreme"] else 99)
        report_lines.append(f"Среда: {environment}")
        for summary in ordered:
            stats = summary["flight_stats"]
            report_lines.append(
                f"  {summary['difficulty']:<12} "
                f"высота={stats['max_altitude']:.3f} м, "
                f"ошибка={summary['tracking']['mae']:.3f} м, "
                f"устойчивость={summary['stability']['overall_stability']:.3f}, "
                f"сред. throttle={stats['avg_throttle']:.3f}"
            )
        report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = save_dir / "flight_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    for scenario_name, summary in scenario_summaries:
        df = loaded_data[scenario_name] if scenario_name in loaded_data else next(iter(loaded_data.values()))
        plot_flight_data(df, str(save_dir) + os.sep, scenario_name)

    print(report_text)
    print(f"\nПолный отчет сохранен: {report_path}")
    return report_text


# =============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================

if __name__ == "__main__":
    generate_flight_report(
        csv_path="flight_data.xlsx",
        save_path="flight_analysis_plots/",
    )
