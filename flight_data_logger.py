import csv
import math
import random
import pandas as pd
from drone_controller import run_flight_simulation

SAVE_PATH = "flight_data.csv"
EXCEL_PATH = "flight_data.xlsx"


BASE_ENVIRONMENTS = {
    "calm_wind": {
        "wind_speed": (0.0, 0.5),
        "air_density": (1.20, 1.25),
        "turbulence": (0.005, 0.02),
        "wind_shear": (0.005, 0.015),
    },
    "moderate_wind": {
        "wind_speed": (0.5, 2.0),
        "air_density": (1.15, 1.22),
        "turbulence": (0.02, 0.08),
        "wind_shear": (0.015, 0.05),
    },
    "strong_wind": {
        "wind_speed": (2.0, 5.0),
        "air_density": (1.05, 1.18),
        "turbulence": (0.06, 0.18),
        "wind_shear": (0.04, 0.12),
    },
}

DIFFICULTY_LEVELS = {
    "light": 0.75,
    "challenging": 1.25,
    "extreme": 1.75,
}


def _scaled_range(value_range, factor, center=None):
    """Расширяет диапазон относительно центра пропорционально сложности."""
    low, high = value_range
    if center is None:
        center = (low + high) / 2.0
    half_width = (high - low) / 2.0 * factor
    return max(0.0, center - half_width), center + half_width


def generate_random_scenarios(rng=None):
    """Создаёт 9 случайных сценариев: 3 среды и 3 уровня сложности."""
    rng = rng or random.Random()
    scenarios = {}

    for environment_name, environment in BASE_ENVIRONMENTS.items():
        for difficulty_name, factor in DIFFICULTY_LEVELS.items():
            wind_low, wind_high = _scaled_range(environment["wind_speed"], factor)
            density_range = _scaled_range(environment["air_density"], factor)
            turbulence_range = _scaled_range(environment["turbulence"], factor)
            shear_range = _scaled_range(environment["wind_shear"], factor)

            speed = rng.uniform(wind_low, wind_high)
            direction = rng.uniform(0.0, 2.0 * math.pi)
            scenarios[f"{environment_name}_{difficulty_name}"] = {
                "wind": (
                    speed * math.cos(direction),
                    speed * math.sin(direction),
                    rng.uniform(-0.15, 0.15) * factor,
                ),
                "air_density": rng.uniform(*density_range),
                "turbulence_amplitude": rng.uniform(*turbulence_range),
                "wind_shear_factor": rng.uniform(*shear_range),
            }

    return scenarios


# =============================================================
# ЛОГИРОВАНИЕ ДАННЫХ ПОЛЕТА
# =============================================================

def save_flight_data_to_csv(flight_data_generator, output_path=SAVE_PATH):
    """
    Сохраняет данные полета в CSV файл.

    Args:
        flight_data_generator: Генератор данных полета из drone_controller
        output_path: Путь к выходному CSV файлу
    """
    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "phase",
            "target_altitude",
            "altitude",
            "altitude_error",
            "vertical_velocity",
            "throttle",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        ])
        for flight_data in flight_data_generator:
            writer.writerow([
                flight_data["time"],
                flight_data["phase"],
                flight_data["target_altitude"],
                flight_data["altitude"],
                flight_data["altitude_error"],
                flight_data["vertical_velocity"],
                flight_data["throttle"],
                flight_data["x"],
                flight_data["y"],
                flight_data["z"],
                flight_data["roll"],
                flight_data["pitch"],
                flight_data["yaw"],
            ])
    print(f"Данные сохранены в {output_path}")


def save_flight_data_to_excel(scenario_data, output_path=EXCEL_PATH):
    """
    Сохраняет результаты нескольких сценариев в один Excel-файл.
    Каждый сценарий сохраняется на отдельном листе.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for scenario_name, records in scenario_data.items():
            df = pd.DataFrame(records)
            sheet_name = str(scenario_name)[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Excel-файл сохранен: {output_path}")


def run_scenarios_and_save_excel(scenarios, gui=False, output_path=EXCEL_PATH):
    """
    Запускает несколько сценариев полета и сохраняет их результаты в Excel.
    """
    scenario_results = {}
    for scenario_name, env_params in scenarios.items():
        print(f"Запуск сценария: {scenario_name}")
        records = []
        for flight_data in run_flight_simulation(gui=gui, env_params=env_params, scenario_name=scenario_name):
            records.append(flight_data)
        scenario_results[scenario_name] = records

    save_flight_data_to_excel(scenario_results, output_path=output_path)
    return scenario_results


# =============================================================
# START
# =============================================================

if __name__ == "__main__":
    scenarios = generate_random_scenarios()
    run_scenarios_and_save_excel(scenarios, gui=False, output_path=EXCEL_PATH)
