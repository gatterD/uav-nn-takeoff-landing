"""Запуск демонстрационного полета без сохранения данных.

Запуск:
    python manual_takeoff.py

Пользователь выбирает скорость ветра и темп показа. Высота управляется
автоматически общей тягой; управление креном, тангажом и рысканием не реализовано.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from drone_controller import altitude_controller
from drone_simulation import DroneEnv
from flight_outcome import assess_landing


MANUAL_LOG_DIR = Path(__file__).resolve().parent / "manual_logs"


def choose_wind_speed():
    """Возвращает выбранную скорость ветра в метрах в секунду."""
    default_speed = 0.0
    print("\nВыберите скорость ветра из физической модели.")
    print(f"Текущее значение wind_x: {default_speed:.1f} м/с")
    while True:
        value = input("Скорость ветра, м/с (Enter = 0): ").strip()
        if not value:
            return default_speed
        try:
            wind_speed = float(value)
        except ValueError:
            wind_speed = -1.0
        if wind_speed >= 0.0:
            return wind_speed
        print("Введите неотрицательное число.")


def save_trajectory_plot(trajectory, output_path=None):
    """Сохраняет траекторию полета в виде 3D-графика."""
    if not trajectory:
        raise ValueError("Cannot save an empty flight trajectory")
    if output_path is None:
        output_path = MANUAL_LOG_DIR / "manual_flight_trajectory.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x, y, z = zip(*trajectory)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(x, y, z, color="royalblue", label="Траектория")
    axis.scatter(x[0], y[0], z[0], color="green", s=40, label="Старт")
    axis.scatter(x[-1], y[-1], z[-1], color="red", s=40, label="Финиш")
    axis.set_xlabel("x, м")
    axis.set_ylabel("y, м")
    axis.set_zlabel("z, м")
    axis.set_title("3D-траектория полета")
    axis.legend()
    axis.grid(True)
    try:
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)
    print(f"График траектории сохранён: {output_path}")
    return output_path


def run_manual_flight(wind_speed=0.0, gui=True):
    """Запускает полет без задержки между шагами и сохраняет его траекторию."""
    env = DroneEnv(gui=gui)
    env.set_wind(wind_x=wind_speed)
    target_altitude = 5.0
    phase = "takeoff"
    hover_steps = int(2.0 / env.TIME_STEP)
    hover_counter = 0
    trajectory = []

    try:
        for step in range(20000):
            state = env.get_state()

            if (
                phase == "takeoff"
                and state["z"] >= 4.95
                and abs(state["vz"]) < 0.15
            ):
                phase = "hover"
                target_altitude = 5.0
            elif phase == "hover":
                hover_counter += 1
                if hover_counter >= hover_steps:
                    phase = "landing"
                    target_altitude = 0.0

            action = {
                "throttle": altitude_controller(
                    target_altitude,
                    state["z"],
                    state["vz"],
                ),
            }

            state = env.step(action)
            trajectory.append((state["x"], state["y"], state["z"]))
            if step % 120 == 0:
                print(
                    f"t={step * env.TIME_STEP:5.1f} с | "
                    f"этап={phase} | высота={state['z']:.2f} м"
                )

            if (
                phase == "landing"
                and assess_landing(state, env.parameters)["landing_success"]
            ):
                print("Посадка завершена")
                break
        else:
            print("Сценарий не завершен: достигнут лимит шагов")
    finally:
        env.close()
    if trajectory:
        save_trajectory_plot(trajectory)


def main():
    parser = argparse.ArgumentParser(
        description="Демонстрационный полет с автоматическим управлением высотой"
    )
    parser.add_argument("--wind-speed", type=float, default=None)
    args = parser.parse_args()

    wind_speed = args.wind_speed if args.wind_speed is not None else choose_wind_speed()
    if wind_speed < 0.0:
        parser.error("--wind-speed должен быть неотрицательным")
    run_manual_flight(wind_speed=wind_speed)


if __name__ == "__main__":
    main()