"""Интерактивный запуск дрона без сохранения данных полета.

Запуск:
    python manual_takeoff.py

Пользователь выбирает скорость ветра. После этого выполняется один
полный сценарий: взлет, зависание и посадка.
"""

import argparse
import time

from drone_controller import altitude_controller
from drone_physic import DroneAerodynamics
from drone_simulation import DroneEnv


def choose_wind_speed():
    """Возвращает выбранную скорость ветра в метрах в секунду."""
    physics = DroneAerodynamics()
    default_speed = physics.wind_effect_x
    print("\nВыберите скорость ветра из физической модели.")
    print(f"Текущее значение wind_effect_x: {default_speed:.1f} м/с")
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


def choose_slowdown():
    """Выбирает задержку между шагами симуляции."""
    while True:
        value = input("Замедление [1, 2, 4, 8, 16] (по умолчанию 8): ").strip()
        if not value:
            return 8.0
        try:
            slowdown = float(value)
        except ValueError:
            slowdown = 0.0
        if slowdown in (1.0, 2.0, 4.0, 8.0, 16.0):
            return slowdown
        print("Выберите одно из значений: 1, 2, 4, 8 или 16.")


def run_manual_flight(wind_speed=0.0, slowdown=8.0, gui=True):
    """Показывает полный цикл полета без создания файлов с результатами."""
    physics = DroneAerodynamics()
    physics.set_wind(wind_x=wind_speed)
    env = DroneEnv(gui=gui)
    env.set_wind(
        wind_x=physics.wind_effect_x,
        wind_y=physics.wind_effect_y,
        wind_z=physics.wind_effect_z,
    )
    delay = env.TIME_STEP * slowdown
    target_altitude = 5.0
    phase = "takeoff"
    hover_steps = int(2.0 / env.TIME_STEP)
    hover_counter = 0

    try:
        for step in range(20000):
            state = env.get_state()

            if phase == "takeoff" and state["z"] >= 4.95:
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
                "roll": 0.0,
                "pitch": 0.0,
                "yaw_rate": 0.0,
            }

            state = env.step(action)
            if step % 120 == 0:
                print(
                    f"t={step * env.TIME_STEP:5.1f} с | "
                    f"этап={phase} | высота={state['z']:.2f} м"
                )
            if phase == "landing" and state["z"] <= 0.35 and abs(state["vz"]) < 0.15:
                print("Посадка завершена")
                break
            time.sleep(delay)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Просмотр полного полета дрона")
    parser.add_argument("--wind-speed", type=float, default=None)
    parser.add_argument("--slowdown", type=float, choices=(1, 2, 4, 8, 16), default=None)
    args = parser.parse_args()

    wind_speed = args.wind_speed if args.wind_speed is not None else choose_wind_speed()
    if wind_speed < 0.0:
        parser.error("--wind-speed должен быть неотрицательным")
    slowdown = args.slowdown or choose_slowdown()
    run_manual_flight(wind_speed=wind_speed, slowdown=slowdown)


if __name__ == "__main__":
    main()