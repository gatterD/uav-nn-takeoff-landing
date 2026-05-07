import csv
import numpy as np
from drone_simulation import DroneEnv

SAVE_PATH = "flight_data.csv"


# =========================================================
# ПРОСТОЙ PID-КОНТРОЛЛЕР ВЫСОТЫ
# =========================================================

def altitude_controller(target_altitude, current_altitude, velocity_z):

    error = target_altitude - current_altitude

    thrust = (
            0.55
            + error * 0.18
            - velocity_z * 0.08
    )

    return np.clip(thrust, 0.0, 1.0)


# =========================================================
# ОСНОВНАЯ СИМУЛЯЦИЯ
# =========================================================

def run_simulation():

    env = DroneEnv(gui=True)

    with open(SAVE_PATH, mode="w", newline="") as f:

        writer = csv.writer(f)

        # =====================================================
        # ЗАГОЛОВКИ CSV
        # =====================================================

        writer.writerow([
            "time",

            "x", "y", "z",

            "vx", "vy", "vz",

            "roll", "pitch", "yaw",

            "wx", "wy", "wz",

            "motor_fl",
            "motor_fr",
            "motor_rl",
            "motor_rr"
        ])

        target_altitude = 5.0

        landing = False

        # =====================================================
        # ГЛАВНЫЙ ЦИКЛ
        # =====================================================

        for step in range(20000):

            state = env.get_state()

            altitude = state["z"]

            velocity_z = state["vz"]

            # =================================================
            # ПОСАДКА
            # =================================================

            if step > 10000:
                landing = True
                target_altitude = 0.2

            # =================================================
            # THROTTLE
            # =================================================

            thrust = altitude_controller(
                target_altitude,
                altitude,
                velocity_z
            )

            # =================================================
            # УПРАВЛЕНИЕ 4 МОТОРАМИ
            # =================================================

            action = [
                thrust,  # front_left
                thrust,  # front_right
                thrust,  # rear_left
                thrust   # rear_right
            ]

            # =================================================
            # STEP
            # =================================================

            next_state = env.step(action)

            # =================================================
            # СОХРАНЕНИЕ
            # =================================================

            writer.writerow([

                step * env.TIME_STEP,

                state["x"],
                state["y"],
                state["z"],

                state["vx"],
                state["vy"],
                state["vz"],

                state["roll"],
                state["pitch"],
                state["yaw"],

                state["wx"],
                state["wy"],
                state["wz"],

                action[0],
                action[1],
                action[2],
                action[3],
            ])

            # =================================================
            # УСЛОВИЕ ПОСАДКИ
            # =================================================

            if (
                    landing
                    and altitude < 0.25
                    and abs(velocity_z) < 0.15
            ):
                print("Посадка завершена")
                break

    env.close()

    print(f"Данные сохранены в {SAVE_PATH}")


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    run_simulation()