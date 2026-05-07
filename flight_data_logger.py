import csv
import numpy as np

from drone_simulation import DroneEnv

SAVE_PATH = "flight_data.csv"


# =============================================================
# КОНТРОЛЛЕР ВЫСОТЫ
# =============================================================

def altitude_controller(target_altitude,
                        current_altitude,
                        velocity_z):

    error = target_altitude - current_altitude

    thrust = (
            0.55
            + error * 0.18
            - velocity_z * 0.08
    )

    return np.clip(thrust, 0.0, 1.0)


# =============================================================
# СИМУЛЯЦИЯ
# =============================================================

def run_simulation():

    env = DroneEnv(gui=True)

    with open(SAVE_PATH, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([

            "time",

            "x",
            "y",
            "z",

            "vx",
            "vy",
            "vz",

            "roll",
            "pitch",
            "yaw",

            "wx",
            "wy",
            "wz",

            "throttle",

            "target_roll",
            "target_pitch",
            "target_yaw_rate"
        ])

        target_altitude = 5.0

        landing = False

        # =====================================================
        # MAIN LOOP
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

            throttle = altitude_controller(
                target_altitude,
                altitude,
                velocity_z
            )

            # =================================================
            # HIGH-LEVEL ACTION
            # =================================================

            action = {

                "throttle": throttle,

                "roll": 0.0,

                "pitch": 0.0,

                "yaw_rate": 0.0
            }

            next_state = env.step(action)

            # =================================================
            # SAVE
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

                action["throttle"],

                action["roll"],
                action["pitch"],
                action["yaw_rate"]
            ])

            # =================================================
            # SUCCESSFUL LANDING
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


# =============================================================
# START
# =============================================================

if __name__ == "__main__":

    run_simulation()