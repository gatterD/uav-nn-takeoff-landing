import numpy as np

from drone_simulation import DroneEnv
from drone_physic import DroneAerodynamics


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
# СИМУЛЯЦИЯ ПОЛЕТА
# =============================================================

def run_flight_simulation(gui=True, env_params=None, scenario_name="default"):
    """
    Запускает симуляцию полета дрона и возвращает генератор данных полета.
    Включает аэродинамические эффекты и позволяет задавать условия среды.
    
    Args:
        gui: Использовать GUI симуляцию (True/False)
        env_params: dict с параметрами среды, например:
            {
                "wind": (wx, wy, wz),
                "air_density": 1.225,
                "turbulence_amplitude": 0.03,
                "wind_shear_factor": 0.02,
            }
        scenario_name: имя сценария, которое будет добавлено к данным
    
    Yields:
        dict: Данные о состоянии дрона и действиях
    """
    
    env = DroneEnv(gui=gui)
    aerodynamics = DroneAerodynamics()

    if env_params is not None:
        env.set_environment(
            wind=env_params.get("wind"),
            air_density=env_params.get("air_density"),
            turbulence_amplitude=env_params.get("turbulence_amplitude"),
            wind_shear_factor=env_params.get("wind_shear_factor"),
        )

    target_altitude = 5.0
    landing = False
    
    try:
        # =====================================================
        # MAIN LOOP
        # =====================================================
        
        for step in range(20000):
            
            state = env.get_state()
            
            altitude = state["z"]
            velocity_z = state["vz"]
            altitude_error = target_altitude - altitude
            
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
            
            env.step(action)
            
            # =================================================
            # APPLY AERODYNAMIC EFFECTS
            # =================================================
            
            # Применяем аэродинамические эффекты к состоянию
            state = aerodynamics.apply_aerodynamic_effects(
                state,
                time_step=env.TIME_STEP
            )
            
            # =================================================
            # YIELD DATA
            # =================================================
            
            phase = "landing" if landing else "takeoff"
            
            flight_data = {
                "scenario": scenario_name,
                "time": step * env.TIME_STEP,
                "phase": phase,
                "target_altitude": target_altitude,
                "altitude": altitude,
                "altitude_error": altitude_error,
                "vertical_velocity": velocity_z,
                "throttle": action["throttle"],
                "x": state["x"],
                "y": state["y"],
                "z": state["z"],
                "roll": state["roll"],
                "pitch": state["pitch"],
                "yaw": state["yaw"],
                "mass": state.get("mass"),
                "air_speed": state.get("air_speed"),
                "angle_of_attack": state.get("angle_of_attack"),
                "sideslip_angle": state.get("sideslip_angle"),
                "flight_path_angle": state.get("flight_path_angle"),
                "path_angle": state.get("path_angle"),
                "roll_angle": state.get("roll_angle"),
                "thrust": state.get("thrust"),
                "aero_force_x": state.get("aero_force_x"),
                "aero_force_y": state.get("aero_force_y"),
                "aero_force_z": state.get("aero_force_z"),
                "inertia_x": state.get("inertia_x"),
                "inertia_y": state.get("inertia_y"),
                "inertia_z": state.get("inertia_z"),
                "angular_velocity_x": state.get("angular_velocity_x"),
                "angular_velocity_y": state.get("angular_velocity_y"),
                "angular_velocity_z": state.get("angular_velocity_z"),
                "aero_moment_x": state.get("aero_moment_x"),
                "aero_moment_y": state.get("aero_moment_y"),
                "aero_moment_z": state.get("aero_moment_z"),
                "air_density": state.get("air_density"),
                "wing_area": state.get("wing_area"),
                "wind_x": state.get("wind_x"),
                "wind_y": state.get("wind_y"),
                "wind_z": state.get("wind_z"),
                "turbulence": state.get("turbulence"),
                "wind_shear": state.get("wind_shear"),
                "ground_effect": state.get("ground_effect"),
            }
            
            yield flight_data
            
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
    
    finally:
        env.close()
