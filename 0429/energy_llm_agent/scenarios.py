# -*- coding: utf-8 -*-
"""
Scenario Definitions  (0422 — 5+ FMU co-simulation)
=====================================================
각 시나리오는 최소 5개 FMU를 묶어 최적화를 수행한다.

시나리오 목록:
  1. construction_vessel_full  — 9 FMU (OSP construction-vessel 전체)
  2. lars_recovery             — 8 FMU (OSP LARS 전체: A-Frame + Winch + 제어계)
  3. dp_ship_station           — 5 FMU (OSP dp-ship 전체: DP 스테이션 유지)

FMU inspect 확인 변수명 (fmpy read_model_description 기준):
  - LARS Winch     inputs:  torqueMotor1.in1, torqueMotor2.in1, torqueMotor3.in1, vesselHeave.x
                   outputs: motorSpeed.om, drum.F1, drum.T2, loadDepth.y
  - dp-ship DPController inputs:  x, y, psi, dx, dy, dpsi + _ref variants + reset1~3
                          outputs: Controlx, Controly, ControlMz
                          params:  Kpx, Kpy, Kppsi, Kdx, Kdy, Kdpsi, Kix, Kiy, Kipsi
  - dp-ship NLPobserver  params:  w01,w02,w03, lambda1,lambda2,lambda3
  - dp-ship ReferenceGenerator params: Tx, Ty, Tpsi, dxMax, dyMax, dpsiMax
"""

import copy
import os

# ══════════════════════════════════════════════════════════════════════════════
# 1. Construction Vessel Full DP  (9 FMU)
# ══════════════════════════════════════════════════════════════════════════════

CONSTRUCTION_VESSEL_FULL = {
    "name": "construction_vessel_full",
    "display_name": "Construction Vessel — Full DP + Winch (9 FMU)",
    "description": (
        "Complete OSP construction-vessel co-simulation (9 FMU): wave model, wind model, "
        "vessel model, DP controller, reference model, thrust allocator, thruster model, "
        "power system, and winch. Goal: tune wave excitation and winch PID gains to achieve "
        "target load depth while minimising power violations across the full system."
    ),
    "fmus": [
        {
            "name": "wave_model",
            "fmu": "linearized_wave_model",
            "case": "construction-vessel",
            "step_size": 0.1,
            "role": "Linearised first-order wave disturbances",
            # PARAMS (11): wave_height, peak_frequency, k_down, k_north, k_east,
            #              k_pitch, k_roll, k_yaw, east_drift_limit, north_drift_limit, yaw_drift_limit
            "default_params": {
                "wave_height": 3.0, "peak_frequency": 0.8,
                "k_down": 1.5, "k_north": 2.0, "k_east": 2.0,
            },
            "tunable_params": {
                "wave_height":    {"min": 1.0, "max": 5.0,  "unit": "m",     "desc": "Significant wave height"},
                "peak_frequency": {"min": 0.3, "max": 2.0,  "unit": "rad/s", "desc": "Wave peak frequency"},
                "k_down":         {"min": 0.5, "max": 4.0,  "unit": "-",     "desc": "Vertical wave stiffness"},
                "k_north":        {"min": 0.5, "max": 4.0,  "unit": "-",     "desc": "North wave stiffness"},
                "k_east":         {"min": 0.5, "max": 4.0,  "unit": "-",     "desc": "East wave stiffness"},
            },
        },
        {
            "name": "wind_model",
            "fmu": "wind_model",
            "case": "construction-vessel",
            "step_size": 0.1,
            "role": "Wind disturbance (vessel position/velocity dependent)",
            # inputs: vessel_position.*, vessel_velocity.*
            # outputs: wind_forces.surge/sway/heave/roll/pitch/yaw
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "vessel_model",
            "fmu": "vessel_model",
            "case": "construction-vessel",
            "step_size": 0.01,
            "role": "6-DOF vessel rigid-body dynamics",
            # inputs: wind_forces.*, wave_forces.*, thruster_states.*
            # outputs: ned_position.*, body_velocity.*
            # PARAMS: none (all hardcoded in FMU)
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "reference_model",
            "fmu": "dp_reference_model",
            "case": "construction-vessel",
            "step_size": 0.01,
            "role": "Reference trajectory / setpoint filter for DP controller",
            # inputs: setpoint.north, setpoint.east, setpoint.yaw
            # outputs: filtered_setpoint.north, filtered_setpoint.east, filtered_setpoint.yaw
            # PARAMS: none
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "dp_controller",
            "fmu": "dp_controller",
            "case": "construction-vessel",
            "step_size": 0.01,
            "role": "Dynamic Positioning controller — surge/sway/yaw",
            # inputs: vessel_position.north/east/down/roll/pitch/yaw,
            #         vessel_setpoint.north/east/yaw
            # outputs: force_command.surge/sway/yaw
            # PARAMS: none (gains hardcoded in FMU)
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "thrust_allocator",
            "fmu": "thrust_allocation",
            "case": "construction-vessel",
            "step_size": 0.01,
            "role": "Thrust allocation: force demand → individual thruster RPM/azimuth",
            # inputs: force_command.surge/sway/yaw
            # outputs: thrust_command.tunnel_thruster_{1,2,3}.rpm,
            #          thrust_command.main_propeller_{port,starboard}.rpm/azimuth
            # PARAMS: none
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "thruster_model",
            "fmu": "thruster_model",
            "case": "construction-vessel",
            "step_size": 0.01,
            "role": "5-thruster dynamics (3×tunnel + 2×azimuthing)",
            # inputs: thrust_command.*
            # outputs: thruster_states.*
            # PARAMS: none
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "power_system",
            "fmu": "power_system",
            "case": "construction-vessel",
            "step_size": 0.1,
            "role": "Shipboard power system — bus loads from thrusters and winch",
            # inputs: bus_1_loads.active_power, thruster_states.*
            # outputs: bus voltages, total_bus_power
            # PARAMS: numberOfBuses (not tunable)
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "winch",
            "fmu": "winch",
            "case": "construction-vessel",
            "step_size": 0.1,
            "role": "Crane winch PID position controller",
            # PARAMS (3): K_p, K_d, K_i
            # INPUTS (7): load_depth_setpoint, vessel_position.north/east/down/roll/pitch/yaw
            # OUTPUTS (3): load_depth, power_consumption, motor_speed
            "default_params": {"K_p": 200.0, "K_d": 1000.0, "K_i": 1.0},
            "tunable_params": {
                "K_p": {"min": 50.0,  "max": 800.0,  "unit": "N/m",   "desc": "Winch PID proportional gain"},
                "K_d": {"min": 100.0, "max": 3000.0, "unit": "N·s/m", "desc": "Winch PID derivative gain"},
                "K_i": {"min": 0.0,   "max": 10.0,   "unit": "N/m·s", "desc": "Winch PID integral gain"},
            },
        },
    ],

    # ── 1:1 scalar 연결 (VariableGroupConnection 전개) ──────────────────────
    "connections": [
        # reference_model → dp_controller (filtered_setpoint → vessel_setpoint)
        {"from_fmu": "reference_model", "from_var": "filtered_setpoint.north",
         "to_fmu": "dp_controller",     "to_var":   "vessel_setpoint.north"},
        {"from_fmu": "reference_model", "from_var": "filtered_setpoint.east",
         "to_fmu": "dp_controller",     "to_var":   "vessel_setpoint.east"},
        {"from_fmu": "reference_model", "from_var": "filtered_setpoint.yaw",
         "to_fmu": "dp_controller",     "to_var":   "vessel_setpoint.yaw"},

        # dp_controller → thrust_allocator (force_command)
        {"from_fmu": "dp_controller",  "from_var": "force_command.surge",
         "to_fmu": "thrust_allocator", "to_var":   "force_command.surge"},
        {"from_fmu": "dp_controller",  "from_var": "force_command.sway",
         "to_fmu": "thrust_allocator", "to_var":   "force_command.sway"},
        {"from_fmu": "dp_controller",  "from_var": "force_command.yaw",
         "to_fmu": "thrust_allocator", "to_var":   "force_command.yaw"},

        # vessel_model → dp_controller (ned_position)
        {"from_fmu": "vessel_model",  "from_var": "ned_position.north",
         "to_fmu": "dp_controller",   "to_var":   "vessel_position.north"},
        {"from_fmu": "vessel_model",  "from_var": "ned_position.east",
         "to_fmu": "dp_controller",   "to_var":   "vessel_position.east"},
        {"from_fmu": "vessel_model",  "from_var": "ned_position.yaw",
         "to_fmu": "dp_controller",   "to_var":   "vessel_position.yaw"},

        # vessel_model → wind_model (ned_position)
        {"from_fmu": "vessel_model", "from_var": "ned_position.north",
         "to_fmu": "wind_model",     "to_var":   "vessel_position.north"},
        {"from_fmu": "vessel_model", "from_var": "ned_position.east",
         "to_fmu": "wind_model",     "to_var":   "vessel_position.east"},
        {"from_fmu": "vessel_model", "from_var": "ned_position.yaw",
         "to_fmu": "wind_model",     "to_var":   "vessel_position.yaw"},

        # vessel_model → wind_model (body_velocity → vessel_velocity)
        {"from_fmu": "vessel_model", "from_var": "body_velocity.surge",
         "to_fmu": "wind_model",     "to_var":   "vessel_velocity.surge"},
        {"from_fmu": "vessel_model", "from_var": "body_velocity.sway",
         "to_fmu": "wind_model",     "to_var":   "vessel_velocity.sway"},
        {"from_fmu": "vessel_model", "from_var": "body_velocity.heave",
         "to_fmu": "wind_model",     "to_var":   "vessel_velocity.heave"},

        # wind_model → vessel_model (wind_forces)
        {"from_fmu": "wind_model",  "from_var": "wind_forces.surge",
         "to_fmu": "vessel_model",  "to_var":   "wind_forces.surge"},
        {"from_fmu": "wind_model",  "from_var": "wind_forces.sway",
         "to_fmu": "vessel_model",  "to_var":   "wind_forces.sway"},
        {"from_fmu": "wind_model",  "from_var": "wind_forces.yaw",
         "to_fmu": "vessel_model",  "to_var":   "wind_forces.yaw"},

        # thrust_allocator → thruster_model (5 thrusters)
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.tunnel_thruster_1.rpm",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.tunnel_thruster_1.rpm"},
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.tunnel_thruster_2.rpm",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.tunnel_thruster_2.rpm"},
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.tunnel_thruster_3.rpm",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.tunnel_thruster_3.rpm"},
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.main_propeller_port.rpm",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.main_propeller_port.rpm"},
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.main_propeller_port.azimuth",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.main_propeller_port.azimuth"},
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.main_propeller_starboard.rpm",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.main_propeller_starboard.rpm"},
        {"from_fmu": "thrust_allocator", "from_var": "thrust_command.main_propeller_starboard.azimuth",
         "to_fmu": "thruster_model",     "to_var":   "thrust_command.main_propeller_starboard.azimuth"},

        # thruster_model → vessel_model (thruster states)
        {"from_fmu": "thruster_model", "from_var": "thruster_states.tunnel_thruster_1.thrust",
         "to_fmu": "vessel_model",     "to_var":   "thruster_states.tunnel_thruster_1.thrust"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.tunnel_thruster_2.thrust",
         "to_fmu": "vessel_model",     "to_var":   "thruster_states.tunnel_thruster_2.thrust"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.tunnel_thruster_3.thrust",
         "to_fmu": "vessel_model",     "to_var":   "thruster_states.tunnel_thruster_3.thrust"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.main_propeller_port.thrust",
         "to_fmu": "vessel_model",     "to_var":   "thruster_states.main_propeller_port.thrust"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.main_propeller_starboard.thrust",
         "to_fmu": "vessel_model",     "to_var":   "thruster_states.main_propeller_starboard.thrust"},

        # thruster_model → power_system (power consumption)
        {"from_fmu": "thruster_model", "from_var": "thruster_states.tunnel_thruster_1.power_consumption",
         "to_fmu": "power_system",     "to_var":   "thruster_states.tunnel_thruster_1.power_consumption"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.tunnel_thruster_2.power_consumption",
         "to_fmu": "power_system",     "to_var":   "thruster_states.tunnel_thruster_2.power_consumption"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.tunnel_thruster_3.power_consumption",
         "to_fmu": "power_system",     "to_var":   "thruster_states.tunnel_thruster_3.power_consumption"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.main_propeller_port.power_consumption",
         "to_fmu": "power_system",     "to_var":   "thruster_states.main_propeller_port.power_consumption"},
        {"from_fmu": "thruster_model", "from_var": "thruster_states.main_propeller_starboard.power_consumption",
         "to_fmu": "power_system",     "to_var":   "thruster_states.main_propeller_starboard.power_consumption"},

        # winch → power_system
        {"from_fmu": "winch",       "from_var": "power_consumption",
         "to_fmu": "power_system",  "to_var":   "bus_1_loads.active_power"},
    ],

    # ── VectorSum: vessel_model.ned + wave_model.disturbances → winch.vessel_pos ─
    "sum_connections": [
        {"from_a": {"fmu": "vessel_model", "var": "ned_position.north"},
         "from_b": {"fmu": "wave_model",   "var": "first_order_disturbances.north"},
         "to_fmu": "winch", "to_var": "vessel_position.north"},
        {"from_a": {"fmu": "vessel_model", "var": "ned_position.east"},
         "from_b": {"fmu": "wave_model",   "var": "first_order_disturbances.east"},
         "to_fmu": "winch", "to_var": "vessel_position.east"},
        {"from_a": {"fmu": "vessel_model", "var": "ned_position.down"},
         "from_b": {"fmu": "wave_model",   "var": "first_order_disturbances.down"},
         "to_fmu": "winch", "to_var": "vessel_position.down"},
        {"from_a": {"fmu": "vessel_model", "var": "ned_position.roll"},
         "from_b": {"fmu": "wave_model",   "var": "first_order_disturbances.roll"},
         "to_fmu": "winch", "to_var": "vessel_position.roll"},
        {"from_a": {"fmu": "vessel_model", "var": "ned_position.pitch"},
         "from_b": {"fmu": "wave_model",   "var": "first_order_disturbances.pitch"},
         "to_fmu": "winch", "to_var": "vessel_position.pitch"},
    ],

    # reference_model에 고정 setpoint 주입
    "control": {
        "reference_model": {"setpoint.north": 0.0, "setpoint.east": 0.0, "setpoint.yaw": 0.0},
        "winch":           {"load_depth_setpoint": 30.0},
    },

    "metric": {
        "depth_fmu":       "winch",
        "depth_var":       "load_depth",
        "target_depth":    30.0,
        "power_fmu":       "winch",
        "power_var":       "power_consumption",
        "power_limit_kw":  800.0,
        "failure_penalty": 0.5,
    },
    "step_size":       0.1,
    "max_duration":    300.0,
    "depth_tolerance": 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LARS Full Recovery System  (8 FMU)
# ══════════════════════════════════════════════════════════════════════════════

LARS_RECOVERY = {
    "name": "lars_recovery",
    "display_name": "LARS — A-Frame + Winch Full Recovery System (8 FMU)",
    "description": (
        "Full Launch and Recovery System (LARS) co-simulation (8 FMU): "
        "Winch + WinchActuator + WinchController + WinchActuatorSetpoint (winch side), "
        "AFrame + AFrameActuator + AFrameController + AFrameActuatorSetpoint (A-Frame side). "
        "Goal: optimise winch mechanical parameters (gear ratio, drum, inertia, efficiency) "
        "and controller gains to reach target depth with minimum oscillation."
    ),
    "fmus": [
        {
            "name": "winch_actuator_setpoint",
            "fmu": "WinchActuatorSetpoint",
            "case": "lars",
            "step_size": 0.001,
            "role": "Constant winch depth setpoint source",
            # PARAMS (1): C (constant setpoint value)
            # OUTPUTS (1): winchSetpoint
            # 깊이 목표는 시나리오 metric.target_depth(200m)와 반드시 일치 — LLM이 임의로 바꾸면 성공률이 붕괴함
            "default_params": {"C": 200.0},
            "tunable_params": {},
        },
        {
            "name": "winch_controller",
            "fmu": "WinchController",
            "case": "lars",
            "step_size": 0.001,
            "role": "PID winch depth/speed controller",
            # PARAMS (10): Controller.K, Controller.Td, Controller.N, Controller.Ti,
            #              Gain1.K, Gain4.K, Gain4.loadDepth_min/max,
            #              Controller.uI_initial, Controller.uDstate_initial
            # INPUTS (3): loadDepth, motorSpeed, winchSetpoint
            # OUTPUTS (1): motorGain
            "default_params": {},
            "tunable_params": {
                # 수치 불안정 방지를 위해 탐색 범위를 안정 영역으로 축소
                "Controller.K":  {"min": 0.5,  "max": 3.0,  "unit": "-", "desc": "Winch PID proportional gain"},
                "Controller.Td": {"min": 0.0,  "max": 0.6,  "unit": "s", "desc": "Winch PID derivative time"},
                "Controller.Ti": {"min": 1.0,  "max": 8.0,  "unit": "s", "desc": "Winch PID integral time"},
                "Controller.N":  {"min": 2.0,  "max": 10.0, "unit": "-", "desc": "Winch PID filter coefficient"},
                "Gain1.K":       {"min": 0.1,  "max": 2.0,  "unit": "-", "desc": "Winch controller inner gain"},
            },
        },
        {
            "name": "winch_actuator",
            "fmu": "WinchActuator",
            "case": "lars",
            "step_size": 0.001,
            "role": "Motor actuator converting gain to torque",
            # PARAMS (1): MSe.T (motor time constant)
            # INPUTS (2): motorGain, motorSpeed
            # OUTPUTS (1): motorTorque
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "winch",
            "fmu": "Winch",
            "case": "lars",
            "step_size": 0.001,
            "role": "Winch mechanical model — drum, gear, rope",
            # PARAMS (14): lm, gr, D, dI, ild, geff, sD, fst,
            #              settings.dtMin/Max, settings.absTol/relTol, settings.iMode/maxOrder
            # INPUTS (4): torqueMotor1.in1, torqueMotor2.in1, torqueMotor3.in1, vesselHeave.x
            # OUTPUTS (4): motorSpeed.om, drum.F1, drum.T2, loadDepth.y
            "default_params": {
                "lm": 500.0, "D": 1.36, "gr": 141.0,
                "dI": 1600.0, "ild": 200.0, "geff": 0.925, "sD": 1.1,
            },
            "tunable_params": {
                "D":    {"min": 1.1,   "max": 1.6,   "unit": "m",     "desc": "Drum diameter"},
                "gr":   {"min": 120.0, "max": 160.0, "unit": "-",     "desc": "Gear ratio"},
                "dI":   {"min": 1200.0,"max": 2200.0,"unit": "kg·m²", "desc": "Drum rotational inertia"},
                "ild":  {"min": 120.0, "max": 320.0, "unit": "kg·m²", "desc": "Inertia of load"},
                "geff": {"min": 0.85,  "max": 0.96,  "unit": "-",     "desc": "Gear efficiency"},
                "sD":   {"min": 1.0,   "max": 1.2,   "unit": "-",     "desc": "Speed ratio"},
            },
        },
        {
            "name": "a_frame_actuator_setpoint",
            "fmu": "AframeActuatorSetPoint",
            "case": "lars",
            "step_size": 0.001,
            "role": "Constant A-Frame deployment setpoint source",
            # PARAMS (1): C (constant setpoint, typically 1.0 = fully deployed)
            # OUTPUTS (1): aFrameSetpoint
            "default_params": {"C": 1.0},
            "tunable_params": {},
        },
        {
            "name": "a_frame_controller",
            "fmu": "AFrameController",
            "case": "lars",
            "step_size": 0.001,
            "role": "PID A-Frame cylinder controller",
            # PARAMS (11): Controller1.K/Td/N/Ti, Gain1.K, Gain4.K, Gain4.cl1_min/max,
            #              Controller1.uI_initial/uDstate_initial, Integrate.initial
            # INPUTS (2): aFrameSetpoint, cylinderSpeed
            # OUTPUTS (1): cylinderGain
            "default_params": {},
            "tunable_params": {
                "Controller1.K":  {"min": 0.5,  "max": 3.0,  "unit": "-", "desc": "A-Frame PID proportional gain"},
                "Controller1.Td": {"min": 0.0,  "max": 0.6,  "unit": "s", "desc": "A-Frame PID derivative time"},
                "Controller1.Ti": {"min": 1.0,  "max": 8.0,  "unit": "s", "desc": "A-Frame PID integral time"},
                "Controller1.N":  {"min": 2.0,  "max": 10.0, "unit": "-", "desc": "A-Frame PID filter coefficient"},
            },
        },
        {
            "name": "a_frame_actuator",
            "fmu": "AFrameActuator",
            "case": "lars",
            "step_size": 0.001,
            "role": "Hydraulic actuator driving A-Frame cylinder",
            # PARAMS (1): MSe.F (hydraulic force gain)
            # INPUTS (2): cylinderGain, cylinderSpeed
            # OUTPUTS (1): cylinderForce
            "default_params": {},
            "tunable_params": {},
        },
        {
            "name": "a_frame",
            "fmu": "AFrame",
            "case": "lars",
            "step_size": 0.001,
            "role": "A-Frame structural dynamics (boom + cylinder)",
            # PARAMS (14): Cylinder.a/b, Gravity.l/a/b, I.aframeMass/loadMass/l,
            #              MTF.l/a/b, Se.aframeMass, I.moment_initial, QSensorCylinder.q_init
            # INPUTS (2): cylinderForce, winchForce
            # OUTPUTS (2): tiltAngleDeg, cylinderSpeed
            "default_params": {},
            "tunable_params": {},
        },
    ],

    # ── connections (실제 FMU 변수명 기준) ──────────────────────────────────
    "connections": [
        # winch_actuator_setpoint → winch_controller
        {"from_fmu": "winch_actuator_setpoint", "from_var": "winchSetpoint",
         "to_fmu":   "winch_controller",        "to_var":   "winchSetpoint"},

        # winch_controller → winch_actuator
        {"from_fmu": "winch_controller", "from_var": "motorGain",
         "to_fmu":   "winch_actuator",   "to_var":   "motorGain"},

        # winch_actuator → winch (3× motors)
        {"from_fmu": "winch_actuator", "from_var": "motorTorque",
         "to_fmu":   "winch",          "to_var":   "torqueMotor1.in1"},
        {"from_fmu": "winch_actuator", "from_var": "motorTorque",
         "to_fmu":   "winch",          "to_var":   "torqueMotor2.in1"},
        {"from_fmu": "winch_actuator", "from_var": "motorTorque",
         "to_fmu":   "winch",          "to_var":   "torqueMotor3.in1"},

        # winch → winch_actuator + winch_controller (feedback)
        {"from_fmu": "winch",           "from_var": "motorSpeed.om",
         "to_fmu":   "winch_actuator",  "to_var":   "motorSpeed"},
        {"from_fmu": "winch",           "from_var": "motorSpeed.om",
         "to_fmu":   "winch_controller","to_var":   "motorSpeed"},
        {"from_fmu": "winch",           "from_var": "loadDepth.y",
         "to_fmu":   "winch_controller","to_var":   "loadDepth"},

        # winch → a_frame (tension force)
        {"from_fmu": "winch",  "from_var": "drum.F1",
         "to_fmu":  "a_frame", "to_var":   "winchForce"},

        # a_frame_actuator_setpoint → a_frame_controller
        {"from_fmu": "a_frame_actuator_setpoint", "from_var": "aFrameSetpoint",
         "to_fmu":   "a_frame_controller",        "to_var":   "aFrameSetpoint"},

        # a_frame_controller → a_frame_actuator
        {"from_fmu": "a_frame_controller", "from_var": "cylinderGain",
         "to_fmu":   "a_frame_actuator",   "to_var":   "cylinderGain"},

        # a_frame_actuator → a_frame
        {"from_fmu": "a_frame_actuator", "from_var": "cylinderForce",
         "to_fmu":   "a_frame",          "to_var":   "cylinderForce"},

        # a_frame → a_frame_actuator + a_frame_controller (feedback)
        {"from_fmu": "a_frame",           "from_var": "cylinderSpeed",
         "to_fmu":   "a_frame_actuator",  "to_var":   "cylinderSpeed"},
        {"from_fmu": "a_frame",           "from_var": "cylinderSpeed",
         "to_fmu":   "a_frame_controller","to_var":   "cylinderSpeed"},
    ],

    "sum_connections": [],

    "control": {},

    "metric": {
        "depth_fmu":       "winch",
        "depth_var":       "loadDepth.y",
        "target_depth":    200.0,
        "power_fmu":       "winch",
        "power_var":       "motorSpeed.om",
        # rad/s — fmu_sim 에서 power_limit_raw 는 kW 환산 없이 그대로 한계로 사용
        "power_limit_raw": 58.0,
        "failure_penalty": 0.45,
        "success_mode":    "range",
    },
    "step_size":       0.001,
    # OSP LARS 데모(약 30s)보다 짧게 두되, 목표 깊이 도달에 충분한 시간 확보
    "max_duration":    14.0,
    "trial_timeout":   35.0,
    "depth_tolerance": 4.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. DP Ship Station Keeping  (5 FMU)
# ══════════════════════════════════════════════════════════════════════════════

DP_SHIP_STATION = {
    "name": "dp_ship_station",
    "display_name": "DP Ship Station Keeping — NLP Observer + MPC Thrust (5 FMU)",
    "description": (
        "Full dp-ship co-simulation (5 FMU): DP Controller, NLP Observer, "
        "Reference Generator, MPC-based Thrust Allocator (ThMPC), and OSOM ship model. "
        "Goal: keep the vessel at station (x=0, y=0) by tuning DP controller PID gains "
        "and reference generator time constants."
    ),
    "fmus": [
        {
            "name": "reference_generator",
            "fmu": "ReferenceGenerator",
            "case": "dp-ship",
            "step_size": 0.04,
            "role": "Low-pass reference filter for position/velocity setpoints",
            # PARAMS (21): Tx, Ty, Tpsi (time const), dxMax, dyMax, dpsiMax (vel limits),
            #              Surge0, Sway0, Yaw0 (initial states), relTol, absTol, ...
            # INPUTS (6): x_wp, y_wp, psi_wp, x_tp, y_tp, psi_tp
            # OUTPUTS (6): x_ref, y_ref, psi_ref, dx_ref, dy_ref, dpsi_ref
            "default_params": {},
            "tunable_params": {
                "Tx":     {"min": 10.0,  "max": 200.0, "unit": "s",     "desc": "Surge reference low-pass time constant"},
                "Ty":     {"min": 10.0,  "max": 200.0, "unit": "s",     "desc": "Sway reference low-pass time constant"},
                "Tpsi":   {"min": 10.0,  "max": 200.0, "unit": "s",     "desc": "Yaw reference low-pass time constant"},
                "dxMax":  {"min": 0.1,   "max": 5.0,   "unit": "m/s",   "desc": "Max surge reference velocity"},
                "dyMax":  {"min": 0.1,   "max": 5.0,   "unit": "m/s",   "desc": "Max sway reference velocity"},
                "dpsiMax":{"min": 0.01,  "max": 0.5,   "unit": "rad/s", "desc": "Max yaw reference rate"},
            },
        },
        {
            "name": "dp_controller",
            "fmu": "DPController",
            "case": "dp-ship",
            "step_size": 0.04,
            "role": "DP PID controller — surge/sway/yaw force demand",
            # PARAMS (21): Kpx, Kpy, Kppsi (P), Kdx, Kdy, Kdpsi (D), Kix, Kiy, Kipsi (I),
            #              ActivationTime, ComTimeStep, DebugMode, relTol, absTol, ...
            # INPUTS (15): x, y, psi, dx, dy, dpsi, x_ref, y_ref, psi_ref,
            #              dx_ref, dy_ref, dpsi_ref, reset1, reset2, reset3
            # OUTPUTS (3): Controlx, Controly, ControlMz
            "default_params": {},
            "tunable_params": {
                "Kpx":   {"min": 1e3, "max": 1e6, "unit": "N/m",      "desc": "DP surge proportional gain"},
                "Kpy":   {"min": 1e3, "max": 1e6, "unit": "N/m",      "desc": "DP sway proportional gain"},
                "Kppsi": {"min": 1e4, "max": 1e7, "unit": "N·m/rad",  "desc": "DP yaw proportional gain"},
                "Kdx":   {"min": 1e3, "max": 1e6, "unit": "N·s/m",    "desc": "DP surge derivative gain"},
                "Kdy":   {"min": 1e3, "max": 1e6, "unit": "N·s/m",    "desc": "DP sway derivative gain"},
                "Kdpsi": {"min": 1e4, "max": 1e7, "unit": "N·m·s/rad","desc": "DP yaw derivative gain"},
                "Kix":   {"min": 0.0, "max": 1e4, "unit": "N/(m·s)",  "desc": "DP surge integral gain"},
                "Kiy":   {"min": 0.0, "max": 1e4, "unit": "N/(m·s)",  "desc": "DP sway integral gain"},
                "Kipsi": {"min": 0.0, "max": 1e5, "unit": "N·m/rad·s","desc": "DP yaw integral gain"},
            },
        },
        {
            "name": "thrust_allocator",
            "fmu": "ThMPC",
            "case": "dp-ship",
            "step_size": 0.04,
            "role": "MPC-based thrust allocation — force demand → F1c/F2c/F3c + azimuth",
            # PARAMS (40): Fmax, MaxIter, QF1x/y, QF2x/y, QF3, QeMz/QeMzN, Qex/QexN, Qey/QeyN,
            #              Qu1x/y, Qu2x/y, Qu3, QuF, biasAngDeg, dalphaMax, dt, uFmax,
            #              x1/x2/x3, y1/y2/y3, F1x0/F1y0/F2x0/F2y0/F30, ...
            # INPUTS (3): refx, refy, refMz
            # OUTPUTS (8): F1c, F2c, F3c, Fxg, Fyg, Mzg, alpha1, alpha2
            "default_params": {},
            "tunable_params": {
                "Fmax":   {"min": 1e4,  "max": 1e6,  "unit": "N",  "desc": "Max thruster force"},
                "Qex":    {"min": 1.0,  "max": 1e4,  "unit": "-",  "desc": "MPC surge error weight"},
                "Qey":    {"min": 1.0,  "max": 1e4,  "unit": "-",  "desc": "MPC sway error weight"},
                "QeMz":   {"min": 1.0,  "max": 1e4,  "unit": "-",  "desc": "MPC yaw moment error weight"},
                "Qu1x":   {"min": 0.01, "max": 100.0,"unit": "-",  "desc": "MPC thruster 1 surge cost weight"},
                "Qu1y":   {"min": 0.01, "max": 100.0,"unit": "-",  "desc": "MPC thruster 1 sway cost weight"},
                "QuF":    {"min": 0.01, "max": 100.0,"unit": "-",  "desc": "MPC total thrust cost weight"},
            },
        },
        {
            "name": "ship",
            "fmu": "OSOM",
            "case": "dp-ship",
            "step_size": 0.04,
            "role": "Open Ship Object Model — 3-DOF vessel dynamics",
            # PARAMS (158): g, rho_w, m_ship, L_ship, B_ship, H_ship,
            #               Current.vcN, Current.vcE, m_1/2/3, rb_*/r*_* (geometry),
            #               Inertia.*, MR.r[*]/Cd, R.r[*], Submodel6.* (arm controller),
            #               Thrusters.*, Waves.* (wave params), TransferFunction2.*
            # INPUTS (5): input[1], input[2] (azimuth), Thrust_d[1~3]
            # OUTPUTS (6): q[1](x), q[2](y), q[3](psi), reset[1~3]
            "default_params": {},
            "tunable_params": {
                "Current.vcN":    {"min": -1.0, "max": 1.0, "unit": "m/s", "desc": "Ocean current velocity (North)"},
                "Current.vcE":    {"min": -1.0, "max": 1.0, "unit": "m/s", "desc": "Ocean current velocity (East)"},
                "Waves.H":        {"min": 0.0,  "max": 5.0, "unit": "m",   "desc": "Irregular wave height"},
                "Waves.T":        {"min": 4.0,  "max": 20.0,"unit": "s",   "desc": "Dominant wave period"},
            },
        },
        {
            "name": "observer",
            "fmu": "NLPobserver",
            "case": "dp-ship",
            "step_size": 0.04,
            "role": "Nonlinear Passive Observer — filters noise, estimates velocity",
            # PARAMS (27): w01/02/03 (bandwidth), lambda1/2/3 (damping),
            #              EulerInit[1~3], invM.m_ship/Ca/Iz_ship (ship model copy),
            #              invT.T1/T2/T3, K1.xi1~3, wc1~3, K3.k31~33, K4.k41~43
            # INPUTS (6): tau[1~3] (thrust), y[1~3] (position measurement)
            # OUTPUTS (6): v_hat[1~3] (velocity), y_hat[1~3] (filtered position)
            "default_params": {},
            "tunable_params": {
                "w01":     {"min": 0.05, "max": 2.0, "unit": "rad/s", "desc": "Observer bandwidth — surge"},
                "w02":     {"min": 0.05, "max": 2.0, "unit": "rad/s", "desc": "Observer bandwidth — sway"},
                "w03":     {"min": 0.05, "max": 2.0, "unit": "rad/s", "desc": "Observer bandwidth — yaw"},
                "lambda1": {"min": 0.1,  "max": 5.0, "unit": "-",     "desc": "Observer damping — surge"},
                "lambda2": {"min": 0.1,  "max": 5.0, "unit": "-",     "desc": "Observer damping — sway"},
                "lambda3": {"min": 0.1,  "max": 5.0, "unit": "-",     "desc": "Observer damping — yaw"},
                "wc1":     {"min": 0.01, "max": 1.0, "unit": "rad/s", "desc": "Observer cutoff — surge"},
                "wc2":     {"min": 0.01, "max": 1.0, "unit": "rad/s", "desc": "Observer cutoff — sway"},
                "wc3":     {"min": 0.01, "max": 1.0, "unit": "rad/s", "desc": "Observer cutoff — yaw"},
            },
        },
    ],

    # ── connections (실제 FMU 변수명 기준) ──────────────────────────────────
    "connections": [
        # thrust_allocator → ship (local thrust forces + azimuth)
        {"from_fmu": "thrust_allocator", "from_var": "F1c",
         "to_fmu":   "ship",             "to_var":   "Thrust_d[1]"},
        {"from_fmu": "thrust_allocator", "from_var": "F2c",
         "to_fmu":   "ship",             "to_var":   "Thrust_d[2]"},
        {"from_fmu": "thrust_allocator", "from_var": "F3c",
         "to_fmu":   "ship",             "to_var":   "Thrust_d[3]"},
        {"from_fmu": "thrust_allocator", "from_var": "alpha1",
         "to_fmu":   "ship",             "to_var":   "input[1]"},
        {"from_fmu": "thrust_allocator", "from_var": "alpha2",
         "to_fmu":   "ship",             "to_var":   "input[2]"},

        # ship → observer (raw position)
        {"from_fmu": "ship",     "from_var": "q[1]",
         "to_fmu":   "observer", "to_var":   "y[1]"},
        {"from_fmu": "ship",     "from_var": "q[2]",
         "to_fmu":   "observer", "to_var":   "y[2]"},
        {"from_fmu": "ship",     "from_var": "q[3]",
         "to_fmu":   "observer", "to_var":   "y[3]"},

        # ship → dp_controller (reset signals)
        {"from_fmu": "ship",         "from_var": "reset[1]",
         "to_fmu":   "dp_controller","to_var":   "reset1"},
        {"from_fmu": "ship",         "from_var": "reset[2]",
         "to_fmu":   "dp_controller","to_var":   "reset2"},
        {"from_fmu": "ship",         "from_var": "reset[3]",
         "to_fmu":   "dp_controller","to_var":   "reset3"},

        # thrust_allocator → observer (global thrust feedback)
        {"from_fmu": "thrust_allocator", "from_var": "Fxg",
         "to_fmu":   "observer",         "to_var":   "tau[1]"},
        {"from_fmu": "thrust_allocator", "from_var": "Fyg",
         "to_fmu":   "observer",         "to_var":   "tau[2]"},
        {"from_fmu": "thrust_allocator", "from_var": "Mzg",
         "to_fmu":   "observer",         "to_var":   "tau[3]"},

        # observer → dp_controller (filtered position + velocity)
        {"from_fmu": "observer",     "from_var": "y_hat[1]",
         "to_fmu":   "dp_controller","to_var":   "x"},
        {"from_fmu": "observer",     "from_var": "y_hat[2]",
         "to_fmu":   "dp_controller","to_var":   "y"},
        {"from_fmu": "observer",     "from_var": "y_hat[3]",
         "to_fmu":   "dp_controller","to_var":   "psi"},
        {"from_fmu": "observer",     "from_var": "v_hat[1]",
         "to_fmu":   "dp_controller","to_var":   "dx"},
        {"from_fmu": "observer",     "from_var": "v_hat[2]",
         "to_fmu":   "dp_controller","to_var":   "dy"},
        {"from_fmu": "observer",     "from_var": "v_hat[3]",
         "to_fmu":   "dp_controller","to_var":   "dpsi"},

        # reference_generator → dp_controller (position + velocity reference)
        {"from_fmu": "reference_generator", "from_var": "x_ref",
         "to_fmu":   "dp_controller",       "to_var":   "x_ref"},
        {"from_fmu": "reference_generator", "from_var": "y_ref",
         "to_fmu":   "dp_controller",       "to_var":   "y_ref"},
        {"from_fmu": "reference_generator", "from_var": "psi_ref",
         "to_fmu":   "dp_controller",       "to_var":   "psi_ref"},
        {"from_fmu": "reference_generator", "from_var": "dx_ref",
         "to_fmu":   "dp_controller",       "to_var":   "dx_ref"},
        {"from_fmu": "reference_generator", "from_var": "dy_ref",
         "to_fmu":   "dp_controller",       "to_var":   "dy_ref"},
        {"from_fmu": "reference_generator", "from_var": "dpsi_ref",
         "to_fmu":   "dp_controller",       "to_var":   "dpsi_ref"},

        # dp_controller → thrust_allocator (force demand)
        {"from_fmu": "dp_controller",    "from_var": "Controlx",
         "to_fmu":   "thrust_allocator", "to_var":   "refx"},
        {"from_fmu": "dp_controller",    "from_var": "Controly",
         "to_fmu":   "thrust_allocator", "to_var":   "refy"},
        {"from_fmu": "dp_controller",    "from_var": "ControlMz",
         "to_fmu":   "thrust_allocator", "to_var":   "refMz"},
    ],

    "sum_connections": [],

    # reference_generator waypoint 고정: 현재 위치(x=0, y=0) 유지
    "control": {
        "reference_generator": {
            "x_wp": 0.0, "y_wp": 0.0, "psi_wp": 0.0,
            "x_tp": 0.0, "y_tp": 0.0, "psi_tp": 0.0,
        },
    },

    "metric": {
        "depth_fmu":       "ship",
        "depth_var":       "q[1]",      # x-position [m]
        "target_depth":    None,         # 전 구간 안정화 모드
        "equilibrium":     0.0,          # 목표 x = 0
        "power_fmu":       "ship",
        "power_var":       "q[1]",
        "power_limit_kw":  2.0,          # 허용 위치 편차 [m] (kw 필드 재활용)
        "failure_penalty": 0.3,
        "success_mode":    "abs_mean",
    },
    "step_size":       0.04,
    "max_duration":    200.0,
    "depth_tolerance": 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 카탈로그 및 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "construction_vessel_full": CONSTRUCTION_VESSEL_FULL,
    "lars_recovery":            LARS_RECOVERY,
    "dp_ship_station":          DP_SHIP_STATION,
}


def get_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name!r}. Available: {list(SCENARIOS.keys())}")
    sdef = copy.deepcopy(SCENARIOS[name])
    profile = os.getenv("SIM_PROTOCOL_PROFILE", "").strip().lower()
    if not profile:
        return sdef

    # v2 전용 난이도/일반화 프로파일
    if name == "construction_vessel_full":
        metric = sdef.get("metric", {})
        if profile in ("v2_id", "v2_ood"):
            metric["failure_penalty"] = 0.8
            metric["power_limit_kw"] = 650.0 if profile == "v2_id" else 550.0
            sdef["max_duration"] = 360.0 if profile == "v2_id" else 420.0
            sdef["depth_tolerance"] = 0.8 if profile == "v2_id" else 0.6
            # 더 강한 외란 기본치
            for f in sdef.get("fmus", []):
                if f.get("name") == "wave_model":
                    f.get("default_params", {})["wave_height"] = 3.8 if profile == "v2_id" else 4.4
                    f.get("default_params", {})["peak_frequency"] = 1.1 if profile == "v2_id" else 1.35
                    f.get("default_params", {})["k_down"] = 2.5 if profile == "v2_id" else 3.2
            sdef["metric"] = metric

    if name == "lars_recovery":
        metric = sdef.get("metric", {})
        if profile in ("v2_id", "v2_ood"):
            metric["failure_penalty"] = 0.9 if profile == "v2_ood" else 0.75
            metric["power_limit_raw"] = 50.0 if profile == "v2_id" else 45.0
            sdef["max_duration"] = 18.0 if profile == "v2_id" else 22.0
            sdef["depth_tolerance"] = 3.0 if profile == "v2_id" else 2.0
            sdef["metric"] = metric

    # 논문 프로토콜 `paper`: Construction Vessel만 대상. 전력 한계·시간·깊이 허용을 빡세게 해
    # viol/step과 목표 미도달(success=False)이 자주 나오게 하고, G0/G1은 넓은 이산 격자에서 불리하게 둠.
    if profile == "paper" and name == "construction_vessel_full":
        metric = dict(sdef.get("metric", {}))
        metric["failure_penalty"] = 1.08
        metric["power_limit_kw"] = 420.0
        sdef["metric"] = metric
        sdef["max_duration"] = 210.0
        sdef["depth_tolerance"] = 0.42
        sdef["trial_timeout"] = 900.0
        for f in sdef.get("fmus", []):
            if f.get("name") == "wave_model":
                dp = f.setdefault("default_params", {})
                dp["wave_height"] = 4.15
                dp["peak_frequency"] = 1.25
                dp["k_down"] = 2.85
                dp["k_north"] = 2.95
                dp["k_east"] = 2.95
            if f.get("name") == "winch":
                dp = f.setdefault("default_params", {})
                dp["K_p"] = 260.0
                dp["K_d"] = 2100.0
                dp["K_i"] = 6.8

    return sdef


def list_scenarios() -> list:
    result = []
    for key, s in SCENARIOS.items():
        fmu_list = [f["name"] + "(" + f["fmu"] + ")" for f in s["fmus"]]
        n_tunable = sum(len(f["tunable_params"]) for f in s["fmus"])
        result.append({
            "id":        key,
            "name":      s["display_name"],
            "fmus":      fmu_list,
            "n_fmus":    len(s["fmus"]),
            "n_tunable": n_tunable,
        })
    return result
