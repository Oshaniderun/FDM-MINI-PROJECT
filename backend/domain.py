"""
Domain Logic, Diagnostic Interpretations, and Actionable Recommendations
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Group: 05 - 'Cognita'
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


CLASS_DIAGNOSTICS: Dict[str, Dict[str, str]] = {
    "No failure": {
        "title": "Normal Operation (Healthy)",
        "severity": "NORMAL",
        "diagnosis": (
            "The equipment is operating within standard thermal, mechanical, and tool-wear thresholds. "
            "No early indicators of impending mechanical or electrical failure were detected."
        ),
        "recommended_action": (
            "No immediate maintenance required. Clear machine for continued production schedule "
            "and maintain routine scheduled inspection intervals."
        ),
    },
    "Power Failure": {
        "title": "Spindle Power Failure (PWF)",
        "severity": "CRITICAL",
        "diagnosis": (
            "Spindle mechanical power is outside the safe operating envelope (< 3,500 W or > 9,000 W). "
            "This indicates either a severe drive overload, high cutting resistance stall, or underpower drive fault."
        ),
        "recommended_action": (
            "EMERGENCY HALT: Immediately shut down the spindle drive. Inspect the drive inverter, motor electrical "
            "supply, and check for cutter workpiece jamming or excessive feed speed."
        ),
    },
    "Heat Dissipation Failure": {
        "title": "Heat Dissipation Failure (HDF)",
        "severity": "CRITICAL",
        "diagnosis": (
            "Thermal dissipation gradient is critically insufficient (Delta T < 8.6 K) at low-to-moderate spindle speeds "
            "(<= 1,380 rpm). Heat generated during milling is not being evacuated from the cutting zone."
        ),
        "recommended_action": (
            "HALT SPINDLE: Inspect the liquid cooling jacket, coolant fluid level, and heat exchanger pump. "
            "Clean blocked coolant nozzles and verify thermal radiator airflow before restarting."
        ),
    },
    "Overstrain Failure": {
        "title": "Mechanical Overstrain Failure (OSF)",
        "severity": "CRITICAL",
        "diagnosis": (
            "The product of cumulative tool wear and cutting torque exceeds structural mechanical limits. "
            "The cutter is severely worn and experiencing high cutting resistance, risking catastrophic fracture."
        ),
        "recommended_action": (
            "IMMEDIATE TOOL CHANGE: Stop the current cycle and replace the worn cutter insert immediately. "
            "Check the workpiece surface for burring and reduce the depth of cut for the new insert."
        ),
    },
    "Tool Wear Failure": {
        "title": "Critical Tool Wear Failure (TWF)",
        "severity": "WARNING",
        "diagnosis": (
            "Cutter tool wear has reached the critical end-of-life threshold (200 - 240 minutes). "
            "Flank wear has degraded dimensional machining tolerances and cutting edge sharpness."
        ),
        "recommended_action": (
            "SCHEDULED TOOL REPLACEMENT: Complete the current sub-operation if safe, then swap out the worn milling tool. "
            "Log the tool usage cycle and inspect cutter runout."
        ),
    },
    "Random Failures": {
        "title": "Random Stochastic Anomaly (RNF)",
        "severity": "WARNING",
        "diagnosis": (
            "Anomalous physical disturbance detected without preceding classical sensor progression. "
            "Indicates potential workpiece material inclusions, structural vibration spikes, or loose fixturing."
        ),
        "recommended_action": (
            "MANUAL AUDIT: Pause automated cycle. Perform acoustic vibration audit, check workpiece clamping firmness, "
            "and inspect mechanical guideway lubrication."
        ),
    },
}


PRESET_SCENARIOS = {
    "normal": {
        "name": "Normal Healthy Operation",
        "description": "Standard balanced milling telemetry in Control Mode A with healthy sensors.",
        "payload": {
            "type": "L",
            "control": "A",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.5,
            "rotational_speed_rpm": 1520.0,
            "torque_nm": 38.5,
            "tool_wear_min": 45.0,
        },
    },
    "power_failure": {
        "name": "Power Failure (PWF Overload)",
        "description": "Extreme cutting torque and high RPM pushing spindle power beyond 9,000 W.",
        "payload": {
            "type": "L",
            "control": "B",
            "air_temperature_k": None,
            "process_temperature_k": None,
            "rotational_speed_rpm": 1350.0,
            "torque_nm": 68.2,
            "tool_wear_min": None,
        },
    },
    "heat_dissipation": {
        "name": "Heat Dissipation Failure (HDF)",
        "description": "Insufficient temperature difference (Delta T < 8.6 K) at low spindle speed (1320 RPM).",
        "payload": {
            "type": "M",
            "control": "A",
            "air_temperature_k": 303.2,
            "process_temperature_k": 309.8,
            "rotational_speed_rpm": 1320.0,
            "torque_nm": 48.0,
            "tool_wear_min": 70.0,
        },
    },
    "overstrain": {
        "name": "Overstrain Failure (OSF)",
        "description": "Severely worn tool combined with high cutting torque in Control Mode C.",
        "payload": {
            "type": "L",
            "control": "C",
            "air_temperature_k": None,
            "process_temperature_k": None,
            "rotational_speed_rpm": None,
            "torque_nm": 60.5,
            "tool_wear_min": 215.0,
        },
    },
    "tool_wear": {
        "name": "Tool Wear Failure (TWF)",
        "description": "Extended cutting duration with tool wear exceeding 230 minutes in Control Mode C.",
        "payload": {
            "type": "H",
            "control": "C",
            "air_temperature_k": None,
            "process_temperature_k": None,
            "rotational_speed_rpm": None,
            "torque_nm": 42.0,
            "tool_wear_min": 235.0,
        },
    },
    "control_b_missing": {
        "name": "Control Mode B (Missing Temperatures & Wear)",
        "description": "Simulates realistic sensor multiplexer B where temperatures & wear are unmeasured.",
        "payload": {
            "type": "M",
            "control": "B",
            "air_temperature_k": None,
            "process_temperature_k": None,
            "rotational_speed_rpm": 1600.0,
            "torque_nm": 35.0,
            "tool_wear_min": None,
        },
    },
}


def compute_physics_telemetry(
    air_temp_k: Optional[float],
    proc_temp_k: Optional[float],
    rot_speed_rpm: Optional[float],
    torque_nm: Optional[float],
    tool_wear_min: Optional[float],
) -> Dict[str, Any]:
    """Computes transparent domain physical indicators matching the pipeline's logic."""
    temp_diff = None
    if proc_temp_k is not None and air_temp_k is not None:
        temp_diff = round(proc_temp_k - air_temp_k, 2)

    power_w = None
    if torque_nm is not None and rot_speed_rpm is not None:
        omega = rot_speed_rpm * (2.0 * np.pi / 60.0)
        power_w = round(torque_nm * omega, 2)

    overstrain = None
    if tool_wear_min is not None and torque_nm is not None:
        overstrain = round(tool_wear_min * torque_nm, 2)

    missing_count = sum(
        1 for v in [air_temp_k, proc_temp_k, rot_speed_rpm, torque_nm, tool_wear_min] if v is None
    )

    return {
        "temp_difference_k": temp_diff,
        "mechanical_power_w": power_w,
        "overstrain_product": overstrain,
        "missing_sensors_count": missing_count,
    }
