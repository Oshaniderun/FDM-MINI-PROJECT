"""
Automated Backend & Inference Integration Tests
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Group: 05 - 'Cognita'
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200, model loaded, and all 6 target classes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "target_classes" in data
    assert len(data["target_classes"]) == 6
    assert "Power Failure" in data["target_classes"]
    assert "No failure" in data["target_classes"]


def test_frontend_index_served():
    """Verify that root GET / serves the HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Predictive Maintenance Diagnostic System" in response.text



def test_schema_endpoint():
    """Verify /api/v1/schema returns valid categorical and sensor definitions and presets."""
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    data = response.json()
    assert "categorical_fields" in data
    assert "sensor_fields" in data
    assert "preset_scenarios" in data
    assert set(data["categorical_fields"]["type"]["allowed_values"]) == {"L", "M", "H"}
    assert set(data["categorical_fields"]["control"]["allowed_values"]) == {"A", "B", "C"}


def test_predict_normal_operation():
    """Verify prediction on a standard healthy operating payload."""
    payload = {
        "type": "L",
        "control": "A",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 38.0,
        "tool_wear_min": 30.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "No failure"
    assert data["is_failure"] is False
    assert data["severity"] == "NORMAL"
    assert "confidence" in data
    assert sum(data["class_probabilities"].values()) == pytest.approx(100.0, abs=1.0)
    assert data["computed_physics"]["temp_difference_k"] == 10.0
    assert data["computed_physics"]["missing_sensors_count"] == 0


def test_predict_power_failure():
    """Verify prediction on a high-torque high-power failure condition."""
    payload = {
        "type": "L",
        "control": "B",
        "air_temperature_k": None,
        "process_temperature_k": None,
        "rotational_speed_rpm": 1350.0,
        "torque_nm": 68.2,
        "tool_wear_min": None,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "Power Failure"
    assert data["is_failure"] is True
    assert data["severity"] == "CRITICAL"
    assert data["confidence"] > 50.0
    assert data["computed_physics"]["mechanical_power_w"] > 9000.0
    assert data["computed_physics"]["missing_sensors_count"] == 3


def test_predict_heat_dissipation_failure():
    """Verify prediction on an HDF failure condition (Delta T < 8.6 K at low RPM)."""
    payload = {
        "type": "M",
        "control": "A",
        "air_temperature_k": 303.0,
        "process_temperature_k": 309.0,
        "rotational_speed_rpm": 1320.0,
        "torque_nm": 48.0,
        "tool_wear_min": 80.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "Heat Dissipation Failure"
    assert data["is_failure"] is True
    assert data["severity"] == "CRITICAL"
    assert data["computed_physics"]["temp_difference_k"] == 6.0


def test_predict_missing_telemetry_control_mode_b():
    """Verify that the model successfully predicts even when multiple sensors are missing (NaN)."""
    payload = {
        "type": "M",
        "control": "B",
        "air_temperature_k": None,
        "process_temperature_k": None,
        "rotational_speed_rpm": 1550.0,
        "torque_nm": 36.0,
        "tool_wear_min": None,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["computed_physics"]["missing_sensors_count"] == 3


def test_validation_invalid_type_rejected():
    """Verify that an illegal machine type returns HTTP 422 Unprocessable Entity."""
    payload = {
        "type": "Z",  # Invalid type
        "control": "A",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_validation_invalid_control_rejected():
    """Verify that an illegal control mode returns HTTP 422."""
    payload = {
        "type": "L",
        "control": "X",  # Invalid control
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_validation_missing_required_field():
    """Verify that omitting a required field (e.g. control) returns HTTP 422."""
    payload = {
        "type": "L",
        # missing control
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_validation_out_of_range_numeric():
    """Verify that absurd out-of-range sensor readings return HTTP 422."""
    payload = {
        "type": "L",
        "control": "A",
        "air_temperature_k": 50.0,  # Below minimum 280 K
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_validation_thermodynamic_violation():
    """Verify that process temperature significantly below air temp returns HTTP 422."""
    payload = {
        "type": "L",
        "control": "A",
        "air_temperature_k": 320.0,
        "process_temperature_k": 290.0,  # Implausible process cooler than ambient
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
