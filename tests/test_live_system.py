"""
Live End-to-End System Integration Test Script
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Group: 05 - 'Cognita'

Executes rigorous end-to-end verification against the live running server:
1. Static asset delivery (HTML, CSS, JS)
2. Health & metadata endpoints
3. All 6 preset scenarios (Normal, PWF, HDF, OSF, TWF, Missing Data)
4. Empirical validation against actual test.csv rows from the held-out partition
5. Full error & validation handling (boundary limits, invalid types, thermodynamic rules)
"""

import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np

BASE_URL = "http://127.0.0.1:8000"


def make_request(path: str, method: str = "GET", data: dict = None):
    url = f"{BASE_URL}{path}"
    headers = {}
    encoded_data = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        encoded_data = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body, resp.headers
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, body, e.headers


def run_comprehensive_tests():
    print("=" * 80)
    print("LIVE END-TO-END INFERENCE & WEB STACK VERIFICATION")
    print(f"Target Server: {BASE_URL}")
    print("=" * 80)

    test_results = []

    # 1. Static Asset Verification
    print("\n[TEST 1] Verifying Static Web Assets Delivery...")
    status, html, _ = make_request("/")
    assert status == 200 and "Predictive Maintenance Diagnostic System" in html
    print("  [PASS] GET / -> 200 OK (index.html served, size: %d bytes)" % len(html))

    status, css, _ = make_request("/static/style.css")
    assert status == 200 and "--bg-main" in css
    print("  [PASS] GET /static/style.css -> 200 OK (style.css served, size: %d bytes)" % len(css))

    status, js, _ = make_request("/static/app.js")
    assert status == 200 and "renderPredictionResult" in js
    print("  [PASS] GET /static/app.js -> 200 OK (app.js served, size: %d bytes)" % len(js))
    test_results.append(("Static Asset Delivery (HTML/CSS/JS)", "GET /, /static/*", "HTTP 200 with code", "PASS"))

    # 2. Health & Schema
    print("\n[TEST 2] Verifying System Health & Metadata Endpoints...")
    status, body, _ = make_request("/health")
    health = json.loads(body)
    assert status == 200 and health["status"] == "healthy" and health["model_loaded"] is True
    print("  [PASS] GET /health -> 200 OK | Model: %s" % health["champion_model"])
    test_results.append(("System Health Check", "GET /health", "status='healthy', model_loaded=True", "PASS"))

    status, body, _ = make_request("/api/v1/schema")
    schema = json.loads(body)
    assert status == 200 and "preset_scenarios" in schema
    print("  [PASS] GET /api/v1/schema -> 200 OK (%d Presets available)" % len(schema["preset_scenarios"]))
    test_results.append(("Schema & Preset Discovery", "GET /api/v1/schema", "HTTP 200 with 6 presets", "PASS"))

    # 3. All 6 Preset Scenarios
    print("\n[TEST 3] Testing All 6 Preset Scenarios on Live ML Pipeline...")
    presets = schema["preset_scenarios"]
    for key, p_info in presets.items():
        payload = p_info["payload"]
        status, body, _ = make_request("/api/v1/predict", method="POST", data=payload)
        assert status == 200
        res = json.loads(body)
        print(f"  Preset '{p_info['name']}':")
        print(f"    -> Predicted Class:    '{res['predicted_class']}' ({res['confidence']:.1f}% confidence)")
        print(f"    -> Severity Level:     {res['severity']}")
        print(f"    -> Derived Physics:    Delta_T={res['computed_physics']['temp_difference_k']} K, Power={res['computed_physics']['mechanical_power_w']} W, Overstrain={res['computed_physics']['overstrain_product']}")
        print(f"    -> Unmeasured Sensors: {res['computed_physics']['missing_sensors_count']} / 5")
        test_results.append((f"Preset: {p_info['name']}", f"POST /api/v1/predict ({key})", f"Predicted: {res['predicted_class']}", "PASS"))

    # 4. Realistic Samples directly from held-out test.csv
    print("\n[TEST 4] Testing Empirical Ground-Truth Samples from held-out test.csv...")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Sample a normal instance
    normal_row = test_df[test_df["Diagnostic"] == "No failure"].iloc[0]
    normal_payload = {
        "type": normal_row["Type"],
        "control": normal_row["Control"],
        "air_temperature_k": None if pd.isna(normal_row["Air temperature (K)"]) else float(normal_row["Air temperature (K)"]),
        "process_temperature_k": None if pd.isna(normal_row["Process temperature (K)"]) else float(normal_row["Process temperature (K)"]),
        "rotational_speed_rpm": None if pd.isna(normal_row["Rotational speed (rpm)"]) else float(normal_row["Rotational speed (rpm)"]),
        "torque_nm": None if pd.isna(normal_row["Torque (Nm)"]) else float(normal_row["Torque (Nm)"]),
        "tool_wear_min": None if pd.isna(normal_row["Tool wear (min)"]) else float(normal_row["Tool wear (min)"]),
    }
    status, body, _ = make_request("/api/v1/predict", method="POST", data=normal_payload)
    res = json.loads(body)
    print("  Test Dataset Normal Sample (Row 0):")
    print(f"    Ground Truth: 'No failure' | Model Prediction: '{res['predicted_class']}' ({res['confidence']:.1f}%)")
    assert res["predicted_class"] == "No failure"
    test_results.append(("Test Partition Normal Sample", "Row from test.csv", "Predicted: 'No failure'", "PASS"))

    # Sample a PWF instance
    pwf_rows = test_df[test_df["Diagnostic"] == "Power Failure"]
    if len(pwf_rows) > 0:
        pwf_row = pwf_rows.iloc[0]
        pwf_payload = {
            "type": pwf_row["Type"],
            "control": pwf_row["Control"],
            "air_temperature_k": None if pd.isna(pwf_row["Air temperature (K)"]) else float(pwf_row["Air temperature (K)"]),
            "process_temperature_k": None if pd.isna(pwf_row["Process temperature (K)"]) else float(pwf_row["Process temperature (K)"]),
            "rotational_speed_rpm": None if pd.isna(pwf_row["Rotational speed (rpm)"]) else float(pwf_row["Rotational speed (rpm)"]),
            "torque_nm": None if pd.isna(pwf_row["Torque (Nm)"]) else float(pwf_row["Torque (Nm)"]),
            "tool_wear_min": None if pd.isna(pwf_row["Tool wear (min)"]) else float(pwf_row["Tool wear (min)"]),
        }
        status, body, _ = make_request("/api/v1/predict", method="POST", data=pwf_payload)
        res = json.loads(body)
        print("  Test Dataset Power Failure Sample:")
        print(f"    Ground Truth: 'Power Failure' | Model Prediction: '{res['predicted_class']}' ({res['confidence']:.1f}%)")
        assert res["predicted_class"] == "Power Failure"
        test_results.append(("Test Partition PWF Sample", "Row from test.csv", "Predicted: 'Power Failure'", "PASS"))

    # Sample an HDF instance
    hdf_rows = test_df[test_df["Diagnostic"] == "Heat Dissipation Failure"]
    if len(hdf_rows) > 0:
        hdf_row = hdf_rows.iloc[0]
        hdf_payload = {
            "type": hdf_row["Type"],
            "control": hdf_row["Control"],
            "air_temperature_k": None if pd.isna(hdf_row["Air temperature (K)"]) else float(hdf_row["Air temperature (K)"]),
            "process_temperature_k": None if pd.isna(hdf_row["Process temperature (K)"]) else float(hdf_row["Process temperature (K)"]),
            "rotational_speed_rpm": None if pd.isna(hdf_row["Rotational speed (rpm)"]) else float(hdf_row["Rotational speed (rpm)"]),
            "torque_nm": None if pd.isna(hdf_row["Torque (Nm)"]) else float(hdf_row["Torque (Nm)"]),
            "tool_wear_min": None if pd.isna(hdf_row["Tool wear (min)"]) else float(hdf_row["Tool wear (min)"]),
        }
        status, body, _ = make_request("/api/v1/predict", method="POST", data=hdf_payload)
        res = json.loads(body)
        print("  Test Dataset Heat Dissipation Failure Sample:")
        print(f"    Ground Truth: 'Heat Dissipation Failure' | Model Prediction: '{res['predicted_class']}' ({res['confidence']:.1f}%)")
        assert res["predicted_class"] == "Heat Dissipation Failure"
        test_results.append(("Test Partition HDF Sample", "Row from test.csv", "Predicted: 'Heat Dissipation Failure'", "PASS"))

    # 5. Invalid / Boundary Input Handling
    print("\n[TEST 5] Testing Input Validation & Boundary Error Handling...")
    invalid_cases = [
        ("Invalid Machine Type ('Z')", {"type": "Z", "control": "A", "air_temperature_k": 300.0}, 422),
        ("Invalid Control Mode ('X')", {"type": "L", "control": "X", "air_temperature_k": 300.0}, 422),
        ("Missing Required Type", {"control": "A", "air_temperature_k": 300.0}, 422),
        ("Air Temp Below Absolute Minimum (200 K)", {"type": "L", "control": "A", "air_temperature_k": 200.0}, 422),
        ("Speed Exceeds Maximum (6000 RPM)", {"type": "L", "control": "A", "rotational_speed_rpm": 6000.0}, 422),
        ("Thermodynamic Inconsistency (Process 290 K < Air 310 K)", {
            "type": "L", "control": "A", "air_temperature_k": 310.0, "process_temperature_k": 290.0
        }, 422),
    ]

    for name, payload, expected_code in invalid_cases:
        status, body, _ = make_request("/api/v1/predict", method="POST", data=payload)
        assert status == expected_code
        print(f"  [PASS] {name} -> Correctly rejected with HTTP {status}")
        test_results.append((name, f"Payload: {list(payload.keys())}", f"HTTP {expected_code} Error", "PASS"))

    # 6. Complete Missing Sensors Test (Only Type & Control provided)
    print("\n[TEST 6] Testing Complete Missing Telemetry (All 5 Sensors Null)...")
    all_null_payload = {
        "type": "L",
        "control": "A",
        "air_temperature_k": None,
        "process_temperature_k": None,
        "rotational_speed_rpm": None,
        "torque_nm": None,
        "tool_wear_min": None,
    }
    status, body, _ = make_request("/api/v1/predict", method="POST", data=all_null_payload)
    assert status == 200
    res = json.loads(body)
    print(f"  [PASS] All 5 sensors null handled cleanly -> Predicted: '{res['predicted_class']}' ({res['confidence']:.1f}%)")
    test_results.append(("All Sensors Unmeasured (MAR)", "Only Type & Control provided", "Handled via training medians", "PASS"))

    print("\n" + "=" * 80)
    print("ALL LIVE TESTS PASSED SUCCESSFULLY! SUMMARY OF RESULTS:")
    print("=" * 80)
    df_res = pd.DataFrame(test_results, columns=["Test Description", "Input / Route", "Expected / Actual Output", "Status"])
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_comprehensive_tests()
