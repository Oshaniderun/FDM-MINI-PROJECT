"""
FastAPI Application Entry Point
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Project: Predictive Maintenance Diagnostic System
Group: 05 - 'Cognita'
"""

import sys
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.schemas import (
    MachineInputSchema,
    PredictionResponseSchema,
    HealthResponseSchema,
)
from backend.service import diagnostic_service
from backend.domain import PRESET_SCENARIOS

app = FastAPI(
    title="Predictive Maintenance Diagnostic System API",
    description=(
        "Production-grade REST API for industrial CNC milling equipment failure classification "
        "using the AI4I-PMDI dataset. Developed for SLIIT IT3051 Fundamentals of Data Mining (Group 05 - 'Cognita')."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponseSchema, tags=["System Health"])
def health_check():
    """Returns the operational status of the service, model loading state, and target classes."""
    try:
        classes = diagnostic_service.classes_
        return HealthResponseSchema(
            status="healthy",
            model_loaded=True,
            champion_model="Tuned Random Forest Classifier (n_estimators=200, depth=10)",
            pipeline_version="1.0.0",
            target_classes=classes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unready or failed to load: {str(e)}",
        )


@app.get("/api/v1/schema", tags=["Metadata & Presets"])
def get_feature_schema() -> Dict[str, Any]:
    """Provides machine input metadata, valid ranges, units, and demonstration presets."""
    return {
        "dataset": "AI4I-PMDI Predictive Maintenance Dataset",
        "group": "05 - 'Cognita'",
        "categorical_fields": {
            "type": {
                "allowed_values": ["L", "M", "H"],
                "descriptions": {
                    "L": "Low Quality Variant (50% of machines)",
                    "M": "Medium Quality Variant (30% of machines)",
                    "H": "High Quality Variant (20% of machines)",
                },
            },
            "control": {
                "allowed_values": ["A", "B", "C"],
                "descriptions": {
                    "A": "Control Mode A (Monitors Temperatures & Rotational Speed)",
                    "B": "Control Mode B (Monitors Rotational Speed & Torque)",
                    "C": "Control Mode C (Monitors Torque & Tool Wear)",
                },
            },
        },
        "sensor_fields": {
            "air_temperature_k": {"unit": "Kelvin (K)", "typical_min": 295.0, "typical_max": 305.0},
            "process_temperature_k": {"unit": "Kelvin (K)", "typical_min": 305.0, "typical_max": 315.0},
            "rotational_speed_rpm": {"unit": "RPM", "typical_min": 1100.0, "typical_max": 2900.0},
            "torque_nm": {"unit": "Newton-meters (Nm)", "typical_min": 3.0, "typical_max": 80.0},
            "tool_wear_min": {"unit": "Minutes (min)", "typical_min": 0.0, "typical_max": 260.0},
        },
        "preset_scenarios": PRESET_SCENARIOS,
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Diagnostic Inference"],
)
def predict_machine_health(payload: MachineInputSchema) -> PredictionResponseSchema:
    """Predicts machine diagnostic condition and failure probability from operational telemetry.

    Applies the exact Scikit-Learn preprocessing pipeline fitted during training:
    - Domain feature engineering (Delta T, Spindle Power, Overstrain Product, Missing Count)
    - Median numerical imputation and missing-indicator generation
    - One-hot categorical encoding
    - Tuned Random Forest classification with vote probability distribution
    """
    try:
        return diagnostic_service.predict(payload)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data validation error: {str(ve)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution failure: {str(exc)}",
        )


# Mount frontend static directory if present
FRONTEND_DIR = REPO_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html not yet initialized"}
