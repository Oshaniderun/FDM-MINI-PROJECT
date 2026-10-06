"""
Prediction Inference Service
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Group: 05 - 'Cognita'
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure repository root is on sys.path for unpickling custom transformers
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure src.features is imported in namespace
from src import features  # noqa: F401
from backend.schemas import MachineInputSchema, PredictionResponseSchema, ComputedPhysicsSchema
from backend.domain import CLASS_DIAGNOSTICS, compute_physics_telemetry

MODEL_PATH = REPO_ROOT / "models" / "champion_pipeline.joblib"


class DiagnosticService:
    """Singleton service wrapping the serialized champion inference pipeline."""

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiagnosticService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Champion pipeline file not found at {MODEL_PATH}. "
                "Ensure Stage 6-8 modeling pipeline has been executed."
            )
        self._pipeline = joblib.load(MODEL_PATH)
        self.classes_ = list(self._pipeline.classes_)

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_model()
        return self._pipeline

    def predict(self, item: MachineInputSchema) -> PredictionResponseSchema:
        """Transforms validated user input and runs inference through the champion pipeline.

        Args:
            item: Validated Pydantic MachineInputSchema.

        Returns:
            PredictionResponseSchema with full probabilities, physics, and recommendations.
        """
        # Map Pydantic input to the exact raw dataframe columns expected by champion_pipeline:
        # Expected: ['Control', 'Type', 'Air temperature (K)', 'Process temperature (K)',
        #            'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)']
        raw_dict = {
            "Control": str(item.control),
            "Type": str(item.type),
            "Air temperature (K)": np.nan if item.air_temperature_k is None else float(item.air_temperature_k),
            "Process temperature (K)": np.nan if item.process_temperature_k is None else float(item.process_temperature_k),
            "Rotational speed (rpm)": np.nan if item.rotational_speed_rpm is None else float(item.rotational_speed_rpm),
            "Torque (Nm)": np.nan if item.torque_nm is None else float(item.torque_nm),
            "Tool wear (min)": np.nan if item.tool_wear_min is None else float(item.tool_wear_min),
        }
        input_df = pd.DataFrame([raw_dict])

        # Execute full Scikit-Learn Pipeline
        predicted_class = str(self.pipeline.predict(input_df)[0])
        probabilities_raw = self.pipeline.predict_proba(input_df)[0]

        # Format probability dictionary
        prob_dict = {
            cls_name: round(float(prob) * 100.0, 2)
            for cls_name, prob in zip(self.classes_, probabilities_raw)
        }

        # Confidence of the predicted class
        confidence = prob_dict.get(predicted_class, round(float(np.max(probabilities_raw)) * 100.0, 2))

        # Compute transparent physics telemetry
        physics = compute_physics_telemetry(
            air_temp_k=item.air_temperature_k,
            proc_temp_k=item.process_temperature_k,
            rot_speed_rpm=item.rotational_speed_rpm,
            torque_nm=item.torque_nm,
            tool_wear_min=item.tool_wear_min,
        )

        # Retrieve diagnostic information
        diag_info = CLASS_DIAGNOSTICS.get(
            predicted_class,
            {
                "title": predicted_class,
                "severity": "WARNING",
                "diagnosis": f"Predicted operating condition: {predicted_class}.",
                "recommended_action": "Review machine telemetry with operating technician.",
            },
        )

        is_failure = (predicted_class != "No failure")

        input_summary = {
            "Type": item.type,
            "Control": item.control,
            "Air temperature (K)": item.air_temperature_k,
            "Process temperature (K)": item.process_temperature_k,
            "Rotational speed (rpm)": item.rotational_speed_rpm,
            "Torque (Nm)": item.torque_nm,
            "Tool wear (min)": item.tool_wear_min,
        }

        return PredictionResponseSchema(
            predicted_class=predicted_class,
            is_failure=is_failure,
            confidence=confidence,
            class_probabilities=prob_dict,
            computed_physics=ComputedPhysicsSchema(**physics),
            diagnosis=diag_info["diagnosis"],
            recommended_action=diag_info["recommended_action"],
            severity=diag_info["severity"],
            input_summary=input_summary,
        )


diagnostic_service = DiagnosticService()
