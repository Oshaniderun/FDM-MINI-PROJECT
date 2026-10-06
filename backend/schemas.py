"""
Pydantic Schemas for AI4I-PMDI Predictive Maintenance System
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Group: 05 - 'Cognita'
"""

from typing import Dict, Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class MachineInputSchema(BaseModel):
    """Input payload representing machine operational telemetry."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "type": "L",
                "control": "A",
                "air_temperature_k": 300.5,
                "process_temperature_k": 310.2,
                "rotational_speed_rpm": 1450.0,
                "torque_nm": 42.0,
                "tool_wear_min": 105.0,
            }
        },
    )

    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Product quality variant: 'L' (Low - 50%), 'M' (Medium - 30%), 'H' (High - 20%).",
        alias="Type",
    )
    control: Literal["A", "B", "C"] = Field(
        ...,
        description="Operational multiplexer mode governing active sensor telemetry channels ('A', 'B', or 'C').",
        alias="Control",
    )
    air_temperature_k: Optional[float] = Field(
        default=None,
        ge=280.0,
        le=340.0,
        description="Ambient air temperature in Kelvin (valid range: 280.0 - 340.0 K; null if unmeasured).",
        alias="Air temperature (K)",
    )
    process_temperature_k: Optional[float] = Field(
        default=None,
        ge=280.0,
        le=350.0,
        description="Internal machine process temperature in Kelvin (valid range: 280.0 - 350.0 K; null if unmeasured).",
        alias="Process temperature (K)",
    )
    rotational_speed_rpm: Optional[float] = Field(
        default=None,
        ge=800.0,
        le=4000.0,
        description="Spindle rotational speed in revolutions per minute (valid range: 800 - 4000 rpm; null if unmeasured).",
        alias="Rotational speed (rpm)",
    )
    torque_nm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=150.0,
        description="Milling cutting torque in Newton-meters (valid range: 0.0 - 150.0 Nm; null if unmeasured).",
        alias="Torque (Nm)",
    )
    tool_wear_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=350.0,
        description="Cumulative cutting tool wear in minutes (valid range: 0.0 - 350.0 min; null if unmeasured).",
        alias="Tool wear (min)",
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_type_case(cls, v: Any) -> str:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in {"L", "M", "H"}:
                return v_upper
        return v

    @field_validator("control", mode="before")
    @classmethod
    def validate_control_case(cls, v: Any) -> str:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in {"A", "B", "C"}:
                return v_upper
        return v

    @model_validator(mode="after")
    def validate_thermodynamic_plausibility(self) -> "MachineInputSchema":
        """Asserts that process temperature is not absurdly lower than ambient temperature."""
        if self.air_temperature_k is not None and self.process_temperature_k is not None:
            if self.process_temperature_k < (self.air_temperature_k - 5.0):
                raise ValueError(
                    f"Thermodynamically implausible: Process temperature ({self.process_temperature_k} K) "
                    f"cannot be significantly lower than ambient Air temperature ({self.air_temperature_k} K)."
                )
        return self


class ComputedPhysicsSchema(BaseModel):
    """Physics-derived continuous variables computed from sensor telemetry."""
    temp_difference_k: Optional[float] = Field(
        default=None,
        description="Thermal gradient Delta T = Process Temp - Air Temp (Kelvin).",
    )
    mechanical_power_w: Optional[float] = Field(
        default=None,
        description="Spindle mechanical power in Watts = Torque * Speed * (2 * pi / 60).",
    )
    overstrain_product: Optional[float] = Field(
        default=None,
        description="Structural strain indicator = Tool wear * Torque (min * Nm).",
    )
    missing_sensors_count: int = Field(
        default=0,
        description="Total count of unmeasured / null sensor channels for this instance.",
    )


class PredictionResponseSchema(BaseModel):
    """Complete diagnostic prediction response."""
    predicted_class: str = Field(..., description="Predicted diagnostic condition.")
    is_failure: bool = Field(..., description="True if any failure mode is detected, False for normal operation.")
    confidence: float = Field(..., description="Model classification confidence percentage (0.0 - 100.0%).")
    class_probabilities: Dict[str, float] = Field(..., description="Probability breakdown across all 6 diagnostic classes.")
    computed_physics: ComputedPhysicsSchema = Field(..., description="Physics-based continuous operational indicators.")
    diagnosis: str = Field(..., description="Human-readable diagnostic interpretation of the machine state.")
    recommended_action: str = Field(..., description="Actionable industrial maintenance next-step.")
    severity: Literal["NORMAL", "WARNING", "CRITICAL"] = Field(..., description="Operational risk severity level.")
    input_summary: Dict[str, Any] = Field(..., description="Normalized echo of the received input parameters.")


class HealthResponseSchema(BaseModel):
    """Backend system health status schema."""
    status: str = "healthy"
    model_loaded: bool
    champion_model: str
    pipeline_version: str
    target_classes: list[str]
