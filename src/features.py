"""
Feature Engineering Transformers for AI4I-PMDI Predictive Maintenance Dataset
SLIIT IT3051 Fundamentals of Data Mining - Group 05 'Cognita'

This module contains scikit-learn compatible custom transformers for:
- Deriving domain physics continuous interactions:
  1. Temperature Difference: Process Temp (K) - Air Temp (K)
  2. Mechanical Power: Torque (Nm) * Rotational Speed (rpm) * (2 * pi / 60)
  3. Overstrain Product: Tool Wear (min) * Torque (Nm)
  4. Missing Sensor Count: Number of unmeasured sensors per instance
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn transformer to engineer physical domain interaction features.
    
    This transformer computes continuous domain representations based on physical
    principles of CNC milling degradation:
    - Thermal dissipation (Delta T)
    - Rotational power loading (Watts)
    - Structural strain (Tool Wear * Torque)
    - Telemetry sensor coverage count
    
    Can be used before or after numerical imputation.
    """
    
    def __init__(
        self,
        air_temp_col: str = "Air temperature (K)",
        proc_temp_col: str = "Process temperature (K)",
        rot_speed_col: str = "Rotational speed (rpm)",
        torque_col: str = "Torque (Nm)",
        tool_wear_col: str = "Tool wear (min)",
    ) -> None:
        self.air_temp_col = air_temp_col
        self.proc_temp_col = proc_temp_col
        self.rot_speed_col = rot_speed_col
        self.torque_col = torque_col
        self.tool_wear_col = tool_wear_col
        self.feature_names_in_ = None
        self.n_features_in_ = None

        
    def fit(self, X, y=None):
        """Fit transformer (stateless; stores input feature names)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
        self.n_features_in_ = len(self.feature_names_in_)
        return self
    
    def transform(self, X):
        """Transforms input matrix by appending continuous physics features.
        
        Args:
            X: pandas DataFrame or 2D numpy array.
            
        Returns:
            pandas DataFrame or numpy array with appended domain features.
        """
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
            
        # 1. Temperature Difference (Thermal Gradient Delta T in Kelvin)
        if self.proc_temp_col in df.columns and self.air_temp_col in df.columns:
            df["Temp_Difference"] = df[self.proc_temp_col] - df[self.air_temp_col]
        else:
            df["Temp_Difference"] = np.nan
            
        # 2. Mechanical Power (Watts) = Torque (Nm) * Angular Velocity (rad/s)
        if self.torque_col in df.columns and self.rot_speed_col in df.columns:
            omega_rad_s = df[self.rot_speed_col] * (2.0 * np.pi / 60.0)
            df["Mechanical_Power_W"] = df[self.torque_col] * omega_rad_s
        else:
            df["Mechanical_Power_W"] = np.nan
            
        # 3. Overstrain Load Product (min * Nm) = Tool Wear * Torque
        if self.tool_wear_col in df.columns and self.torque_col in df.columns:
            df["Overstrain_Product"] = df[self.tool_wear_col] * df[self.torque_col]
        else:
            df["Overstrain_Product"] = np.nan
            
        # 4. Missing Sensor Count (Number of NaN sensors per row)
        sensor_cols = [c for c in [self.air_temp_col, self.proc_temp_col, self.rot_speed_col,
                                   self.torque_col, self.tool_wear_col] if c in df.columns]
        df["Missing_Sensors_Count"] = df[sensor_cols].isnull().sum(axis=1)
        
        if is_df:
            return df
        return df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        """Returns feature names out for sklearn ColumnTransformer/Pipeline compatibility."""
        if input_features is None:
            input_features = self.feature_names_in_
        new_features = list(input_features) + [
            "Temp_Difference", "Mechanical_Power_W", "Overstrain_Product", "Missing_Sensors_Count"
        ]
        return np.array(new_features, dtype=object)
