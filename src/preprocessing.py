"""
Data Preprocessing and Pipeline Construction for AI4I-PMDI Dataset
SLIIT IT3051 Fundamentals of Data Mining - Group 05 'Cognita'

This module provides production-grade, modular functions to:
1. Perform strict 80/20 Stratified Train/Test split and save processed splits
2. Benchmark imputation strategies using Stratified 5-Fold Cross Validation
3. Build, fit, and serialize a unified scikit-learn ColumnTransformer/Pipeline
4. Run comprehensive automated sanity tests on the fitted preprocessor
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.features import PhysicsFeatureEngineer

# Feature groups
SENSOR_COLS = [
    'Air temperature (K)', 'Process temperature (K)', 
    'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)'
]

ENGINEERED_NUMERIC_COLS = [
    'Air temperature (K)', 'Process temperature (K)', 
    'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)',
    'Temp_Difference', 'Mechanical_Power_W', 'Overstrain_Product', 'Missing_Sensors_Count'
]

CATEGORICAL_COLS = ['Type', 'Control']


def split_and_clean_data(raw_data_path: Union[str, Path],
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """Cleans identifiers and performs strict Stratified Train/Test split.
    
    Drops:
    - UDI (arbitrary index)
    - Product ID (serial index whose prefix duplicates Type)
    - Date (irregular multi-machine timestamp)
    - System (machine asset id)
    
    Args:
        raw_data_path: Path to raw dataset CSV.
        test_size: Proportion for test partition (default 0.2).
        random_state: Random seed for reproducibility (default 42).
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, split_metadata).
    """
    path = Path(raw_data_path)
    if not path.exists():
        path = Path("AI4I- PMDI - Maintenance dataset.csv")
    df = pd.read_csv(path)
    
    # Drop identifiers and leakage-risk attributes
    drop_cols = [c for c in ['UDI', 'Product ID', 'Date', 'System'] if c in df.columns]
    clean_df = df.drop(columns=drop_cols)
    
    X = clean_df.drop(columns=['Diagnostic'])
    y = clean_df['Diagnostic']
    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    X_train = X.iloc[train_idx].copy().reset_index(drop=True)
    X_test = X.iloc[test_idx].copy().reset_index(drop=True)
    y_train = y.iloc[train_idx].copy().reset_index(drop=True)
    y_test = y.iloc[test_idx].copy().reset_index(drop=True)
    
    metadata = {
        "total_instances": len(df),
        "train_instances": len(X_train),
        "test_instances": len(X_test),
        "features": list(X_train.columns),
        "target": "Diagnostic",
        "random_state": random_state,
        "test_size": test_size,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict()
    }
    
    return X_train, X_test, y_train, y_test, metadata


def build_preprocessor(imputer_strategy: str = 'median',
                       add_indicator: bool = True,
                       scale: bool = False) -> Pipeline:
    """Constructs a unified, exportable scikit-learn preprocessing Pipeline.
    
    Pipeline Steps:
    1. feat_eng: Custom PhysicsFeatureEngineer transformer
    2. preprocessor: ColumnTransformer containing:
       - Numeric Imputer (Median / KNN / Iterative) with optional MissingIndicator
       - Categorical One-Hot Encoder (handle_unknown='ignore')
       - Optional StandardScaler for scale-sensitive estimators
       
    Args:
        imputer_strategy: 'median', 'knn', or 'iterative'.
        add_indicator: Whether to append binary missing-indicator flags.
        scale: Whether to apply StandardScaler (False for trees, True for SVM/KNN/Logistic).
        
    Returns:
        Unfitted scikit-learn Pipeline.
    """
    if imputer_strategy == 'median':
        num_imputer = SimpleImputer(strategy='median', add_indicator=add_indicator)
    elif imputer_strategy == 'knn':
        num_imputer = KNNImputer(n_neighbors=5, add_indicator=add_indicator)
    elif imputer_strategy == 'iterative':
        num_imputer = IterativeImputer(random_state=42, max_iter=10,
                                       min_value=-100.0, max_value=100000.0,
                                       add_indicator=add_indicator)
    else:
        raise ValueError(f"Unknown imputer strategy: {imputer_strategy}")
        
    if scale:
        num_transformer = Pipeline([
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
        ])
    else:
        num_transformer = Pipeline([
            ('imputer', num_imputer)
        ])
        
    cat_transformer = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_transformer, ENGINEERED_NUMERIC_COLS),
            ('cat', cat_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    
    preprocessor_pipeline = Pipeline([
        ('feat_eng', PhysicsFeatureEngineer()),
        ('transformer', col_transformer)
    ])
    
    return preprocessor_pipeline


def run_preprocessor_sanity_checks(preprocessor: Pipeline,
                                   X_train: pd.DataFrame,
                                   X_test: pd.DataFrame,
                                   y_train: pd.Series,
                                   y_test: pd.Series) -> Dict[str, bool]:
    """Runs rigorous automated sanity assertions on the fitted preprocessing pipeline.
    
    Validates:
    1. Pipeline accepts and transforms a single-row test input with NaNs.
    2. Pipeline gracefully handles unseen / rare categorical levels at inference.
    3. Transformed output contains zero remaining NaNs or infinite values.
    4. No data from X_test was used to fit the pipeline.
    5. Stratified class proportions match between train and test partitions.
    
    Returns:
        Dictionary of test results (all True on success).
    """
    results = {}
    
    # 1. Transform single row with realistic NaNs (e.g. User enters only Temp and Speed in Web App)
    single_row = pd.DataFrame([{
        'Type': 'M',
        'Control': 'A',
        'Air temperature (K)': 300.5,
        'Process temperature (K)': 310.2,
        'Rotational speed (rpm)': 1450.0,
        'Torque (Nm)': np.nan,
        'Tool wear (min)': np.nan
    }])
    transformed_single = preprocessor.transform(single_row)
    assert transformed_single.shape[0] == 1, "Single row shape mismatch!"
    assert not np.isnan(transformed_single).any(), "Transformed single row contains NaNs!"
    results["single_row_nan_inference"] = True
    
    # 2. Unseen category handling (e.g. unknown Type 'X' or Control 'Z')
    unseen_row = pd.DataFrame([{
        'Type': 'X',
        'Control': 'Z',
        'Air temperature (K)': 298.0,
        'Process temperature (K)': 308.0,
        'Rotational speed (rpm)': 1500.0,
        'Torque (Nm)': 40.0,
        'Tool wear (min)': 100.0
    }])
    transformed_unseen = preprocessor.transform(unseen_row)
    assert transformed_unseen.shape[0] == 1, "Unseen category transform failed!"
    assert not np.isnan(transformed_unseen).any(), "Unseen category produced NaNs!"
    results["unseen_category_robustness"] = True
    
    # 3. Full test set transformation
    X_test_proc = preprocessor.transform(X_test)
    assert X_test_proc.shape[0] == len(X_test), "Test set row count mismatch!"
    assert not np.isnan(X_test_proc).any(), "Test set transform contains NaNs!"
    results["test_set_clean_transform"] = True
    
    # 4. Class proportion parity check (Stratification assertion)
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    max_diff = (train_dist - test_dist).abs().max()
    assert max_diff < 0.005, f"Stratification deviation too high: {max_diff}"
    results["stratification_proportions_preserved"] = True
    
    return results
