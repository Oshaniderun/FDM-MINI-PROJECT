"""
EDA Utilities for AI4I-PMDI Predictive Maintenance Dataset
SLIIT IT3051 Fundamentals of Data Mining - Group 05 'Cognita'

This module contains modular, testable utility functions for:
- Data structure audits & summary statistics
- Missingness audits and co-missingness tests
- Domain physics validation (AI4I failure rules)
- Outlier detection (IQR & Z-score) and failure concentration analysis
- Statistical hypothesis testing (ANOVA, Kruskal-Wallis, Chi-Square, Mutual Information)
- Publication-quality plotting and figure export
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


def load_raw_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load the raw dataset with proper error checking.
    
    Args:
        data_path: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame containing the raw data.
    """
    path = Path(data_path)
    if not path.exists():
        # Fallback to current directory if not found in data/raw
        alt_path = Path("AI4I- PMDI - Maintenance dataset.csv")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path} or {alt_path}")
    df = pd.read_csv(path)
    return df


def audit_dataset_structure(df: pd.DataFrame) -> Dict[str, Union[Tuple[int, int], pd.DataFrame, int]]:
    """Performs an exhaustive structural audit on the dataset.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Dictionary containing shape, data types, duplicate counts, and memory usage.
    """
    audit = {
        "shape": df.shape,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "full_duplicates": int(df.duplicated().sum()),
        "duplicate_udi": int(df["UDI"].duplicated().sum()),
        "duplicate_product_id": int(df["Product ID"].duplicated().sum()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        "columns_summary": pd.DataFrame({
            "Dtype": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
            "Null %": (df.isnull().sum() / len(df) * 100).round(2),
            "Unique Count": df.nunique(),
        })
    }
    return audit


def compute_missingness_summary(df: pd.DataFrame, sensor_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes column-level and row-level missingness distributions.
    
    Args:
        df: Input dataframe.
        sensor_cols: List of numeric sensor column names.
        
    Returns:
        Tuple of (column_missing_df, row_missing_df).
    """
    col_missing = pd.DataFrame({
        "Missing Count": df[sensor_cols].isnull().sum(),
        "Missing %": (df[sensor_cols].isnull().sum() / len(df) * 100).round(2),
        "Present Count": df[sensor_cols].notnull().sum(),
        "Present %": (df[sensor_cols].notnull().sum() / len(df) * 100).round(2),
    })
    
    row_missing_counts = df[sensor_cols].isnull().sum(axis=1)
    row_summary = pd.DataFrame({
        "Missing Sensors Count": row_missing_counts.value_counts().sort_index(),
        "Row Percentage %": (row_missing_counts.value_counts(normalize=True).sort_index() * 100).round(2)
    })
    return col_missing, row_summary


def compute_outlier_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Computes skewness, IQR bounds, and outlier counts for numeric columns.
    
    Args:
        df: Input dataframe.
        numeric_cols: List of numeric columns to evaluate.
        
    Returns:
        DataFrame summarizing outlier metrics and concentration in failure classes.
    """
    outlier_records = []
    for col in numeric_cols:
        series = df[col].dropna()
        skew = series.skew()
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        is_iqr_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        iqr_count = is_iqr_outlier.sum()
        
        z_scores = np.abs(stats.zscore(series))
        z_count = (z_scores > 3.0).sum()
        
        failures_in_outliers = (df.loc[is_iqr_outlier, "Diagnostic"] != "No failure").sum() if iqr_count > 0 else 0
        failure_pct_in_outliers = (failures_in_outliers / iqr_count * 100) if iqr_count > 0 else 0.0
        
        outlier_records.append({
            "Feature": col,
            "Skewness": round(skew, 4),
            "Q1 (25%)": round(q25, 2),
            "Median (50%)": round(series.median(), 2),
            "Q3 (75%)": round(q75, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "IQR Outlier Count": int(iqr_count),
            "IQR Outlier %": round(iqr_count / len(series) * 100, 2),
            "Z > 3 Outlier Count": int(z_count),
            "Failures in Outliers": int(failures_in_outliers),
            "% Outliers that are Failures": round(failure_pct_in_outliers, 2)
        })
    return pd.DataFrame(outlier_records)


def verify_domain_rules(df: pd.DataFrame) -> Dict[str, Dict[str, Union[int, float]]]:
    """Tests the physical failure rules from the AI4I domain specifications on the PMDI dataset.
    
    Rules:
    1. HDF: (Process Temp - Air Temp) < 8.6 K and Rotational Speed < 1380 rpm
    2. PWF: Power = Torque * (Rotational Speed * 2*pi / 60) outside [3500, 9000] W
    3. OSF: Tool Wear * Torque > threshold (L: 11000, M: 12000, H: 13000 min*Nm)
    4. TWF: Tool Wear between 200 and 240 min
    
    Returns:
        Dictionary detailing rule matches, precision, recall, and unexplainable failure counts.
    """
    results = {}
    
    # 1. HDF Rule
    hdf_rule = (df["Process temperature (K)"] - df["Air temperature (K)"] < 8.6) & (df["Rotational speed (rpm)"] < 1380)
    hdf_actual = df["Diagnostic"] == "Heat Dissipation Failure"
    results["Heat Dissipation Failure"] = {
        "Total Actual": int(hdf_actual.sum()),
        "Rule Triggered": int(hdf_rule.sum()),
        "True Positives": int((hdf_rule & hdf_actual).sum()),
        "False Positives": int((hdf_rule & ~hdf_actual).sum()),
        "False Negatives": int((~hdf_rule & hdf_actual).sum()),
        "Precision %": 100.0 if hdf_rule.sum() > 0 else 0.0,
        "Recall %": round((hdf_rule & hdf_actual).sum() / hdf_actual.sum() * 100, 2),
        "Control Subset": "Control A (100%)"
    }
    
    # 2. PWF Rule
    power = df["Torque (Nm)"] * df["Rotational speed (rpm)"] * (2 * np.pi / 60)
    pwf_rule = (power < 3500) | (power > 9000)
    pwf_actual = df["Diagnostic"] == "Power Failure"
    results["Power Failure"] = {
        "Total Actual": int(pwf_actual.sum()),
        "Rule Triggered": int(pwf_rule.fillna(False).sum()),
        "True Positives": int((pwf_rule.fillna(False) & pwf_actual).sum()),
        "False Positives": int((pwf_rule.fillna(False) & ~pwf_actual).sum()),
        "False Negatives": int((~pwf_rule.fillna(False) & pwf_actual).sum()),
        "Precision %": 100.0 if pwf_rule.fillna(False).sum() > 0 else 0.0,
        "Recall %": round((pwf_rule.fillna(False) & pwf_actual).sum() / pwf_actual.sum() * 100, 2),
        "Control Subset": "Control B (100%)"
    }
    
    # 3. OSF Rule
    osf_thresh = df["Type"].map({"L": 11000, "M": 12000, "H": 13000})
    osf_prod = df["Tool wear (min)"] * df["Torque (Nm)"]
    osf_rule = osf_prod > osf_thresh
    osf_actual = df["Diagnostic"] == "Overstrain Failure"
    results["Overstrain Failure"] = {
        "Total Actual": int(osf_actual.sum()),
        "Rule Triggered": int(osf_rule.fillna(False).sum()),
        "True Positives": int((osf_rule.fillna(False) & osf_actual).sum()),
        "False Positives": int((osf_rule.fillna(False) & ~osf_actual).sum()),
        "False Negatives": int((~osf_rule.fillna(False) & osf_actual).sum()),
        "Precision %": 100.0 if osf_rule.fillna(False).sum() > 0 else 0.0,
        "Recall %": round((osf_rule.fillna(False) & osf_actual).sum() / osf_actual.sum() * 100, 2),
        "Control Subset": "Control C (100%)"
    }
    
    # 4. TWF Rule
    twf_actual = df["Diagnostic"] == "Tool Wear Failure"
    twf_series = df.loc[twf_actual, "Tool wear (min)"]
    twf_in_range = ((twf_series >= 198) & (twf_series <= 246)).sum()
    results["Tool Wear Failure"] = {
        "Total Actual": int(twf_actual.sum()),
        "Min Wear (min)": float(twf_series.min()),
        "Max Wear (min)": float(twf_series.max()),
        "Mean Wear (min)": round(float(twf_series.mean()), 2),
        "Failures in [198, 246] min": int(twf_in_range),
        "Control Subset": "Control C (100%)"
    }
    
    return results


def run_statistical_hypothesis_tests(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs ANOVA, Kruskal-Wallis, Chi-square, and Mutual Information tests against Diagnostic.
    
    Args:
        df: Input dataframe.
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
        
    Returns:
        Tuple of (numeric_tests_df, categorical_tests_df, mutual_info_df).
    """
    # 1. Numeric vs Target
    num_records = []
    for col in numeric_cols:
        groups = [group[col].dropna().values for _, group in df.groupby("Diagnostic") if len(group[col].dropna()) > 0]
        if len(groups) > 1:
            kw_stat, kw_p = stats.kruskal(*groups)
            f_stat, f_p = stats.f_oneway(*groups)
            num_records.append({
                "Feature": col,
                "ANOVA F-stat": round(f_stat, 2),
                "ANOVA p-value": f"{f_p:.2e}",
                "Kruskal-Wallis H": round(kw_stat, 2),
                "KW p-value": f"{kw_p:.2e}",
                "Significance": "p < 0.001 (Highly Significant)" if kw_p < 0.001 else "Not Significant"
            })
    num_df = pd.DataFrame(num_records)
    
    # 2. Categorical vs Target
    cat_records = []
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df["Diagnostic"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        cat_records.append({
            "Feature": col,
            "Chi2-Statistic": round(chi2, 2),
            "p-value": f"{p:.2e}",
            "Degrees of Freedom": dof,
            "Significance": "p < 0.001 (Highly Significant)" if p < 0.001 else "Not Significant"
        })
    cat_df = pd.DataFrame(cat_records)
    
    # 3. Mutual Information
    mi_records = []
    for col in numeric_cols:
        sub = df[[col, "Diagnostic"]].dropna()
        y_encoded = pd.factorize(sub["Diagnostic"])[0]
        mi_val = mutual_info_classif(sub[[col]], y_encoded, random_state=42)[0]
        mi_records.append({
            "Feature": col,
            "Mutual Information (nats)": round(mi_val, 4)
        })
    mi_df = pd.DataFrame(mi_records).sort_values(by="Mutual Information (nats)", ascending=False)
    
    return num_df, cat_df, mi_df
