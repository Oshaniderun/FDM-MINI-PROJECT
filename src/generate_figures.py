"""
Generate publication-quality EDA figures for Stage 3 EDA Report.
SLIIT IT3051 Fundamentals of Data Mining - Group 05 'Cognita'
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Destination directories
ART_DIR = Path(r"C:\Users\oshani\.gemini\antigravity\brain\aae4b8d4-a59b-4c10-a0b2-057dca792d27")
ART_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
csv_path = Path("AI4I- PMDI - Maintenance dataset.csv")
if not csv_path.exists():
    csv_path = Path("data/raw/AI4I-_PMDI_-_Maintenance_dataset.csv")
df = pd.read_csv(csv_path)

sensor_cols = [
    'Air temperature (K)', 'Process temperature (K)', 
    'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)'
]

palette = {
    'No failure': '#2b5c8f',
    'Heat Dissipation Failure': '#e74c3c',
    'Power Failure': '#f39c12',
    'Overstrain Failure': '#8e44ad',
    'Tool Wear Failure': '#16a085',
    'Random Failures': '#34495e'
}

# -------------------------------------------------------------
# Figure 1: Target Class Imbalance (Linear & Log Scale)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
counts = df['Diagnostic'].value_counts()
pcts = (counts / len(df) * 100)

bars1 = ax1.bar(counts.index, counts.values, color=[palette[c] for c in counts.index], edgecolor='black', alpha=0.85)
ax1.set_title("Diagnostic Class Distribution (Linear Scale)", fontweight='bold')
ax1.set_ylabel("Instance Count (N)")
ax1.set_xlabel("Diagnostic Condition")
ax1.tick_params(axis='x', rotation=30)
for bar, count, pct in zip(bars1, counts.values, pcts.values):
    ax1.annotate(f"{count:,}\n({pct:.2f}%)",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='semibold')
ax1.set_ylim(0, 11000)

bars2 = ax2.bar(counts.index, counts.values, color=[palette[c] for c in counts.index], edgecolor='black', alpha=0.85)
ax2.set_yscale('log')
ax2.set_title("Diagnostic Class Distribution (Log Scale - Showing Minority Classes)", fontweight='bold')
ax2.set_ylabel("Instance Count (Log10 Scale)")
ax2.set_xlabel("Diagnostic Condition")
ax2.tick_params(axis='x', rotation=30)
for bar, count in zip(bars2, counts.values):
    ax2.annotate(f"N={count}",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='semibold')
ax2.set_ylim(1, 20000)

plt.tight_layout()
plt.savefig(ART_DIR / "fig01_target_imbalance.png", dpi=300)
plt.close()
print("Saved fig01_target_imbalance.png")

# -------------------------------------------------------------
# Figure 2: Missingness Overview (Column % and Row Missing Counts)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
miss_pct = (df[sensor_cols].isnull().sum() / len(df) * 100)
bars_miss = ax1.barh(miss_pct.index, miss_pct.values, color='#e67e22', edgecolor='black', alpha=0.85)
ax1.set_title("Missing Value Rate per Sensor Column (%)", fontweight='bold')
ax1.set_xlabel("Missing Percentage (%)")
ax1.set_xlim(0, 100)
for bar in bars_miss:
    w = bar.get_width()
    ax1.annotate(f"{w:.2f}% ({int(w*100):,} rows)",
                 xy=(w, bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=9, fontweight='semibold')

row_miss = df[sensor_cols].isnull().sum(axis=1).value_counts().sort_index()
bars_row = ax2.bar([f"{k} Missing Sensors" for k in row_miss.index], row_miss.values, color='#3498db', edgecolor='black', alpha=0.85)
ax2.set_title("Distribution of Missing Sensors per Row (Row-level Sparsity)", fontweight='bold')
ax2.set_ylabel("Number of Rows")
ax2.set_ylim(0, 8000)
for bar, count in zip(bars_row, row_miss.values):
    pct = count / len(df) * 100
    ax2.annotate(f"{count:,}\n({pct:.2f}%)",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='semibold')

plt.tight_layout()
plt.savefig(ART_DIR / "fig02_missingness_overview.png", dpi=300)
plt.close()
print("Saved fig02_missingness_overview.png")

# -------------------------------------------------------------
# Figure 3: Co-Missingness Heatmap by Control Channel
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ctrl_avail = df.groupby('Control')[sensor_cols].apply(lambda g: g.notnull().mean() * 100)
sns.heatmap(ctrl_avail, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Sensor Availability (%)'},
            linewidths=1.5, linecolor='white', ax=ax, vmin=0, vmax=100)
ax.set_title("Sensor Availability Heatmap by Diagnostic Control Configuration Mode", fontweight='bold')
ax.set_xlabel("Monitored Sensor Channel")
ax.set_ylabel("Control Mode (Multiplexing Setting)")
plt.tight_layout()
plt.savefig(ART_DIR / "fig03_comissingness_by_control.png", dpi=300)
plt.close()
print("Saved fig03_comissingness_by_control.png")

# -------------------------------------------------------------
# Figure 4: Numeric Distributions (Histograms + KDEs)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

units = {
    'Air temperature (K)': 'Kelvin (K)',
    'Process temperature (K)': 'Kelvin (K)',
    'Rotational speed (rpm)': 'Revolutions per Minute (rpm)',
    'Torque (Nm)': 'Newton-meters (Nm)',
    'Tool wear (min)': 'Minutes (min)'
}

for i, col in enumerate(sensor_cols):
    ax = axes[i]
    s = df[col].dropna()
    sns.histplot(s, kde=True, ax=ax, color='#2980b9', edgecolor='black', alpha=0.6, bins=35)
    ax.axvline(s.median(), color='red', linestyle='--', linewidth=1.5, label=f"Median: {s.median():.1f}")
    ax.axvline(s.mean(), color='green', linestyle=':', linewidth=1.5, label=f"Mean: {s.mean():.1f}")
    ax.set_title(f"{col}\n(Skew: {s.skew():.2f}, N={len(s):,})", fontweight='bold')
    ax.set_xlabel(units[col])
    ax.set_ylabel("Density / Count")
    ax.legend(loc='upper right', fontsize=8)

# Empty 6th plot used for summary text
axes[5].axis('off')
summary_text = (
    "Summary of Distribution Profiles:\n\n"
    "• Air & Process Temp: Normal, symmetric\n"
    "  distributions (Skew ≈ 0.05 - 0.10).\n\n"
    "• Rotational Speed: Strongly right-skewed\n"
    "  (Skew = +2.17) due to high-speed no-load excursions.\n\n"
    "• Torque: Symmetrical (Skew ≈ 0.05) with\n"
    "  extreme values concentrated in failure events.\n\n"
    "• Tool Wear: Uniformly distributed across 0-250 min\n"
    "  reflecting linear operational tool aging."
)
axes[5].text(0.1, 0.5, summary_text, fontsize=11, va='center', bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9f9', edgecolor='#bdc3c7'))

plt.tight_layout()
plt.savefig(ART_DIR / "fig04_sensor_distributions_kde.png", dpi=300)
plt.close()
print("Saved fig04_sensor_distributions_kde.png")

# -------------------------------------------------------------
# Figure 5: Sensor Boxplots Stratified by Diagnostic Condition
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    ax = axes[i]
    sub = df[[col, 'Diagnostic']].dropna()
    sns.boxplot(data=sub, x='Diagnostic', y=col, ax=ax, palette=palette, flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax.set_title(f"{col} by Diagnostic Condition", fontweight='bold')
    ax.set_ylabel(units[col])
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=35)

axes[5].axis('off')
box_summary = (
    "Outlier & Failure Signature Insights:\n\n"
    "• Power Failure: Characterized by extreme low/high\n"
    "  rotational speeds combined with extreme torque.\n\n"
    "• Heat Dissipation Failure: Characterized by low\n"
    "  rotational speed and elevated process temperature.\n\n"
    "• Overstrain Failure: Manifests exclusively at high\n"
    "  torque and high tool wear.\n\n"
    "• Tool Wear Failure: Clustered tightly at high\n"
    "  tool wear (200 - 240 min).\n\n"
    "CONCLUSION: Outliers are diagnostic signals, not noise!"
)
axes[5].text(0.05, 0.5, box_summary, fontsize=11, va='center', bbox=dict(boxstyle='round,pad=0.8', facecolor='#fef9e7', edgecolor='#f39c12'))

plt.tight_layout()
plt.savefig(ART_DIR / "fig05_sensor_boxplots_by_diagnostic.png", dpi=300)
plt.close()
print("Saved fig05_sensor_boxplots_by_diagnostic.png")

# -------------------------------------------------------------
# Figure 6: Pearson & Spearman Correlation Heatmaps
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

pearson_corr = df[sensor_cols].corr(method='pearson')
spearman_corr = df[sensor_cols].corr(method='spearman')

sns.heatmap(pearson_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=1, ax=ax1, cbar_kws={'label': 'Pearson Correlation (r)'})
ax1.set_title("Pairwise Pearson Correlation Heatmap", fontweight='bold')

sns.heatmap(spearman_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=1, ax=ax2, cbar_kws={'label': 'Spearman Rank Correlation (rho)'})
ax2.set_title("Pairwise Spearman Correlation Heatmap", fontweight='bold')

plt.tight_layout()
plt.savefig(ART_DIR / "fig06_correlation_heatmaps.png", dpi=300)
plt.close()
print("Saved fig06_correlation_heatmaps.png")

# -------------------------------------------------------------
# Figure 7: Within-Control Physical Correlations
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Control B: Rotational Speed vs Torque
ctrl_b = df[df['Control'] == 'B']
sns.scatterplot(data=ctrl_b, x='Rotational speed (rpm)', y='Torque (Nm)', hue='Diagnostic',
                palette=palette, alpha=0.7, ax=ax1, edgecolor='none', s=25)
ax1.set_title(f"Spindle Speed vs Torque (Control B Mode, N={len(ctrl_b):,})\nInverse Physical Load Relationship (Pearson r = -0.859)", fontweight='bold')
ax1.set_xlabel("Rotational Speed (rpm)")
ax1.set_ylabel("Torque (Nm)")

# Control A: Air Temp vs Process Temp
ctrl_a = df[df['Control'] == 'A']
sns.scatterplot(data=ctrl_a, x='Air temperature (K)', y='Process temperature (K)', hue='Diagnostic',
                palette=palette, alpha=0.7, ax=ax2, edgecolor='none', s=25)
ax2.set_title(f"Air Temp vs Process Temp (Control A Mode, N={len(ctrl_a):,})\nThermal Dissipation Relationship (Pearson r = +0.871)", fontweight='bold')
ax2.set_xlabel("Air Temperature (K)")
ax2.set_ylabel("Process Temperature (K)")

plt.tight_layout()
plt.savefig(ART_DIR / "fig07_within_control_correlations.png", dpi=300)
plt.close()
print("Saved fig07_within_control_correlations.png")

# -------------------------------------------------------------
# Figure 8: Domain Failure Physics & Rule Boundaries
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. HDF: Delta T vs RPM
df_a = df[df['Control'] == 'A'].copy()
df_a['Delta_T'] = df_a['Process temperature (K)'] - df_a['Air temperature (K)']
sns.scatterplot(data=df_a, x='Rotational speed (rpm)', y='Delta_T', hue='Diagnostic',
                palette=palette, ax=axes[0, 0], alpha=0.75, s=30)
axes[0, 0].axvline(1380, color='red', linestyle='--', linewidth=1.5, label='RPM Boundary (<1380)')
axes[0, 0].axhline(8.6, color='darkred', linestyle='--', linewidth=1.5, label='Delta T Boundary (<8.6 K)')
axes[0, 0].set_title("Heat Dissipation Failure Boundary (Control A)\n(Process - Air Temp < 8.6 K and RPM < 1380)", fontweight='bold')
axes[0, 0].set_xlabel("Rotational Speed (rpm)")
axes[0, 0].set_ylabel("Temperature Difference ΔT (K)")
axes[0, 0].legend(loc='upper right', fontsize=8)

# 2. PWF: Power Curve
df_b = df[df['Control'] == 'B'].copy()
df_b['Power_W'] = df_b['Torque (Nm)'] * df_b['Rotational speed (rpm)'] * (2 * np.pi / 60)
sns.scatterplot(data=df_b, x='Rotational speed (rpm)', y='Torque (Nm)', hue='Diagnostic',
                palette=palette, ax=axes[0, 1], alpha=0.75, s=30)
# Hyperbolic boundary curves
rpm_grid = np.linspace(1100, 2900, 200)
torque_low = 3500 / (rpm_grid * 2 * np.pi / 60)
torque_high = 9000 / (rpm_grid * 2 * np.pi / 60)
axes[0, 1].plot(rpm_grid, torque_low, 'r--', label='Lower Power Boundary (3500 W)')
axes[0, 1].plot(rpm_grid, torque_high, 'm--', label='Upper Power Boundary (9000 W)')
axes[0, 1].set_title("Power Failure Boundary Curves (Control B)\n(Power < 3500 W or Power > 9000 W)", fontweight='bold')
axes[0, 1].set_xlabel("Rotational Speed (rpm)")
axes[0, 1].set_ylabel("Torque (Nm)")
axes[0, 1].set_ylim(0, 80)
axes[0, 1].legend(loc='upper right', fontsize=8)

# 3. OSF: Overstrain Boundary
df_c = df[df['Control'] == 'C'].copy()
df_c['Overstrain'] = df_c['Tool wear (min)'] * df_c['Torque (Nm)']
sns.scatterplot(data=df_c, x='Torque (Nm)', y='Tool wear (min)', hue='Diagnostic',
                palette=palette, ax=axes[1, 0], alpha=0.75, s=30)
torque_grid = np.linspace(20, 80, 200)
axes[1, 0].plot(torque_grid, 11000 / torque_grid, 'r--', label='Type L Limit (11,000 min*Nm)')
axes[1, 0].plot(torque_grid, 12000 / torque_grid, 'g--', label='Type M Limit (12,000 min*Nm)')
axes[1, 0].plot(torque_grid, 13000 / torque_grid, 'b--', label='Type H Limit (13,000 min*Nm)')
axes[1, 0].set_title("Overstrain Failure Boundaries (Control C)\n(Tool Wear x Torque > Limit by Type)", fontweight='bold')
axes[1, 0].set_xlabel("Torque (Nm)")
axes[1, 0].set_ylabel("Tool Wear (min)")
axes[1, 0].set_ylim(0, 260)
axes[1, 0].legend(loc='lower left', fontsize=8)

# 4. TWF: Tool Wear Distribution
sns.histplot(data=df_c, x='Tool wear (min)', hue='Diagnostic', palette=palette,
             ax=axes[1, 1], multiple='stack', bins=30)
axes[1, 1].axvspan(198, 246, color='teal', alpha=0.2, label='TWF Failure Band (198-246 min)')
axes[1, 1].set_title("Tool Wear Failure Concentration (Control C)\n(Failures Occur Between 198 - 246 min)", fontweight='bold')
axes[1, 1].set_xlabel("Tool Wear (min)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(ART_DIR / "fig08_domain_physics_failure_boundaries.png", dpi=300)
plt.close()
print("Saved fig08_domain_physics_failure_boundaries.png")

# -------------------------------------------------------------
# Figure 9: Categorical Failure Distribution by Type & Control
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ct_type = pd.crosstab(df['Type'], df['Diagnostic'], normalize='index') * 100
ct_type.plot(kind='bar', stacked=True, ax=ax1, color=[palette[c] for c in ct_type.columns], edgecolor='black', alpha=0.85)
ax1.set_title("Failure Proportion by Machine Quality Type (%)", fontweight='bold')
ax1.set_ylabel("Percentage (%)")
ax1.set_xlabel("Product Quality Tier (Type)")
ax1.tick_params(axis='x', rotation=0)
ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)

ct_ctrl = pd.crosstab(df['Control'], df['Diagnostic'])
ct_ctrl.plot(kind='bar', stacked=True, ax=ax2, color=[palette[c] for c in ct_ctrl.columns], edgecolor='black', alpha=0.85)
ax2.set_title("Failure Counts by Diagnostic Control Mode", fontweight='bold')
ax2.set_ylabel("Record Count")
ax2.set_xlabel("Control Configuration Mode")
ax2.tick_params(axis='x', rotation=0)
ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)

plt.tight_layout()
plt.savefig(ART_DIR / "fig09_categorical_failure_breakdown.png", dpi=300)
plt.close()
print("Saved fig09_categorical_failure_breakdown.png")

# -------------------------------------------------------------
# Figure 10: Temporal Dynamics and Sampling Step Statistics
# -------------------------------------------------------------
df['Date_parsed'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
df_chrono = df.sort_values('Date_parsed').copy()
df_chrono['Year'] = df_chrono['Date_parsed'].dt.year

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

yearly_fail = df_chrono.groupby(['Year', 'Diagnostic']).size().unstack(fill_value=0)
yearly_fail.plot(kind='bar', stacked=True, ax=ax1, color=[palette[c] for c in yearly_fail.columns], edgecolor='black', alpha=0.85)
ax1.set_title("Annual Telemetry & Failure Counts (2014 - 2023)", fontweight='bold')
ax1.set_ylabel("Record Count")
ax1.set_xlabel("Year")
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper right', fontsize=8)

time_diff_hours = df_chrono['Date_parsed'].diff().dt.total_seconds() / 3600.0
sns.histplot(time_diff_hours.dropna(), bins=40, ax=ax2, color='#8e44ad', edgecolor='black', log_scale=(False, True))
ax2.set_title("Chronological Sampling Step Interval Distribution", fontweight='bold')
ax2.set_xlabel("Sampling Step Interval (Hours between consecutive logs)")
ax2.set_ylabel("Count (Log Scale)")

plt.tight_layout()
plt.savefig(ART_DIR / "fig10_temporal_sampling_dynamics.png", dpi=300)
plt.close()
print("Saved fig10_temporal_sampling_dynamics.png")

print("ALL 10 EDA FIGURES GENERATED AND SAVED SUCCESSFULLY!")
