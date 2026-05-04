# =============================================================
# Planning Informatics Assignment
# Title: Spatial Inequality in School Accessibility Across Wards
# Tools: GeoPandas, Scikit-learn, Matplotlib, Seaborn
# =============================================================

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# STEP 0: FILE PATHS — UPDATE THESE TO YOUR ACTUAL PATHS
# -------------------------------------------------------
         # CSV exported from QGIS
WARD_SHP = 'data/Ward boundary.gpkg'
SCHOOLS_SHP = 'data/Govt Schools.gpkg'
OUTPUT_FOLDER = 'outputs/'

import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("="*60)
print("PLANNING INFORMATICS - SCHOOL ACCESSIBILITY ANALYSIS")
print("="*60)

# -------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------
print("\n[1] Loading spatial data...")

wards = gpd.read_file(WARD_SHP)
schools = gpd.read_file(SCHOOLS_SHP)

# Ensure same CRS
wards = wards.to_crs(epsg=32645)
schools = schools.to_crs(epsg=32645)

print(f"    Wards loaded: {len(wards)} wards")
print(f"    Schools loaded: {len(schools)} schools")
print(f"    Ward columns: {list(wards.columns)}")

# -------------------------------------------------------
# STEP 2: SPATIAL JOIN — COUNT SCHOOLS PER WARD
# -------------------------------------------------------
print("\n[2] Spatial join: counting schools per ward...")

schools_in_wards = gpd.sjoin(schools, wards[['wardno', 'geometry']], 
                              how='left', predicate='within')
school_count = schools_in_wards.groupby('wardno').size().reset_index(name='school_count')

wards = wards.merge(school_count, on='wardno', how='left')
wards['school_count'] = wards['school_count'].fillna(0).astype(int)

print(f"    Total schools mapped to wards: {wards['school_count'].sum()}")
print(f"    Wards with 0 schools: {(wards['school_count'] == 0).sum()}")

# -------------------------------------------------------
# STEP 3: COMPUTE ACCESSIBILITY INDICATORS
# -------------------------------------------------------
print("\n[3] Computing accessibility indicators...")

# Rename using ACTUAL column names from your file
wards = wards.rename(columns={
    'totalwardp': 'population',
    'POP_Dens_1': 'pop_density',
    'Area_km2_': 'area_km2'
})

# Calculate area from geometry directly (backup)
wards['area_km2'] = wards.geometry.area / 1_000_000

# Accessibility Indicator 1: Schools per 10,000 people
wards['schools_per_10k'] = (wards['school_count'] / wards['population']) * 10000
wards['schools_per_10k'] = wards['schools_per_10k'].replace([np.inf, -np.inf], 0).fillna(0)

# Accessibility Indicator 2: Schools per sq km
wards['schools_per_sqkm'] = wards['school_count'] / wards['area_km2']
wards['schools_per_sqkm'] = wards['schools_per_sqkm'].replace([np.inf, -np.inf], 0).fillna(0)

# Accessibility Indicator 3: Population per school (demand pressure)
wards['pop_per_school'] = wards['population'] / wards['school_count'].replace(0, np.nan)
wards['pop_per_school'] = wards['pop_per_school'].fillna(wards['population'])

print(f"    Avg schools per 10k population: {wards['schools_per_10k'].mean():.2f}")
print(f"    Avg population per school: {wards['pop_per_school'].mean():.0f}")
# -------------------------------------------------------
# STEP 4: CORRELATION ANALYSIS
# -------------------------------------------------------
print("\n[4] Running correlation analysis...")

corr_vars = ['school_count', 'schools_per_10k', 'schools_per_sqkm', 
             'pop_per_school', 'population', 'pop_density', 'area_km2']

corr_matrix = wards[corr_vars].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, ax=ax, linewidths=0.5,
            annot_kws={'size': 9})
ax.set_title('Correlation Matrix: School Accessibility Indicators\nvs Ward Characteristics', 
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: correlation_matrix.png")

# Print key correlations
r1, p1 = pearsonr(wards['population'], wards['school_count'])
r2, p2 = pearsonr(wards['pop_density'], wards['schools_per_10k'])
print(f"    Population vs School Count: r = {r1:.3f}, p = {p1:.4f}")
print(f"    Pop Density vs Schools per 10k: r = {r2:.3f}, p = {p2:.4f}")

# -------------------------------------------------------
# STEP 5: K-MEANS CLUSTERING
# -------------------------------------------------------
print("\n[5] Running K-Means clustering...")

cluster_features = ['schools_per_10k', 'schools_per_sqkm', 
                    'pop_per_school', 'pop_density']

X = wards[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K using Elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)', fontsize=11)
ax.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=11)
ax.set_title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: elbow_method.png")

# Apply K=4 clusters (standard for accessibility classification)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
wards['cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters based on accessibility score
cluster_means = wards.groupby('cluster')['schools_per_10k'].mean().sort_values()
cluster_labels = {}
labels = ['Very Low Accessibility', 'Low Accessibility', 
          'Moderate Accessibility', 'High Accessibility']
for i, (cluster_id, _) in enumerate(cluster_means.items()):
    cluster_labels[cluster_id] = labels[i]

wards['accessibility_class'] = wards['cluster'].map(cluster_labels)

print("    Cluster distribution:")
print(wards['accessibility_class'].value_counts().to_string())

# -------------------------------------------------------
# STEP 6: VISUALIZATIONS
# -------------------------------------------------------
print("\n[6] Generating maps and charts...")

# --- MAP 1: School Count per Ward ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
wards.plot(column='school_count', ax=ax, legend=True, cmap='YlOrRd',
           scheme='quantiles', k=5,
           legend_kwds={'title': 'School Count', 'loc': 'lower right'})
schools.plot(ax=ax, color='blue', markersize=3, alpha=0.5, label='School')
ax.set_title('Number of Government Schools per Ward\n(BMC Area)', 
             fontsize=14, fontweight='bold')
ax.set_axis_off()
ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'map_school_count.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: map_school_count.png")

# --- MAP 2: Schools per 10,000 population ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
wards.plot(column='schools_per_10k', ax=ax, legend=True, cmap='RdYlGn',
           scheme='quantiles', k=5,
           legend_kwds={'title': 'Schools per\n10,000 Population', 'loc': 'lower right'})
ax.set_title('School Accessibility: Schools per 10,000 Population per Ward',
             fontsize=13, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'map_accessibility_per10k.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: map_accessibility_per10k.png")

# --- MAP 3: K-Means Cluster Map ---
color_map = {
    'Very Low Accessibility': '#d73027',
    'Low Accessibility': '#fc8d59',
    'Moderate Accessibility': '#91cf60',
    'High Accessibility': '#1a9641'
}

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
for cls, color in color_map.items():
    subset = wards[wards['accessibility_class'] == cls]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, label=cls, edgecolor='white', linewidth=0.3)

wards.boundary.plot(ax=ax, color='gray', linewidth=0.4, alpha=0.5)
ax.set_title('K-Means Clustering: Ward Classification\nby School Accessibility Patterns',
             fontsize=14, fontweight='bold')
ax.set_axis_off()
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10, 
          title='Accessibility Class', title_fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'map_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: map_kmeans_clusters.png")

# --- MAP 4: Population per School (Demand Pressure) ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
wards.plot(column='pop_per_school', ax=ax, legend=True, cmap='Reds',
           scheme='quantiles', k=5,
           legend_kwds={'title': 'Population\nper School', 'loc': 'lower right'})
ax.set_title('Demand Pressure: Population per School per Ward',
             fontsize=13, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'map_pop_per_school.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: map_pop_per_school.png")

# --- CHART: Bar chart of accessibility by cluster ---
fig, ax = plt.subplots(figsize=(10, 6))
cluster_summary = wards.groupby('accessibility_class').agg(
    ward_count=('wardno', 'count'),
    avg_schools=('school_count', 'mean'),
    avg_per_10k=('schools_per_10k', 'mean'),
    avg_pop_per_school=('pop_per_school', 'mean')
).reset_index()

order = ['Very Low Accessibility', 'Low Accessibility', 
         'Moderate Accessibility', 'High Accessibility']
cluster_summary['accessibility_class'] = pd.Categorical(
    cluster_summary['accessibility_class'], categories=order, ordered=True)
cluster_summary = cluster_summary.sort_values('accessibility_class')

colors = ['#d73027', '#fc8d59', '#91cf60', '#1a9641']
bars = ax.bar(cluster_summary['accessibility_class'], 
              cluster_summary['avg_per_10k'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Accessibility Class', fontsize=11)
ax.set_ylabel('Average Schools per 10,000 Population', fontsize=11)
ax.set_title('Average School Accessibility by Cluster Category', 
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

for bar, val in zip(bars, cluster_summary['avg_per_10k']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'chart_cluster_accessibility.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: chart_cluster_accessibility.png")

# -------------------------------------------------------
# STEP 7: EXPORT FINAL DATA
# -------------------------------------------------------
print("\n[7] Exporting final processed data...")

output_cols = ['wardno', 'population', 'pop_density', 'area_km2',
               'school_count', 'schools_per_10k', 'schools_per_sqkm',
               'pop_per_school', 'cluster', 'accessibility_class']

wards[output_cols].to_csv(OUTPUT_FOLDER + 'ward_accessibility_results.csv', index=False)
print("    Saved: ward_accessibility_results.csv")

# Save final GeoPackage for QGIS import
wards.to_file(OUTPUT_FOLDER + 'ward_accessibility_final.gpkg', driver='GPKG')
print("    Saved: ward_accessibility_final.gpkg  ← Import this back into QGIS!")

# -------------------------------------------------------
# STEP 8: SUMMARY STATISTICS REPORT
# -------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Wards Analyzed: {len(wards)}")
print(f"Total Government Schools: {int(wards['school_count'].sum())}")
print(f"Average Schools per Ward: {wards['school_count'].mean():.1f}")
print(f"Min Schools in a Ward: {wards['school_count'].min()}")
print(f"Max Schools in a Ward: {wards['school_count'].max()}")
print(f"\nAccessibility Range:")
print(f"  Min (schools/10k pop): {wards['schools_per_10k'].min():.2f}")
print(f"  Max (schools/10k pop): {wards['schools_per_10k'].max():.2f}")
print(f"  Mean (schools/10k pop): {wards['schools_per_10k'].mean():.2f}")
print(f"\nCluster Summary:")
print(wards.groupby('accessibility_class')[['school_count','schools_per_10k',
                                             'pop_per_school']].mean().round(2).to_string())
print("\n" + "="*60)
print("ANALYSIS COMPLETE! All outputs saved to:", OUTPUT_FOLDER)
print("="*60)