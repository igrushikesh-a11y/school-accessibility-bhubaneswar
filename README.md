# Spatial Inequality in School Accessibility — Bhubaneswar

**Course:** Planning Informatics — GIS Lab End Semester Evaluation  
**Institution:** IIT Kharagpur | Department of Architecture and Regional Planning  
**Author:** Rushikesh Satish Thakare | 25AR60R47  
**Semester:** Spring 2025-2026  

---

## About This Project

This project analyses spatial inequality in government school 
accessibility across all 67 wards of Bhubaneswar Municipal 
Corporation (BMC) using geospatial analysis and machine learning.

### Techniques Used
- Spatial join (GeoPandas) — counting schools per ward
- Accessibility indicator computation (schools per 10k population)
- Pearson correlation analysis
- K-Means clustering (k=4, Elbow Method)
- Choropleth map generation (Matplotlib + GeoPandas)

---

## How to Run This Project

### 1. Clone or download this repository

Click the green "Code" button above → Download ZIP  
Extract the ZIP on your computer.

### 2. Install required libraries

Open terminal / command prompt in the project folder and run:

pip install -r requirements.txt

### 3. Run the analysis

python school_accessibility.py

### 4. Find your outputs

All maps, charts, and data files will be saved in the outputs/ folder.

---

## Input Data

| File | Description |
|------|-------------|
| data/Ward boundary.gpkg | BMC ward boundaries with population attributes |
| data/Govt Schools.gpkg | Government school point locations |

**Sources:** BhubaneswarOne, BMC GIS Cell, Census of India 2011

---

## Output Files Generated

| File | Description |
|------|-------------|
| map_school_count.png | Choropleth map of schools per ward |
| map_accessibility_per10k.png | Schools per 10,000 population |
| map_pop_per_school.png | Demand pressure map |
| map_kmeans_clusters.png | K-Means ward classification map |
| correlation_matrix.png | Pearson correlation heatmap |
| elbow_method.png | Optimal K selection chart |
| chart_cluster_accessibility.png | Bar chart by accessibility class |
| ward_accessibility_results.csv | Final ward-level data table |
| ward_accessibility_final.gpkg | GeoPackage for QGIS import |

---

## Project Report

The full project report (PDF) is included in this repository:  
`25AR60R47_Rushikesh_Report.pdf`

---

## References

- BhubaneswarOne GIS Portal
- Census of India 2011
- GeoPandas Documentation: https://geopandas.org
- Scikit-learn Documentation: https://scikit-learn.org