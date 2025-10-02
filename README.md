# Earthquake_Risk_Analytics

##  Overview
This repository presents a **comprehensive seismic hazard analysis** focusing on earthquakes in the **Marmara Region (2000–2025)**.  
The workflow integrates **Python-based statistical and spatio-temporal modeling** with **Power BI visualization**, enabling both scientific interpretation and decision-making support.

The project follows a structured 7-step pipeline:
1. **Data Preparation**
2. **Time-Series Analysis**
3. **Gutenberg–Richter Relationship**
4. **Magnitude–Depth Correlation**
5. **Fault Segment Analysis**
6. **Seismic Moment → Mw Conversion**
7. **Result Export (CSV, PNG, TXT)**

Final outputs are directly connected to **Power BI dashboards** for interactive exploration.

---

##  Project Structure

```markdown
Seismic-Hazard-SpatioTemporal-Analysis/
│
├── data/
│   └── marmara_faults_earthquakes_2000_2025.csv
│
├── src/
│   └── analysis.py               # Python main analysis script
│
├── analysis_outputs/             # Generated after running the script
│   ├── annual_summary.csv
│   ├── monthly_summary.csv
│   ├── gutenberg_richter.csv
│   ├── fault_summary.csv
│   ├── dataset_with_mw_from_m0.csv
│   ├── annual_counts.png
│   ├── annual_mean_magnitude.png
│   ├── gutenberg_richter.png
│   ├── magnitude_vs_depth.png
│   └── summary.txt
│
├── powerbi/
│   └── seismic_dashboard.pbix     # Power BI interactive report
│
└── README.md

---
```

##  Analysis Workflow (Python)

### 1. Data Preparation
- Date parsing and validation (`Date` → `datetime`).
- Feature engineering: `Year`, `Month`, `Magnitude bins`.
- Missing value summary per column.

### 2. Time-Series Analysis
- Annual and monthly earthquake frequency.
- Visualization of temporal patterns and mean magnitudes.

### 3. Gutenberg–Richter Law
- Relationship: **log10(N) = a – bM**.
- Estimation of **a** and **b** parameters using linear regression.
- Output: CSV + regression plot.

### 4. Magnitude–Depth Analysis
- Correlation between event magnitude and hypocentral depth.
- Scatter plot + correlation coefficient.

### 5. Fault Segment Analysis
- Aggregated statistics by `Nearest_Fault`:
  - Event count
  - Mean magnitude
  - Slip rate
  - Slip deficit
  - Elapsed time since last event
  - Mw potential
- Top-10 most active faults reported.

### 6. Moment → Mw Conversion
- Formula:  

- Comparison of dataset Mw vs. computed Mw.

### 7. Export & Documentation
- Results stored as `.csv`, `.png`, `.txt`.
- All outputs saved in `/analysis_outputs/`.
- `summary.txt` contains run timestamp, Gutenberg–Richter coefficients, correlation values, and missing data overview.

---

##  Power BI Dashboard

All generated `.csv` outputs can be directly imported into Power BI:

- **Annual & Monthly Trends** → Line/Bar charts  
- **Magnitude Distribution** → Histogram  
- **Fault Activity** → Treemap or Table  
- **Magnitude vs Depth** → Scatter plot  
- **Gutenberg–Richter Fit** → Line chart  

Interactive dashboards enhance decision-making for **earthquake risk management** and **hazard assessment**.

---

##  Installation & Usage

### Requirements
- Python ≥ 3.8  
- Libraries:  
```bash
pip install pandas numpy matplotlib scikit-learn
