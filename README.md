# Stress-Testing Meta-Analyses: A Reproducible Python Framework for Robustness Assessment of Continuous Outcomes

## Overview

This repository contains the complete executable Python code accompanying the technical report:

**Stress-Testing Meta-Analyses: A Reproducible Python Framework for Robustness Assessment of Continuous Outcomes**

The framework provides a reproducible workflow for stress-testing pairwise meta-analyses of continuous outcomes.

## Analyses included

The Python script performs:

- Input-data validation
- Hedges' g effect-size calculation
- Sampling-variance calculation
- Fixed-effect meta-analysis
- DerSimonian-Laird random-effects meta-analysis
- Paule-Mandel estimation of between-study variance
- Conventional confidence intervals
- HKSJ-style confidence intervals
- Prediction intervals
- Leave-one-out sensitivity analysis
- Identification of the study producing the largest absolute leave-one-out change
- Egger-type regression when applicable
- Automated robustness assessment
- Export of analysis results and a forest plot

## Software environment

The analysis was executed using:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- Matplotlib 3.10.8

## Data

The demonstration uses a synthetic dataset of 12 independent studies. The dataset is intended solely to demonstrate the computational workflow and does not represent clinical evidence.

## Files

- `stress_testing_meta_analysis.py` — complete executable Python analysis script
- `README.md` — repository documentation

## Synthetic Dataset
The file synthetic_meta_analysis_data.csv contains 12 study synthetic dataset used for the demonstration in this technical report. The dataset is entirely synthetic and contains no real patient-level or clinical data.
The Python script and google Colab notebook reproduce the meta-analysis workflow using this dataset.

## Reproducibility

The script is intended to be run in a Python environment such as Google Colab. Users can replace the synthetic demonstration data with verified study-level data after prespecifying the analytical assumptions and sensitivity analyses.

## Citation

This repository accompanies the technical report:

**Stress-Testing Meta-Analyses: A Reproducible Python Framework for Robustness Assessment of Continuous Outcomes.**
