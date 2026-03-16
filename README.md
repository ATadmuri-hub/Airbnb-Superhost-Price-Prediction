# Advanced Modelling Final Project — Airbnb Superhost Classification & Price Prediction

**Course**: Advanced Modelling (UC3M, 2025-2026)
**Instructor**: Pablo Morala
**Author**: Abdullah Tadmuri (100502844)

## Why this project

Airbnb's publicly available listing data offers a rare opportunity to study how digital platforms shape real markets — how hosts behave, how guests choose, and how prices form across competitive local environments. This project uses that data to demonstrate the complete supervised learning workflow taught in the Advanced Modelling course, and to push beyond it.

The goal is threefold:

1. **Apply and compare a wide range of supervised learning methods** on a real-world dataset with meaningful complexity — class imbalance, mixed feature types, geographic heterogeneity, and non-linear relationships.
2. **Go beyond prediction accuracy** by treating interpretability as a first-class concern. Every model is not just evaluated on AUC or RMSE, but also interrogated through SHAP values, calibration curves, GAM smooth terms, counterfactual analysis, and model stacking — asking not just "how well does it predict?" but "what does it tell us about how this market works?"
3. **Demonstrate methodological rigour** in a complete, reproducible pipeline: leakage-safe preprocessing, proper train/test discipline, pairwise significance testing, and honest acknowledgment of what the models can and cannot tell us (predictive association vs. causation).

The two prediction tasks — Superhost classification and nightly price regression — were chosen because they capture fundamentally different aspects of the same platform. As the analysis reveals, what predicts *who a host is* (behaviour-driven) is almost entirely different from what predicts *what a listing costs* (property-driven). That contrast is one of the project's central findings.

## What this project demonstrates

- The full supervised learning pipeline from raw data to actionable insight
- How 21 models (13 classification, 8 regression) compare on the same data — and why the "best" model depends on what you care about (discrimination vs. calibration vs. interpretability)
- That feature engineering (neighbourhood intelligence, NLP signals) can meaningfully improve prediction beyond raw listing attributes
- How to use interpretability tools (SHAP, GAM smooths, counterfactuals) to translate model output into domain-relevant knowledge
- The importance of going beyond a single accuracy metric: calibration, threshold optimisation, price tier analysis, and city-stratified testing each reveal something that overall metrics hide
- A beyond-course experimental benchmark using Kolmogorov-Arnold Networks (KAN), a 2024 neural architecture

## Overview

The analysis applies these methods to Airbnb listing data from three European cities — **Madrid**, **Barcelona**, and **Amsterdam** — each with its own regulatory environment, tourism profile, and housing market dynamics:

1. **Classification** — Predicting whether a host holds Superhost status (binary: Yes/No)
2. **Regression** — Predicting nightly listing price (continuous, log-transformed)

### Key findings

- **Superhost status is a behaviour story** — host responsiveness, acceptance rate, and tenure are the strongest predictors across all model families
- **Pricing is a property story** — listing capacity, location, and neighbourhood context dominate price prediction, while host behaviour features contribute little
- **Non-linear models outperform linear baselines** in both tasks, with Random Forest and XGBoost leading the classification leaderboard
- **Calibration matters** — linear models produce better-calibrated probabilities despite lower AUC, which has implications for real-world deployment
- **Model agreement builds confidence** — when SHAP, GAM smooth terms, variable importance, and odds ratios all point to the same features, that signal is robust across fundamentally different modelling assumptions

## Project structure

```
.
├── final_project.Rmd        # Main analysis (R Markdown source)
├── style.css                 # Custom CSS theme for HTML output
├── kan_train.py              # Python script for KAN classification training
├── kan_reg_train.py          # Python script for KAN regression training
├── run_full_pipeline.sh      # Shell script to run the full pipeline
├── airbnb_data/              # Raw data (not tracked — see Data section)
│   ├── madrid_listings.csv.gz
│   ├── barcelona_listings.csv.gz
│   └── amsterdam_listings.csv.gz
└── README.md
```

## Data

Raw data is sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/) (November 2025 scrape). Due to file size (~1.2 GB uncompressed), the data is **not included in this repository**.

To reproduce the analysis:

1. Download the `listings.csv.gz` files for Madrid, Barcelona, and Amsterdam from Inside Airbnb
2. Place them in `airbnb_data/` with the naming convention `{city}_listings.csv.gz`
3. The Rmd will load and combine them automatically

## Models

### Classification (Superhost status)

| # | Model | Method |
|---|-------|--------|
| C1 | Logistic Regression | `glm` |
| C2 | Elastic Net | `glmnet` |
| C3 | Random Forest | `ranger` |
| C4 | XGBoost | `xgbTree` |
| C5 | GAM | `mgcv::bam` |
| C6 | Neural Network | `nnet` |
| C7 | Decision Tree | `rpart` (pruned) |
| C8 | LDA | `MASS::lda` |
| C9 | QDA | `MASS::qda` |
| C10 | KNN | `knn` |
| C11 | Naive Bayes | `naive_bayes` |
| C12 | SVM (Radial) | `kernlab::svmRadial` |
| C13 | KAN | `pykan` (external Python) |

### Regression (nightly price)

| # | Model | Method |
|---|-------|--------|
| R1 | Linear Regression | `lm` |
| R2 | Stepwise (AIC/BIC) | `step` |
| R3 | PCR & PLS | `pls` |
| R4 | Elastic Net | `glmnet` |
| R5 | Random Forest | `ranger` |
| R6 | XGBoost | `xgbTree` |
| R7 | GAM | `mgcv` via `caret` |
| R8 | KAN | `pykan` (external Python) |

## How to reproduce

### Prerequisites

- **R** >= 4.3.0
- **Python** >= 3.9 (only for KAN models — optional)
- R packages listed below

### R packages

```r
install.packages(c(
  # Core ML
  "tidyverse", "caret", "pROC", "ranger", "xgboost", "glmnet",
  "mgcv", "nnet", "rpart", "rpart.plot", "MASS", "naivebayes", "pls",
  "kernlab",          # SVM
  "caretEnsemble",    # OOF interface for stacking
  # Diagnostics / interpretation
  "car",              # VIF
  "shapviz", "pdp", "vip", "broom",
  # Visualisation
  "ggplot2", "patchwork", "corrplot", "gridExtra", "grid",
  "scales", "viridis", "ggrepel",
  "maps",               # required by ggplot2::map_data() — install but do not load
  # Utilities
  "kableExtra", "readr", "stringr", "purrr", "tibble", "tidyr", "dplyr",
  "janitor"
))
```

### Python packages (optional, for KAN only)

```bash
pip install pykan torch numpy pandas scikit-learn
```

### Running the analysis

**Option 1 — Knit in RStudio:**
Open `final_project.Rmd` in RStudio and click Knit (or run `rmarkdown::render("final_project.Rmd")`).

**Option 2 — Full pipeline (includes KAN):**
```bash
# 1. Generate KAN predictions (optional)
python kan_train.py
python kan_reg_train.py

# 2. Render the R Markdown
Rscript -e 'rmarkdown::render("final_project.Rmd")'
```

If the KAN prediction files (`kan_pred_test.csv`, `kan_reg_pred_test.csv`) are absent, the pipeline will skip KAN gracefully and produce results for all other models.

## Beyond-course methods

The following methods extend beyond the course curriculum:

- **Kolmogorov-Arnold Networks (KAN)** — a 2024 neural architecture with learnable spline activations
- **SHAP values** — per-observation feature attribution via TreeSHAP (XGBoost)
- **Model stacking** — logistic/OLS meta-learner combining out-of-fold predictions
- **GAM smooth terms** — non-linear partial effects via penalised splines
- **Counterfactual analysis** — cumulative "what-if" scenarios for Superhost pathway
- **Calibration analysis** — reliability diagrams comparing predicted vs observed probabilities
- **Prediction diversity analysis** — Spearman correlation heatmap of ensemble member predictions

## License

This project is submitted as coursework for UC3M's Advanced Modelling course (2025-2026). The data is provided by Inside Airbnb under their terms of use.
