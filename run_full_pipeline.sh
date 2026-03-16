#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Runner — Airbnb Advanced Modeling Project (Nov 2025 Data)
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT THIS DOES:
#   1. Knits the Rmd (data processing + all models EXCEPT KAN)
#   2. Re-knits with KAN export enabled → generates x_kan_*.csv files
#   3. Runs KAN classification in Python → kan_pred_test.csv
#   4. Runs KAN regression in Python → kan_reg_pred_test.csv
#   5. Re-knits final time to incorporate KAN predictions into report
#
# PREREQUISITES:
#   - R with: tidyverse, caret, glmnet, ranger, xgboost, mgcv, nnet,
#             shapviz, pROC, scales, kableExtra, reticulate
#   - Python with: numpy, pandas, torch, pykan
#   - Working directory = this project folder
#
# USAGE:
#   cd "/path/to/Advanced Modeling Final Project 2026"
#   bash run_full_pipeline.sh
#
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════"
echo "  STEP 1: Knit Rmd (all models except KAN)"
echo "═══════════════════════════════════════════════════════════════"

# First pass: process data + train all caret models + GAM
# KAN export flags are FALSE by default → KAN skipped gracefully
Rscript -e '
  rmarkdown::render(
    "final_project.Rmd",
    output_file = "final_project_step1.html",
    envir = new.env(),
    quiet = FALSE
  )
'
echo "✓ Step 1 complete — base models trained"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  STEP 2: Re-knit with KAN CSV export enabled"
echo "═══════════════════════════════════════════════════════════════"

# Second pass: enable KAN export to generate x_kan_*.csv, y_kan_*.csv
Rscript -e '
  # Override the flags before rendering
  env <- new.env()
  env$run_kan_export <- TRUE
  env$run_kan_reg_export <- TRUE
  rmarkdown::render(
    "final_project.Rmd",
    output_file = "final_project_step2.html",
    envir = env,
    quiet = FALSE
  )
'
echo "✓ Step 2 complete — KAN CSV files exported"

# Verify exports exist
for f in x_kan_train.csv y_kan_train.csv x_kan_test.csv \
         x_kan_reg_train.csv y_kan_reg_train.csv x_kan_reg_test.csv; do
  if [ ! -f "$f" ]; then
    echo "✗ Missing: $f — KAN export failed"
    exit 1
  fi
done
echo "✓ All KAN CSV files present"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  STEP 3: Train KAN (classification) in Python"
echo "═══════════════════════════════════════════════════════════════"

KAN_PYTHON="$HOME/venvs/r-kan311/bin/python"
$KAN_PYTHON kan_train.py
if [ ! -f "kan_pred_test.csv" ]; then
  echo "✗ KAN classification predictions not generated"
  exit 1
fi
echo "✓ KAN classification predictions saved"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  STEP 4: Train KAN (regression) in Python"
echo "═══════════════════════════════════════════════════════════════"

$KAN_PYTHON kan_reg_train.py
if [ ! -f "kan_reg_pred_test.csv" ]; then
  echo "✗ KAN regression predictions not generated"
  exit 1
fi
echo "✓ KAN regression predictions saved"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  STEP 5: Final knit with KAN predictions incorporated"
echo "═══════════════════════════════════════════════════════════════"

# Final pass: KAN export OFF (CSVs already exist), models read kan_pred_test.csv
Rscript -e '
  rmarkdown::render(
    "final_project.Rmd",
    output_file = "final_project.html",
    envir = new.env(),
    quiet = FALSE
  )
'
echo "✓ Final report generated: final_project.html"

# Cleanup intermediate HTML files
rm -f final_project_step1.html final_project_step2.html

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  final_project.html      — Full report with all 13+8=21 models"
echo "  airbnb_processed.csv    — Cleaned Nov 2025 dataset"
echo "  kan_pred_test.csv       — KAN classification predictions"
echo "  kan_reg_pred_test.csv   — KAN regression predictions"
echo ""
