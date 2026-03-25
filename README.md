# ICS-435-Assignment-3
# Machine Learning Models Evaluation
## Assignment Overview
This assignment focuses on building a machine learning model to classify whether a particle collision event produces a Higgs Boson or not. The task is a binary classification problem using a structured dataset with numerical features describing particle decay properties.

A Histogram-Based Gradient Boosting model is trained and optimized, and its performance is evaluated using appropriate classification metrics. The goal is to maximize predictive performance, particularly in terms of AUROC.
## Dataset
Dataset: Higgs Boson dataset

Training set: 50k samples with 28 features + 1 label column
Test set: 50k samples with 28 features (no labels)
Features: All numerical
Task: Binary classification (Higgs signal vs. background)

The dataset is split into 80% training and 20% validation using stratified sampling to preserve class distribution.


## Preprocessing
Non-feature columns such as ID and Weight are removed
Missing values are handled using median imputation
Only numerical features are used (no categorical encoding required)
Labels are encoded into binary format:
1 = Higgs (signal)
0 = Background

A preprocessing pipeline ensures that all transformations are applied consistently.

## Models Implemented

Final Model
Histogram-Based Gradient Boosting Classifier

This model was chosen for its strong performance on tabular data and ability to model nonlinear relationships efficiently.

Other Model Tested
Random Forest Classifier

## Evaluation Metrics

The model is evaluated using:

ROC-AUC (primary metric)
Accuracy
Precision
Recall
F1-score

ROC-AUC is emphasized because it measures how well the model ranks predictions across all thresholds.

## Results

Strong validation performance with high AUROC
Final Kaggle submission score: 0.80311

The Histogram-Based Gradient Boosting model outperformed the Random Forest model, particularly in terms of AUROC and generalization performance.

## How to Run

Make sure required packages are installed:
```
pip install numpy pandas matplotlib scikit-learn
```

Then run:
```
python run_higgs2.py
```

The script will:
Train the model
Perform hyperparameter tuning
Output validation metrics
Generate a submission file (sample_submission2.csv)
