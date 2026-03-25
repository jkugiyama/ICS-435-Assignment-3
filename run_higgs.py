import os
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from sklearn import metrics

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


TRAIN_PATH = "ICS-435-Assignment-3/train.csv"
TEST_PATH = "ICS-435-Assignment-3/test.csv"
SUBMISSION_PATH = "ICS-435-Assignment-3/sample_submission.csv"

def find_label_column(df: pd.DataFrame) -> str:
    candidates = ["Label", "label", "target", "Target", "Class", "class", "y"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a label column in train.csv. "
        f"Expected one of: {candidates}"
    )


def find_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["EventId", "Id", "id"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def encode_labels(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        mapping = {
            "s": 1, "signal": 1, "sig": 1, "higgs": 1, "true": 1, "yes": 1, "1": 1,
            "b": 0, "background": 0, "bg": 0, "not_higgs": 0, "false": 0, "no": 0, "0": 0,
        }
        y_encoded = y.astype(str).str.strip().str.lower().map(mapping)
        if y_encoded.isna().any():
            unknown = sorted(y.astype(str).str.strip().str.lower()[y_encoded.isna()].unique())
            raise ValueError(f"Unsupported label values found: {unknown}")
        return y_encoded.astype(int)

    unique_vals = sorted(pd.Series(y).dropna().unique())
    if set(unique_vals).issubset({0, 1}):
        return y.astype(int)

    if len(unique_vals) == 2:
        low, high = unique_vals[0], unique_vals[1]
        return y.map({low: 0, high: 1}).astype(int)

    raise ValueError("Label column must be binary.")


def build_feature_matrix(df: pd.DataFrame, label_col: Optional[str] = None):
    drop_cols = []
    if label_col and label_col in df.columns:
        drop_cols.append(label_col)

    id_col = find_id_column(df)
    if id_col:
        drop_cols.append(id_col)

    if "Weight" in df.columns:
        drop_cols.append("Weight")

    X = df.drop(columns=drop_cols, errors="ignore")
    return X, id_col


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    submission_df = pd.read_csv(SUBMISSION_PATH)

    label_col = find_label_column(train_df)

    X, train_id_col = build_feature_matrix(train_df, label_col=label_col)
    y_all = encode_labels(train_df[label_col])

    X_test, test_id_col = build_feature_matrix(test_df)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = [c for c in X.columns if c not in numeric_features]

    if non_numeric_features:
        raise ValueError(
            f"Non-numeric feature columns found: {non_numeric_features}. "
            f"Please encode them before training."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_all,
        test_size=0.2,
        stratify=y_all,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    print("Validation Accuracy:", round(accuracy_score(y_val, val_pred), 4))
    print("Validation ROC AUC:", round(roc_auc_score(y_val, val_prob), 4))
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=["Not Higgs", "Higgs"]))

    demo_y = np.array([1, 1, 2, 2])
    demo_pred = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(demo_y, demo_pred, pos_label=2)
    auc_score = metrics.auc(fpr, tpr)
    print(auc_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Higgs vs Not Higgs ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Refit on all training data before predicting test data
    model.fit(X, y_all)

    # Predict probabilities
    test_prob = model.predict_proba(X_test)[:, 1]

    # Get IDs
    if test_id_col is None:
        ids = test_df.index.to_numpy()
    else:
        ids = test_df[test_id_col].to_numpy()

    # Convert BOTH to float
    ids = ids.astype(float)
    test_prob = test_prob.astype(float)

    # Stack
    predictions = np.column_stack((ids, test_prob))

    # Save in scientific notation
    np.savetxt(
        fname="sample_submission.csv",
        X=predictions,
        header="Id,Predicted",
        delimiter=",",
        comments="",
        fmt="%.18e"
    )

    submission = pd.DataFrame({
        "Id": ids.astype(str),
        "Predicted": test_prob
    })
    submission.to_csv(os.path.join("ICS-435-Assignment-3", "sample_submission.csv"), index=False)

if __name__ == "__main__":
    main()
