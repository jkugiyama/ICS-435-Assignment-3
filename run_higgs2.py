import os
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

TRAIN_PATH = "ICS-435-Assignment-3/train.csv"
TEST_PATH = "ICS-435-Assignment-3/test.csv"


def find_label_column(df: pd.DataFrame) -> str:
    candidates = ["Label", "label", "target", "Target", "Class", "class", "y"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find label column. Expected one of: {candidates}")


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
        y_enc = y.astype(str).str.strip().str.lower().map(mapping)
        if y_enc.isna().any():
            unknown = sorted(y.astype(str).str.strip().str.lower()[y_enc.isna()].unique())
            raise ValueError(f"Unsupported label values: {unknown}")
        return y_enc.astype(int)

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

    label_col = find_label_column(train_df)

    X, _ = build_feature_matrix(train_df, label_col=label_col)
    y = encode_labels(train_df[label_col])

    X_test, test_id_col = build_feature_matrix(test_df)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = [c for c in X.columns if c not in numeric_features]
    if non_numeric_features:
        raise ValueError(f"Non-numeric features found: {non_numeric_features}")

    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), numeric_features)],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    # Holdout set for final sanity check
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_distributions = {
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "classifier__max_iter": [200, 300, 500, 700],
        "classifier__max_depth": [None, 4, 6, 8, 10],
        "classifier__min_samples_leaf": [10, 20, 40, 80, 120],
        "classifier__l2_regularization": [0.0, 0.01, 0.1, 0.5, 1.0],
        "classifier__max_bins": [127, 255],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest CV ROC AUC:", round(search.best_score_, 5))
    print("Best params:", search.best_params_)

    val_pred = best_model.predict(X_val)
    val_prob = best_model.predict_proba(X_val)[:, 1]

    print("\nValidation Accuracy:", round(accuracy_score(y_val, val_pred), 5))
    print("Validation ROC AUC:", round(roc_auc_score(y_val, val_prob), 5))
    print(classification_report(y_val, val_pred, target_names=["Not Higgs", "Higgs"]))

    # Refit on full training set with best params
    best_model.fit(X, y)

    test_prob = best_model.predict_proba(X_test)[:, 1]

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
        fname="sample_submission2.csv",
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
    submission.to_csv(os.path.join("ICS-435-Assignment-3", "sample_submission2.csv"), index=False)

if __name__ == "__main__":
    main()
