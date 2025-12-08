import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


TARGET_COL = "Pathfinding Success"
CATEGORICAL_COLS = ["Pathfinding Algorithm"]


def make_ohe():
    """
    Handle sklearn's OneHotEncoder API change:
    - new versions use sparse_output
    - older versions use sparse
    """
    try:
        # Newer sklearn (>=1.4)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Test-set predictions
    y_pred = pipeline.predict(X_test)

    # ---- Test metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC-AUC (probabilities preferred; fallback to decision_function)
    try:
        y_score = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = pipeline.decision_function(X_test)
    auc = roc_auc_score(y_test, y_score)

    # ---- Cross-validation on training data (5-fold)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n=== {name} ===")
    print(f"Test accuracy:        {acc:.3f}")
    print(f"5-fold CV accuracy:   {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"Precision:            {prec:.3f}")
    print(f"Recall:               {rec:.3f}")
    print(f"F1-score:             {f1:.3f}")
    print(f"ROC-AUC:              {auc:.3f}")

    return {
        "model": name,
        "test_accuracy": acc,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def main(csv_path: str):
    # ---- Load data
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Identify columns
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print("=== Dataset Info ===")
    print(f"CSV: {csv_path}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Numeric columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # ---- Preprocessor
    ohe = make_ohe()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )

    # ---- Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---- Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=None,  # you can tune this if you want
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
    }

    results = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", clf),
        ])
        res = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        results.append(res)

    # ---- Save results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("test_accuracy", ascending=False)

    out_name = "model_comparison_cv.csv"
    df_results.to_csv(out_name, index=False)

    print("\n=== Summary (saved to model_comparison_cv.csv) ===")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ML models with train/test and 5-fold CV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="pathfinding_robot_navigation_dataset_engineered.csv",
        help="Path to dataset CSV (original or engineered).",
    )
    args = parser.parse_args()
    main(args.csv)
