import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

TARGET_COL = "Pathfinding Success"
ALGO_COL = "Pathfinding Algorithm"

def save_confusion_matrix(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail (0)", "Success (1)"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)

def save_roc_curve(y_true, y_prob, title, fname):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return roc_auc

def main(csv_path: str, test_size: float = 0.2, seed: int = 42):
    # ---- Load
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found.")
    if ALGO_COL not in df.columns:
        raise ValueError(f"Expected categorical column '{ALGO_COL}' not found.")

    # ---- Features/target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    cat_cols = [ALGO_COL]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # ---- Preprocess
    # NOTE: GaussianNB expects dense arrays; set OneHotEncoder(sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


    # ---- Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ---- Build pipeline
    nb_clf = Pipeline(steps=[
        ("pre", pre),
        ("nb", GaussianNB())
    ])

    # ---- Train
    nb_clf.fit(X_train, y_train)

    # ---- Predict + Metrics
    y_pred = nb_clf.predict(X_test)
    print("\n=== Gaussian Naïve Bayes ===")
    print(classification_report(y_test, y_pred, digits=3))

    # ---- Confusion Matrix
    save_confusion_matrix(y_test, y_pred,
                          "Confusion Matrix — GaussianNB",
                          "naive_bayes_confusion.png")
    print("Saved: naive_bayes_confusion.png")

    # ---- ROC Curve
    y_prob = nb_clf.predict_proba(X_test)[:, 1]
    auc_val = save_roc_curve(y_test, y_prob,
                             "ROC Curve — GaussianNB",
                             "naive_bayes_roc.png")
    print(f"Saved: naive_bayes_roc.png (AUC={auc_val:.3f})")

    # ---- Export per-class feature means/variances after preprocessing
    # Recompute on TRAIN SET to avoid peeking at test.
    Xtr_proc = nb_clf.named_steps["pre"].fit_transform(X_train)
    # Fit a fresh GaussianNB on the processed arrays to access theta_/var_
    nb_raw = GaussianNB()
    nb_raw.fit(Xtr_proc, y_train)

    # Recover feature names
    ohe = nb_clf.named_steps["pre"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    feat_names = num_cols + cat_names

    stats = []
    classes = nb_raw.classes_
    # GaussianNB stores class-wise means (theta_) and variances (var_)
    for ci, cls in enumerate(classes):
        for fi, fname in enumerate(feat_names):
            stats.append({
                "class": int(cls),
                "feature": fname,
                "mean": nb_raw.theta_[ci, fi],
                "var": nb_raw.var_[ci, fi]
            })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("nb_feature_stats.csv", index=False)
    print("Saved: nb_feature_stats.csv (per-class means/variances after preprocessing)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Naïve Bayes on Robot Navigation dataset")
    parser.add_argument("--csv", type=str, default="pathfinding_robot_navigation_dataset_engineered.csv",
                        help="Path to the CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.csv, test_size=args.test_size, seed=args.seed)
