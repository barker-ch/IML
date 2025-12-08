import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# ---------- helpers ----------
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

def evaluate_and_save(name, pipe, X_train, X_test, y_train, y_test, out_rows):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion Matrix
    save_confusion_matrix(y_test, y_pred,
                          f"Confusion Matrix — {name}",
                          f"{name.lower().replace(' ', '_')}_confusion.png")
    # ROC
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        # SVC without probability=True would land here; but we set probability=True.
        y_prob = None

    if y_prob is not None:
        auc_val = save_roc_curve(
            y_test, y_prob,
            f"ROC Curve — {name}",
            f"{name.lower().replace(' ', '_')}_roc.png"
        )
    else:
        auc_val = np.nan

    out_rows.append({
        "model": name, "accuracy": acc,
        "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc_val
    })
    return pipe

def build_preprocessor(df):
    algo_col = "Pathfinding Algorithm"
    target_col = "Pathfinding Success"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    cat_cols = [algo_col]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return X, y, pre, num_cols, cat_cols

def save_rf_feature_importance(fitted_pipe, pre, num_cols, cat_cols):
    # Build feature names from preprocessor
    ohe = pre.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    feat_names = num_cols + cat_names

    rf = fitted_pipe.named_steps["clf"]
    if hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        fi = pd.DataFrame({"feature": feat_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv("rf_feature_importance.csv", index=False)
        # Bar plot top 20
        topk = fi.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(topk["feature"][::-1], topk["importance"][::-1])
        ax.set_title("Random Forest — Top 20 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig("rf_feature_importance.png", dpi=200)
        plt.close(fig)
        print("Saved: rf_feature_importance.csv, rf_feature_importance.png")

def save_pca_plot(df):
    # Make a quick PCA visualization (unsupervised) on preprocessed features
    target_col = "Pathfinding Success"
    algo_col = "Pathfinding Algorithm"
    y = df[target_col].astype(int)

    X = df.drop(columns=[target_col])
    num_cols = [c for c in X.columns if c != algo_col]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [algo_col]),
        ]
    )
    Xp = pre.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xp.toarray() if hasattr(Xp, "toarray") else Xp)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=16, alpha=0.8, cmap="coolwarm")
    ax.set_title("PCA (2D) of Preprocessed Features — colored by Success")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1])
    cb.set_label("Success (0=Fail, 1=Success)")
    fig.tight_layout()
    fig.savefig("pca_2d_success.png", dpi=200)
    plt.close(fig)
    print("Saved: pca_2d_success.png")

def main(csv_path, test_size=0.2, seed=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Build preprocessor and split
    X, y, pre, num_cols, cat_cols = build_preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    results = []

    # ---------- Models ----------
    # 1) Logistic Regression (baseline)
    logreg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    evaluate_and_save("Logistic Regression", logreg, X_train, X_test, y_train, y_test, results)

    # 2) Decision Tree
    dt = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=5, random_state=seed))]
    )
    evaluate_and_save("Decision Tree", dt, X_train, X_test, y_train, y_test, results)

    # 3) Random Forest (bagging)
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=seed, n_jobs=-1))]
    )
    rf_fitted = evaluate_and_save("Random Forest", rf, X_train, X_test, y_train, y_test, results)

    # Save RF feature importance
    # (Need a fitted preprocessor to access names; reuse the one inside pipeline)
    save_rf_feature_importance(rf_fitted, rf_fitted.named_steps["pre"], num_cols, cat_cols)

    # 4) SVM (RBF)
    svm = Pipeline([("pre", pre), ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=seed))])
    evaluate_and_save("SVM (RBF)", svm, X_train, X_test, y_train, y_test, results)

    # ---------- Compare & save table ----------
    df_res = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print("\n=== Model Comparison ===")
    print(df_res.to_string(index=False))
    df_res.to_csv("model_comparison.csv", index=False)
    print("Saved: model_comparison.csv")

    # ---------- PCA Visualization ----------
    save_pca_plot(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare classic classifiers on Robot Navigation dataset")
    # parser.add_argument("--csv", type=str, default="pathfinding_robot_navigation_dataset.csv")
    parser.add_argument("--csv", type=str, default="pathfinding_robot_navigation_dataset_engineered.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.csv, test_size=args.test_size, seed=args.seed)