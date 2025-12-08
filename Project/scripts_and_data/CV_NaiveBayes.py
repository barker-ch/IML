import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

### NAIVE BAYES 5 FOLD CROSS VALIDATION ###
#------------------------------------------
### Data sourcing

#df = pd.read_csv("pathfinding_robot_navigation_dataset.csv")
df = pd.read_csv("pathfinding_robot_navigation_dataset_engineered.csv")  # engineered dataset

# Target
y = df["Pathfinding Success"]

# Features
X = df.drop(columns=["Pathfinding Success"])

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = ["Pathfinding Algorithm"]


### Split data 80-20 Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)


### Preprocess pipeline forming

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),])

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("nb", GaussianNB())])

### 5 fold cross validation prediction and report

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv)

print("=== 5-FOLD CROSS-VALIDATION CLASSIFICATION REPORT ===")
print(classification_report(y_train, y_train_pred))

### train final model
pipeline.fit(X_train, y_train)

### test set prediction

y_pred = pipeline.predict(X_test)

print("\n=== FINAL TEST SET CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

### Summary of prediction

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# ROC-AUC (probabilities preferred; fallback to decision_function)
try:
    y_score = pipeline.predict_proba(X_test)[:, 1]
except Exception:
    y_score = pipeline.decision_function(X_test)

roc_auc = roc_auc_score(y_test, y_score)


# 5-fold CV accuracy
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print("\n=== METRICS SUMMARY (NAIVE BAYES) ===")
print(f"Test Accuracy:        {acc:.3f}")
print(f"5-Fold CV Accuracy:   {cv_mean:.3f} ± {cv_std:.3f}")
print(f"Precision:            {prec:.3f}")
print(f"Recall:               {rec:.3f}")
print(f"F1-score:             {f1:.3f}")
print(f"ROC-AUC:              {roc_auc:.3f}")


### Confusion matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()


### ROC curve

fpr, tpr, thresholds = roc_curve(y_test, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
