import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score)

### NAIVE BAYES TRAIN/TEST ###
#------------------------------

### Data find

df = pd.read_csv("pathfinding_robot_navigation_dataset.csv")
#df = pd.read_csv("pathfinding_robot_navigation_dataset_engineered.csv")  # Optional engineered dataset

# Target
y = df["Pathfinding Success"]

# Features
X = df.drop(columns=["Pathfinding Success"])

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = ["Pathfinding Algorithm"]

### Split data into training and test set

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

### Preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),])

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("nb", GaussianNB())])

### Train final model

pipeline.fit(X_train, y_train)

### Test set predictions and report

y_pred = pipeline.predict(X_test)

print("\n=== FINAL TEST SET CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=3))

### Confusion matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()

### ROC curve and AUC
# Get probability scores for positive class
y_score = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_score)
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
