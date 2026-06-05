import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
train_test_split,
GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score
)

# -----------------------------

# Load Data

# -----------------------------

df = pd.read_csv(
"data/processed/processed_data.csv"
)

# Remove CustomerId if present

if "CustomerId" in df.columns:
df = df.drop(
columns=["CustomerId"]
)

# Features and Target

X = df.drop(
columns=["is_high_risk"]
)

y = df["is_high_risk"]

# -----------------------------

# Train-Test Split

# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# -----------------------------

# MLflow Experiment

# -----------------------------

mlflow.set_experiment(
"credit-risk-model"
)

results = {}

# ==================================================

# Logistic Regression

# ==================================================

with mlflow.start_run(
run_name="Logistic_Regression"
):

```
lr = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr.fit(
    X_train,
    y_train
)

predictions = lr.predict(
    X_test
)

probabilities = lr.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

results["LogisticRegression"] = {
    "model": lr,
    "roc_auc": roc_auc
}

mlflow.log_param(
    "model",
    "LogisticRegression"
)

mlflow.log_metric(
    "accuracy",
    accuracy
)

mlflow.log_metric(
    "precision",
    precision
)

mlflow.log_metric(
    "recall",
    recall
)

mlflow.log_metric(
    "f1_score",
    f1
)

mlflow.log_metric(
    "roc_auc",
    roc_auc
)

mlflow.sklearn.log_model(
    lr,
    artifact_path="logistic_regression"
)
```

# ==================================================

# Random Forest + Grid Search

# ==================================================

with mlflow.start_run(
run_name="Random_Forest_GridSearch"
):

```
rf = RandomForestClassifier(
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_rf = grid_search.best_estimator_

predictions = best_rf.predict(
    X_test
)

probabilities = best_rf.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

results["RandomForest"] = {
    "model": best_rf,
    "roc_auc": roc_auc
}

mlflow.log_param(
    "model",
    "RandomForest"
)

mlflow.log_params(
    grid_search.best_params_
)

mlflow.log_metric(
    "accuracy",
    accuracy
)

mlflow.log_metric(
    "precision",
    precision
)

mlflow.log_metric(
    "recall",
    recall
)

mlflow.log_metric(
    "f1_score",
    f1
)

mlflow.log_metric(
    "roc_auc",
    roc_auc
)

mlflow.sklearn.log_model(
    best_rf,
    artifact_path="random_forest",
    registered_model_name="CreditRiskModel"
)
```

# ==================================================

# Automatic Best Model Selection

# ==================================================

best_model_name = max(
results,
key=lambda x: results[x]["roc_auc"]
)

best_model = results[
best_model_name
]["model"]

best_auc = results[
best_model_name
]["roc_auc"]

print(
f"Best Model: {best_model_name}"
)

print(
f"ROC-AUC: {best_auc:.4f}"
)

# ==================================================

# Save Best Model

# ==================================================

os.makedirs(
"models",
exist_ok=True
)

joblib.dump(
best_model,
"models/best_model.pkl"
)

print(
"Best model saved to models/best_model.pkl"
)

print(
"Training complete."
)



    


