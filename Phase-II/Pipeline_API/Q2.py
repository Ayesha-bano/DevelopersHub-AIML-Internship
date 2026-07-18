import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

# ============================================================
# Step 1: Load & inspect data
# ============================================================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# TotalCharges looks numeric but is stored as text with some blank entries
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# ============================================================
# Step 2: Train/test split
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# Step 3: Preprocessing with Pipeline + ColumnTransformer
# ============================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X.columns if col not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ============================================================
# Step 4: Model pipelines
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])

# ============================================================
# Step 5: Hyperparameter tuning with GridSearchCV
# ============================================================
from sklearn.model_selection import GridSearchCV

# NOTE: removed "classifier__penalty": ["l2"] — it's already the default in
# current scikit-learn, and explicitly setting it triggers a FutureWarning.
log_reg_params = {
    "classifier__C": [0.01, 0.1, 1, 10],
}
log_reg_grid = GridSearchCV(log_reg_pipeline, log_reg_params, cv=5, scoring="f1", n_jobs=-1)
log_reg_grid.fit(X_train, y_train)

rf_params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4],
}
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train, y_train)

# ============================================================
# Step 6: Evaluate on test set
# ============================================================
from sklearn.metrics import classification_report, accuracy_score, f1_score

log_reg_preds = log_reg_grid.best_estimator_.predict(X_test)
rf_preds = rf_grid.best_estimator_.predict(X_test)

log_reg_f1 = f1_score(y_test, log_reg_preds)
rf_f1 = f1_score(y_test, rf_preds)

for name, grid, preds in [
    ("Logistic Regression", log_reg_grid, log_reg_preds),
    ("Random Forest", rf_grid, rf_preds),
]:
    print(f"\n=== {name} ===")
    print("Best params:", grid.best_params_)
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# ============================================================
# Step 7: Export the best-performing pipeline with joblib
# ============================================================
import joblib

# NOTE: this now picks automatically based on test F1 score,
# instead of always hardcoding Random Forest.
if rf_f1 >= log_reg_f1:
    best_overall = rf_grid.best_estimator_
    chosen_name = "Random Forest"
    chosen_f1 = rf_f1
else:
    best_overall = log_reg_grid.best_estimator_
    chosen_name = "Logistic Regression"
    chosen_f1 = log_reg_f1

print(f"\nChosen model: {chosen_name} (Test F1: {chosen_f1:.4f})")

joblib.dump(best_overall, "churn_pipeline.joblib")
print("Saved to churn_pipeline.joblib")