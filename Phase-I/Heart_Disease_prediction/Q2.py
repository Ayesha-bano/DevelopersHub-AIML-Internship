import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, classification_report
)

# Load the dataset
df = pd.read_csv('heart_disease_cleveland.csv')
 
print("Shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Handling missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())
 
   # fill column with mean
df.fillna(df.mean(), inplace=True)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)
print("\nDescriptive statistics:")
print(df.describe())

   # Target distribution (0 = no disease, 1 = disease)
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution (0 = No, 1 = Yes)')
plt.tight_layout()
plt.savefig('target_distribution.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Age distribution by target
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x='age', hue='target', kde=True, bins=20)
plt.title('Age Distribution by Heart Disease Outcome')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()

# Train a classification model
X = df.drop(columns=['target'])
y = df['target']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# Scale features (helps Logistic Regression converge well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# --- Logistic Regression ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]
 
# --- Decision Tree ---
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
tree_probs = tree_model.predict_proba(X_test)[:, 1]
 
 #  Evaluate models:— accuracy, confusion matrix, ROC curve

def evaluate_model(name, y_test, preds, probs):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
 
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.show()
 
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'roc_curve_{name.replace(" ", "_")}.png')
    plt.show()
 
evaluate_model("Logistic Regression", y_test, log_preds, log_probs)
evaluate_model("Decision Tree", y_test, tree_preds, tree_probs)
 

# Highlight important features

# Logistic Regression coefficients (magnitude = importance)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)
 
print("\nLogistic Regression Feature Importance (by coefficient magnitude):")
print(coef_df)
 
plt.figure(figsize=(8, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title('Logistic Regression Feature Importance')
plt.tight_layout()
plt.savefig('logreg_feature_importance.png')
plt.show()
 
# Decision Tree feature importance
tree_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tree_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
 
print("\nDecision Tree Feature Importance:")
print(tree_importance_df)
 
plt.figure(figsize=(8, 6))
sns.barplot(data=tree_importance_df, x='Importance', y='Feature')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.savefig('tree_feature_importance.png')
plt.show()