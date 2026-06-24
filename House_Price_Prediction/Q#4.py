# Loading dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('house_prices_dataset.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Pre-process the features like square footage, bedrooms location
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Here Regression is applying
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Linear Regression (uses scaled features)
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
lin_preds = lin_model.predict(X_test_scaled)

# Gradient Boosting (tree-based models don't need scaling)
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)


# Evaluate with MAE and RMSE
# -MAE (Mean Absolute Error) — average dollar amount your prediction is off by
# -RMSE (Root Mean Squared Error) — useful for spotting models that occasionally make huge errors
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(name, y_true, preds):
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}\n")

evaluate("Linear Regression", y_test, lin_preds)
evaluate("Gradient Boosting", y_test, gb_preds)

# Visulize and predict actual values
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, lin_preds, alpha=0.5, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title('Linear Regression: Predicted vs Actual')

axes[1].scatter(y_test, gb_preds, alpha=0.5, color='seagreen')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title('Gradient Boosting: Predicted vs Actual')

plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.show()