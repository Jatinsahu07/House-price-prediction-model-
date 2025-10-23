import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing # Using a modern alternative

# Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
# The target (price) is in the 'MedHouseVal' column
X = df.drop('MedHouseVal', axis=1) # Features
y = df['MedHouseVal'] # Target variable (Median House Value in 100,000s)

print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values check:")
print(df.isnull().sum())

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Optional: Visualize the distribution of the target variable
# plt.figure(figsize=(8, 6))
# sns.histplot(y, kde=True)
# plt.title('Distribution of Median House Value')
# plt.show()

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

print("\nModel training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation (Linear Regression) ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Optional: Visualize predictions vs. actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.3)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
# plt.xlabel('Actual Prices (y_test)')
# plt.ylabel('Predicted Prices (y_pred)')
# plt.title('Actual vs. Predicted House Prices')
# plt.show()


To do it more accurately 
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_y_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"\n--- Model Evaluation (Random Forest Regressor) ---")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R2 Score: {rf_r2:.4f}")

