🏡 House Price Prediction Model Summary
This project uses Python and scikit-learn (sklearn) to build and evaluate models for predicting house prices (using the California Housing Dataset)

core libraries 
Library Role
pandas, numpy Data loading, manipulation, and numerical operations.
scikit-learn The primary machine learning library for data splitting, model training, and evaluation.

🧠 Workflow in 5 Steps
Load & Prepare Data: Load the California Housing dataset into a DataFrame and check for missing values.
Split Data: Divide the features (X) and the target variable (price, y) into training (80%) and testing (20%) sets using train_test_split.
Train Models:
Initialize and train a Linear Regression model (as a baseline).
Initialize and train a Random Forest Regressor (for better performance).
Predict: Use the trained models to make price predictions on the unseen testing data.
Evaluate: Measure model performance using metrics like Root Mean Squared Error (RMSE) and R-squared (R^2 score) to determine accuracy. The Random Forest model typically achieves the better R^2 score.
