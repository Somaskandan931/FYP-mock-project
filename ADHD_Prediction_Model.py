import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load your synthetic datasets with error handling
try:
    eye_tracking_data = pd.read_csv("C:/Users/somas/PycharmProjects/FYP_mock_project/synthetic_adhd_dataset.csv")
    speech_data = pd.read_csv("C:/Users/somas/PycharmProjects/FYP_mock_project/synthetic_speech_dataset.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Ensure 'Participant_ID' exists in both DataFrames before merging
if 'Participant_ID' not in eye_tracking_data.columns or 'Participant_ID' not in speech_data.columns:
    print("Participant_ID column missing in one of the datasets.")
    exit()

# Merge datasets on Participant_ID
combined_data = pd.merge(eye_tracking_data, speech_data, on='Participant_ID')

# Print all columns to find the correct ones
print("Columns in the dataset:", combined_data.columns)

# Adjust these column names as per your dataset
required_columns = ['Fixation_Duration', 'Saccadic_Amplitude', 'Saccadic_Velocity', 'Speech_Rate', 'Pitch_Variability']

# Ensure all required columns are present
for col in required_columns:
    if col not in combined_data.columns:
        print(f"Column {col} is missing from the combined dataset.")
        exit()

# Features and target variable
X = combined_data[required_columns]
y = combined_data['Label_x']  # Assuming Label_x is your target variable for ADHD prediction

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model (optional, for your own analysis)
y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"Model accuracy: {accuracy:.2f}")

# Feature importance analysis
feature_importances = model.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# Save the trained model and scaler
save_dir = "C:/Users/somas/PycharmProjects/FYP_mock_project/model_files"
os.makedirs(save_dir, exist_ok=True)
joblib.dump(model, os.path.join(save_dir, 'random_forest_model.pkl'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

print("Model and scaler saved successfully.")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Example: Using GridSearch to tune hyperparameters of a GradientBoostingClassifier
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"Model accuracy after tuning: {accuracy:.2f}")
