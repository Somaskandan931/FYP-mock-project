import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
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

# Build a pipeline for feature selection and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for tuning
param_grid = {
    'feature_selection__n_features_to_select': [3, 4, 5],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Model accuracy after tuning: {accuracy:.2f}")

# Save the trained model and scaler
save_dir = "C:/Users/somas/PycharmProjects/FYP_mock_project/model_files"
os.makedirs(save_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(save_dir, 'best_random_forest_model.pkl'))

print("Model saved successfully.")
