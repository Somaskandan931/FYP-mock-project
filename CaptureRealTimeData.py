import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os


def train_model(eye_tracking_data, speech_data):
    # Merge datasets on Participant_ID
    combined_data = pd.merge(eye_tracking_data, speech_data, on='Participant_ID')

    # Features and target variable
    X = combined_data[
        ['Fixation_Duration', 'Saccadic_Amplitude', 'Saccadic_Velocity', 'Speech_Rate', 'Pitch_Variability']]
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

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()

    # Feature importance analysis
    feature_importances = model.feature_importances_

    return model, scaler, accuracy, feature_importances


def save_model(model, scaler):
    save_dir = 'model_files'
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    return os.path.join(save_dir, 'random_forest_model.pkl'), os.path.join(save_dir, 'scaler.pkl')


def main():
    st.title("ADHD Prediction Model Training")

    st.write("""
    Upload your synthetic datasets for training the ADHD prediction model. The datasets should include eye tracking data and speech data.
    """)

    uploaded_eye_tracking_file = st.file_uploader("Upload Eye Tracking Data CSV", type=["csv"])
    uploaded_speech_file = st.file_uploader("Upload Speech Data CSV", type=["csv"])

    if uploaded_eye_tracking_file and uploaded_speech_file:
        eye_tracking_data = pd.read_csv(uploaded_eye_tracking_file)
        speech_data = pd.read_csv(uploaded_speech_file)

        st.write("Data successfully uploaded!")

        model, scaler, accuracy, feature_importances = train_model(eye_tracking_data, speech_data)

        st.write(f"Model accuracy: {accuracy:.2f}")

        st.write("Feature Importances:")
        for feature, importance in zip(
                ['Fixation_Duration', 'Saccadic_Amplitude', 'Saccadic_Velocity', 'Speech_Rate', 'Pitch_Variability'],
                feature_importances):
            st.write(f"Feature: {feature}, Importance: {importance:.4f}")

        model_path, scaler_path = save_model(model, scaler)
        st.write(f"Model and scaler saved successfully.")
        st.write(f"Model saved at: {model_path}")
        st.write(f"Scaler saved at: {scaler_path}")


if __name__ == "__main__":
    main()
