import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import speech_recognition as sr
import numpy as np
import time

def train_model(eye_tracking_data, speech_data):
    # Merge datasets on Participant_ID
    combined_data = pd.merge(eye_tracking_data, speech_data, on='Participant_ID')

    # Features and target variable
    X = combined_data[['Fixation_Duration', 'Saccadic_Amplitude', 'Saccadic_Velocity', 'Speech_Rate', 'Pitch_Variability']]
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


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.fixation_duration = 0
        self.saccadic_amplitude = 0
        self.saccadic_velocity = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Simulate eye-tracking data
        self.fixation_duration = np.random.rand()
        self.saccadic_amplitude = np.random.rand()
        self.saccadic_velocity = np.random.rand()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_eye_tracking_data(self):
        return self.fixation_duration, self.saccadic_amplitude, self.saccadic_velocity


def collect_audio_data():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("Calibrating microphone... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        st.info("Recording... Please read the passage.")
        audio = recognizer.listen(source, timeout=5)

    try:
        st.info("Processing audio data...")
        # Simulate speech data extraction
        speech_rate = np.random.rand()
        pitch_variability = np.random.rand()
        return speech_rate, pitch_variability
    except Exception as e:
        st.error(f"Could not process audio: {e}")
        return None, None


def main():
    st.title("ADHD Prediction Model")

    st.write("""
    Read the following passage aloud while the application captures data using your webcam and microphone.
    """)

    passage = """
    In a village, there was a wise old man. People from all over came to him with their problems, and he helped them find solutions.
    One day, a young man came to him with a problem he couldn't solve. The old man listened carefully and then told him a story. 
    """
    st.text_area("Reading Passage", passage, height=150)

    # Webcam stream initialization
    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if st.button("Start Capture"):
        st.write("Capturing data... Please continue reading.")

        # Start capturing video and audio data
        speech_rate, pitch_variability = collect_audio_data()

        if ctx.video_transformer:
            fixation_duration, saccadic_amplitude, saccadic_velocity = ctx.video_transformer.get_eye_tracking_data()

            if speech_rate is not None and pitch_variability is not None:
                # Combine all features
                X_new = np.array([[fixation_duration, saccadic_amplitude, saccadic_velocity, speech_rate, pitch_variability]])

                # Load the saved model and scaler
                model = joblib.load('model_files/random_forest_model.pkl')
                scaler = joblib.load('model_files/scaler.pkl')

                # Scale the input data
                X_new_scaled = scaler.transform(X_new)

                # Predict ADHD likelihood
                prediction = model.predict(X_new_scaled)
                st.write(f"ADHD Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")

    st.write("""
    You can also train the model by uploading synthetic datasets. The datasets should include eye-tracking data and speech data.
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
