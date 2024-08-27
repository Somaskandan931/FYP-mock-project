import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import speech_recognition as sr
import numpy as np


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.fixation_duration = np.random.rand()
        self.saccadic_amplitude = np.random.rand()
        self.saccadic_velocity = np.random.rand()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Returning the video frame without modification
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_eye_tracking_data(self):
        # Simulate eye-tracking data (replace with actual logic if available)
        return np.random.rand(), np.random.rand(), np.random.rand()


def collect_audio_data():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5)

    try:
        speech_rate = np.random.rand()
        pitch_variability = np.random.rand()
        return speech_rate, pitch_variability
    except Exception as e:
        st.error(f"Could not process audio: {e}")
        return None, None


def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


def save_model(model, scaler):
    save_dir = 'model_files'
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))


def main():
    st.title("ADHD Prediction Model")

    st.write("Read the following passage aloud while the application captures data using your webcam and microphone.")
    passage = """
    Once upon a time in a faraway land, there lived a wise old owl...
    """
    st.text_area("Reading Passage", passage, height=150)

    # Initialize webcam stream
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # Initialize session states for capture and data collection
    if 'capture' not in st.session_state:
        st.session_state['capture'] = False
        st.session_state['data'] = []
        st.session_state['labels'] = []

    capture_button = st.button("Start Capture")
    stop_button = st.button("Stop Capture")

    if capture_button:
        st.session_state['capture'] = True
        st.info("Capturing data... Please continue reading.")

    if stop_button:
        st.session_state['capture'] = False
        st.info("Capture stopped.")

        # Convert collected data to DataFrame
        if st.session_state['data']:
            st.write("Training model with captured data...")
            data = np.array(st.session_state['data'])
            labels = np.array(st.session_state['labels'])
            model, scaler = train_model(data, labels)
            save_model(model, scaler)
            st.success("Model trained and saved successfully!")
        else:
            st.warning("No data collected to train the model.")

    # Real-time data feedback and collection
    if st.session_state['capture']:
        if ctx.video_processor:
            fixation_duration, saccadic_amplitude, saccadic_velocity = ctx.video_processor.get_eye_tracking_data()
            speech_rate, pitch_variability = collect_audio_data()

            st.write(f"Fixation Duration: {fixation_duration:.2f}")
            st.write(f"Saccadic Amplitude: {saccadic_amplitude:.2f}")
            st.write(f"Saccadic Velocity: {saccadic_velocity:.2f}")
            st.write(f"Speech Rate: {speech_rate:.2f}")
            st.write(f"Pitch Variability: {pitch_variability:.2f}")

            # Collect data for training
            if speech_rate is not None and pitch_variability is not None:
                st.session_state['data'].append(
                    [fixation_duration, saccadic_amplitude, saccadic_velocity, speech_rate, pitch_variability])
                st.session_state['labels'].append(
                    1 if np.random.rand() > 0.5 else 0)  # Random label, replace with actual logic
        else:
            st.error("Failed to initialize webcam stream.")
    else:
        st.write("Click 'Start Capture' to begin data capture.")


if __name__ == "__main__":
    main()
