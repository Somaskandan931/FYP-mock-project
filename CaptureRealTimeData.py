import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import speech_recognition as sr
import streamlit as st

# Load the pre-trained model and scaler
model = joblib.load("C:/Users/somas/PycharmProjects/FYP_mock_project/model_files/random_forest_model.pkl")
scaler = joblib.load("C:/Users/somas/PycharmProjects/FYP_mock_project/model_files/scaler.pkl")

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize SpeechRecognition
recognizer = sr.Recognizer()

def capture_eye_tracking_data(frame):
    # Process the frame for face detection
    results = mp_face_detection.process(frame)

    if results.detections:
        # Mock processing to extract features
        fixation_duration = np.random.uniform(200, 400)
        saccadic_amplitude = np.random.uniform(1, 5)
        saccadic_velocity = np.random.uniform(100, 400)
    else:
        fixation_duration, saccadic_amplitude, saccadic_velocity = None, None, None

    return fixation_duration, saccadic_amplitude, saccadic_velocity

def capture_speech_data():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            speech_text = recognizer.recognize_google(audio)
            speech_rate = len(speech_text.split()) / 60  # Words per minute
            pitch_variability = np.random.uniform(2, 4)  # Mock pitch variability
        except sr.UnknownValueError:
            speech_rate, pitch_variability = None, None

    return speech_rate, pitch_variability

def main():
    st.title("Real-Time ADHD Likelihood Prediction While Reading")

    st.write("""
    ### Read the passage below. Real-time data will be captured to predict the likelihood of ADHD.
    """)

    passage = """Once upon a time in a faraway land, there lived a wise old owl. The owl was known throughout the forest for its wisdom and kindness. It spent its days watching over the animals and offering advice to those in need. One day, a young fox approached the owl, seeking guidance on how to find its way home. The owl, with a gentle hoot, pointed the fox in the right direction, and the young fox trotted off happily. The owl watched as the fox disappeared into the woods, knowing that it had helped another creature find its path. 

    Several months passed, and the seasons began to change. As autumn arrived, the leaves turned golden and fell gently to the ground. The owl, now a bit older, still sat perched on its favorite branch, watching over the forest. One crisp morning, a lost rabbit came hopping along, tears in its eyes. The owl listened patiently as the rabbit explained how it had wandered too far from its burrow. With a knowing nod, the owl gave the rabbit some comforting words and pointed it toward the familiar trails leading back to its home.

    Winter came soon after, bringing snow and cold winds to the forest. The wise owl, prepared for the harsh season, wrapped itself in its warm feathers. But even in the coldest of days, it continued to look after the forest dwellers. It guided the birds to safe nests and showed the deer where to find the last bits of food. Each act of kindness made the owl's heart feel fuller, despite the icy weather.

    The months turned again, and spring brought new life to the forest. The trees blossomed, and flowers bloomed across the meadow. The owl, feeling rejuvenated, was visited by many animals that it had helped throughout the year. They came with gifts of gratitude and stories of how the owl’s wisdom had changed their lives. The owl, with a humble smile, listened to each story, grateful for the opportunity to have made a difference.

    As the sun set on that beautiful spring day, the owl closed its eyes, feeling a deep sense of peace and contentment. It had spent its life in service to others, and now, surrounded by friends and the beauty of the forest, it felt truly at home."""
    st.text_area("Passage", value=passage, height=200, max_chars=None)

    # Initialize session state variables
    if "capturing" not in st.session_state:
        st.session_state.capturing = False
    if "latest_data" not in st.session_state:
        st.session_state.latest_data = None

    # Define start and stop capture functions
    def start_capture():
        st.session_state.capturing = True

    def stop_capture():
        st.session_state.capturing = False

    # Buttons for controlling data capture
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Capture"):
            start_capture()
    with col2:
        if st.button("Stop Capture"):
            stop_capture()

    # Webcam setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access webcam. Please check your device permissions.")
        return

    # Capturing data in real-time
    if st.session_state.capturing:
        st.write("Capturing data... Please continue reading the passage above.")

        while st.session_state.capturing:
            ret, frame = cap.read()
            if not ret:
                st.error("Error capturing frame from webcam.")
                break

            fixation_duration, saccadic_amplitude, saccadic_velocity = capture_eye_tracking_data(frame)
            speech_rate, pitch_variability = capture_speech_data()

            if None not in (fixation_duration, saccadic_amplitude, saccadic_velocity, speech_rate, pitch_variability):
                st.session_state.latest_data = {
                    'Fixation_Duration': fixation_duration,
                    'Saccadic_Amplitude': saccadic_amplitude,
                    'Saccadic_Velocity': saccadic_velocity,
                    'Speech_Rate': speech_rate,
                    'Pitch_Variability': pitch_variability
                }

                # Display the real-time data to the user
                st.write("### Real-Time Data")
                st.write(f"**Fixation Duration:** {fixation_duration:.2f} ms")
                st.write(f"**Saccadic Amplitude:** {saccadic_amplitude:.2f} degrees")
                st.write(f"**Saccadic Velocity:** {saccadic_velocity:.2f} degrees/second")
                st.write(f"**Speech Rate:** {speech_rate:.2f} words/min")
                st.write(f"**Pitch Variability:** {pitch_variability:.2f} Hz")

                # Prepare input data for prediction
                input_data = pd.DataFrame([st.session_state.latest_data])

                # Normalize the input data using the pre-fitted scaler
                scaled_input_data = scaler.transform(input_data)

                # Predict the likelihood of ADHD
                prediction_probability = model.predict_proba(scaled_input_data)[0, 1]
                threshold = 0.6  # Adjust based on model performance and validation
                prediction = 1 if prediction_probability > threshold else 0

                # Display the prediction and probability
                st.write(f"**Prediction: {'ADHD Likely' if prediction == 1 else 'ADHD Unlikely'}**")
                st.write(f"**Probability of ADHD: {prediction_probability:.2f}**")

            # Delay to simulate real-time processing
            time.sleep(1)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
