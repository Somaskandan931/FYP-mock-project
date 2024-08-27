import os
import time
import joblib
import numpy as np
import speech_recognition as sr
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load pre-trained model and scaler
model_path = 'C:/Users/somas/PycharmProjects/FYP_mock_project/model_files/random_forest_model.pkl'
scaler_path = 'C:/Users/somas/PycharmProjects/FYP_mock_project/model_files/scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.fixation_duration = np.random.rand()
        self.saccadic_amplitude = np.random.rand()
        self.saccadic_velocity = np.random.rand()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_eye_tracking_data(self):
        # Simulate eye-tracking data
        return np.random.rand(), np.random.rand(), np.random.rand()

def collect_audio_data():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5)

    try:
        # Simulate speech data extraction
        speech_rate = np.random.rand()
        pitch_variability = np.random.rand()
        return speech_rate, pitch_variability
    except Exception as e:
        st.error(f"Could not process audio: {e}")
        return None, None

def main():
    st.title("ADHD Prediction Model")

    st.write("Read the following passage aloud while the application captures data using your webcam and microphone for 5 minutes.")
    passage = """
    Once upon a time in a faraway land, there lived a wise old owl. The owl was known throughout the forest for its wisdom and kindness. It spent its days watching over the animals and offering advice to those in need. One day, a young fox approached the owl, seeking guidance on how to find its way home. The owl, with a gentle hoot, pointed the fox in the right direction, and the young fox trotted off happily. The owl watched as the fox disappeared into the woods, knowing that it had helped another creature find its path. 

    Several months passed, and the seasons began to change. As autumn arrived, the leaves turned golden and fell gently to the ground. The owl, now a bit older, still sat perched on its favorite branch, watching over the forest. One crisp morning, a lost rabbit came hopping along, tears in its eyes. The owl listened patiently as the rabbit explained how it had wandered too far from its burrow. With a knowing nod, the owl gave the rabbit some comforting words and pointed it toward the familiar trails leading back to its home.

    Winter came soon after, bringing snow and cold winds to the forest. The wise owl, prepared for the harsh season, wrapped itself in its warm feathers. But even in the coldest of days, it continued to look after the forest dwellers. It guided the birds to safe nests and showed the deer where to find the last bits of food. Each act of kindness made the owl's heart feel fuller, despite the icy weather.

    The months turned again, and spring brought new life to the forest. The trees blossomed, and flowers bloomed across the meadow. The owl, feeling rejuvenated, was visited by many animals that it had helped throughout the year. They came with gifts of gratitude and stories of how the owl’s wisdom had changed their lives. The owl, with a humble smile, listened to each story, grateful for the opportunity to have made a difference.

    As the sun set on that beautiful spring day, the owl closed its eyes, feeling a deep sense of peace and contentment. It had spent its life in service to others, and now, surrounded by friends and the beauty of the forest, it felt truly at home.
    """
    st.text_area("Reading Passage", passage, height=150)

    ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformerBase)

    start_button = st.button("Start Reading")
    stop_button = st.button("Stop Reading")

    if 'capture' not in st.session_state:
        st.session_state['capture'] = False
        st.session_state['data'] = []

    if start_button:
        st.session_state['capture'] = True
        st.session_state['start_time'] = time.time()
        st.info("Data capture started. Please read the passage for 5 minutes.")

    if stop_button:
        st.session_state['capture'] = False
        st.info("Data capture stopped.")
        if st.session_state['data']:
            # Convert list to numpy array for model prediction
            data = np.array(st.session_state['data'])
            data_scaled = scaler.transform(data)
            predictions = model.predict(data_scaled)

            # Display results
            for i, prediction in enumerate(predictions):
                result = 'Positive' if prediction == 1 else 'Negative'
                st.write(f"Prediction for data point {i+1}: {result}")
        else:
            st.warning("No data captured.")

    if st.session_state['capture']:
        if ctx.video_processor:
            current_time = time.time()
            elapsed_time = current_time - st.session_state['start_time']

            if elapsed_time < 300:  # 5 minutes = 300 seconds
                fixation_duration, saccadic_amplitude, saccadic_velocity = ctx.video_processor.get_eye_tracking_data()
                speech_rate, pitch_variability = collect_audio_data()

                st.write(f"Fixation Duration: {fixation_duration:.2f}")
                st.write(f"Saccadic Amplitude: {saccadic_amplitude:.2f}")
                st.write(f"Saccadic Velocity: {saccadic_velocity:.2f}")
                st.write(f"Speech Rate: {speech_rate:.2f}")
                st.write(f"Pitch Variability: {pitch_variability:.2f}")

                if speech_rate is not None and pitch_variability is not None:
                    st.session_state['data'].append(
                        [fixation_duration, saccadic_amplitude, saccadic_velocity, speech_rate, pitch_variability]
                    )
        else:
            st.error("Failed to initialize webcam stream.")
    else:
        st.write("Click 'Start Reading' to begin data capture.")

if __name__ == "__main__":
    main()
