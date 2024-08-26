import streamlit as st
import pandas as pd
import numpy as np
import time

# Mock functions to simulate real-time data capture
def capture_eye_tracking_data():
    fixation_duration = np.random.uniform(200, 400)  # milliseconds (Typical fixation duration range)
    saccadic_amplitude = np.random.uniform(1, 5)  # degrees (Typical saccadic amplitude range)
    saccadic_velocity = np.random.uniform(100, 400)  # degrees/second (Typical saccadic velocity range)
    return fixation_duration, saccadic_amplitude, saccadic_velocity

def capture_speech_data():
    speech_rate = np.random.uniform(110, 160)  # words per minute (Typical speech rate for adults)
    pitch_variability = np.random.uniform(2, 4)  # Hz (Typical pitch variability range)
    return speech_rate, pitch_variability

def complex_predictor(data):
    # Define weights for each feature
    weights = {
        'Fixation_Duration': 0.4,  # Weight for fixation duration
        'Saccadic_Amplitude': 0.3,  # Weight for saccadic amplitude
        'Speech_Rate': 0.3,  # Weight for speech rate
    }
    
    # Calculate weighted scores for each feature
    score_fixation = max(0, min(1, (data['Fixation_Duration'] - 200) / 400))
    score_amplitude = max(0, min(1, (5 - data['Saccadic_Amplitude']) / 5))
    score_speech_rate = max(0, min(1, (160 - data['Speech_Rate']) / 160))
    
    # Compute the weighted average score
    weighted_score = (
        weights['Fixation_Duration'] * score_fixation +
        weights['Saccadic_Amplitude'] * score_amplitude +
        weights['Speech_Rate'] * score_speech_rate
    )
    
    # Determine ADHD likelihood based on the weighted score
    probability = weighted_score
    prediction = 1 if probability > 0.5 else 0

    return prediction, probability

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
    if "data" not in st.session_state:
        st.session_state.data = []

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

    # Capturing data in real-time simulation
    if st.session_state.capturing:
        st.write("Capturing data... Please continue reading the passage above.")

        fixation_duration, saccadic_amplitude, saccadic_velocity = capture_eye_tracking_data()
        speech_rate, pitch_variability = capture_speech_data()

        # Store the captured data in session state
        st.session_state.data.append({
            'Fixation_Duration': fixation_duration,
            'Saccadic_Amplitude': saccadic_amplitude,
            'Saccadic_Velocity': saccadic_velocity,
            'Speech_Rate': speech_rate,
            'Pitch_Variability': pitch_variability
        })

        # Display the captured data
        st.write(f"Fixation Duration: {fixation_duration:.2f} ms")
        st.write(f"Saccadic Amplitude: {saccadic_amplitude:.2f} degrees")
        st.write(f"Saccadic Velocity: {saccadic_velocity:.2f} degrees/second")
        st.write(f"Speech Rate: {speech_rate:.2f} words/min")
        st.write(f"Pitch Variability: {pitch_variability:.2f} Hz")

        # Add a small delay to simulate real-time data capture
        time.sleep(1)

    # Display all captured data and make predictions when capturing stops
    if not st.session_state.capturing and st.session_state.data:
        st.write("Data capture stopped. Displaying all captured data and predictions:")

        for i, data in enumerate(st.session_state.data):
            st.write(f"Data Point {i+1}:")
            st.write(f"Fixation Duration: {data['Fixation_Duration']:.2f} ms")
            st.write(f"Saccadic Amplitude: {data['Saccadic_Amplitude']:.2f} degrees")
            st.write(f"Saccadic Velocity: {data['Saccadic_Velocity']:.2f} degrees/second")
            st.write(f"Speech Rate: {data['Speech_Rate']:.2f} words/min")
            st.write(f"Pitch Variability: {data['Pitch_Variability']:.2f} Hz")

            # Use the captured data to make a prediction
            prediction, probability = complex_predictor(data)
            st.write(f"ADHD Likelihood: {'High' if prediction == 1 else 'Low'} ({probability:.2f})")

if __name__ == "__main__":
    main()
