import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of participants
num_participants: int = 1000

# Generate Participant IDs
participant_ids: np.ndarray = np.arange(1, num_participants + 1)

# Generate synthetic data for Speech Rate (words per minute)
# Assuming non-ADHD has a slightly higher speech rate than ADHD
speech_rate_non_adhd: np.ndarray = np.random.normal(140, 15, int(num_participants * 0.7))
speech_rate_adhd: np.ndarray = np.random.normal(120, 15, int(num_participants * 0.3))

# Generate synthetic data for Pitch Variability (standard deviation of pitch in Hz)
# Assuming ADHD has higher pitch variability
pitch_variability_non_adhd: np.ndarray = np.random.normal(20, 5, int(num_participants * 0.7))
pitch_variability_adhd: np.ndarray = np.random.normal(35, 10, int(num_participants * 0.3))

# Combine the data
speech_rate: np.ndarray = np.concatenate([speech_rate_non_adhd, speech_rate_adhd])
pitch_variability: np.ndarray = np.concatenate([pitch_variability_non_adhd, pitch_variability_adhd])
labels: np.ndarray = np.concatenate([np.zeros(int(num_participants * 0.7)), np.ones(int(num_participants * 0.3))])

# Shuffle the data
indices: np.ndarray = np.random.permutation(num_participants)
speech_rate = speech_rate[indices]
pitch_variability = pitch_variability[indices]
labels = labels[indices]

# Create DataFrame
data: pd.DataFrame = pd.DataFrame({
    'Participant_ID': participant_ids,
    'Speech_Rate': speech_rate,
    'Pitch_Variability': pitch_variability,
    'Label': labels
})

# Save to CSV
data.to_csv('synthetic_speech_dataset.csv', index=False)

print("Synthetic dataset created successfully.")
