from numpy import random, concatenate, arange, ones, zeros
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Number of participants
num_participants = 1000

# Generate Participant IDs
participant_ids = arange(1, num_participants + 1)

# Generate synthetic data
fixation_duration_non_adhd = random.normal(300, 50, int(num_participants * 0.7))
fixation_duration_adhd = random.normal(350, 50, int(num_participants * 0.3))

saccadic_amplitude_non_adhd = random.normal(5, 1, int(num_participants * 0.7))
saccadic_amplitude_adhd = random.normal(6, 1, int(num_participants * 0.3))

saccadic_velocity_non_adhd = random.normal(400, 50, int(num_participants * 0.7))
saccadic_velocity_adhd = random.normal(350, 50, int(num_participants * 0.3))

# Combine the data
fixation_duration = concatenate([fixation_duration_non_adhd, fixation_duration_adhd])
saccadic_amplitude = concatenate([saccadic_amplitude_non_adhd, saccadic_amplitude_adhd])
saccadic_velocity = concatenate([saccadic_velocity_non_adhd, saccadic_velocity_adhd])
labels = concatenate([zeros(int(num_participants * 0.7)), ones(int(num_participants * 0.3))])

# Shuffle the data
indices = random.permutation(num_participants)
fixation_duration = fixation_duration[indices]
saccadic_amplitude = saccadic_amplitude[indices]
saccadic_velocity = saccadic_velocity[indices]
labels = labels[indices]

# Create DataFrame
data = pd.DataFrame({
    'Participant_ID': participant_ids,
    'Fixation_Duration': fixation_duration,
    'Saccadic_Amplitude': saccadic_amplitude,
    'Saccadic_Velocity': saccadic_velocity,
    'Label': labels
})

# Save to CSV
data.to_csv('synthetic_adhd_dataset.csv', index=False)

print("Synthetic dataset created successfully.")
