# MTC-AIC3 BCI Competition Dataset

## Dataset Overview
- **EEG Data**: Recordings from 8 channels.
- **Sampling Rate**: 250 Hz.
- **Participants**: 40 male subjects, average age 20 years.
- **Tasks**: Two distinct Brain-Computer Interface (BCI) paradigms:
    - **Motor Imagery (MI)**: Imagining left or right hand movement.
    - **Steady-State Visual Evoked Potential (SSVEP)**: Focusing on visual stimuli.
- **Classes**:
    - MI: 2 classes (Left, Right)
    - SSVEP: 4 classes (Left, Right, Forward, Backward)
- **Trial Duration**:
    - MI: 9 seconds per trial.
    - SSVEP: 7 seconds per trial.
- **Trials per Session**: 10 trials for each experimental session.

## Directory Structure
The dataset is organized into two main task directories (`MI/` and `SSVEP/`) within the `mtc-aic3_dataset` folder. Each task directory contains three subdirectories for data splitting:
- `train/`: Data for model training (30 subjects, 8 trial sessions per subject, 4800 total trials)
- `validation/`: Data for model validation (5 subjects, 1 trial session per subject, 100 total trials)
- `test/`: Data for model testing (5 subjects, 1 trial session per subject, 100 total trials)

Each subject's directory (e.g., `S1/`, `S2/`) contains session directories (e.g., `1/`), representing experimental sessions.

```
mtc-aic3_dataset/
├── MI/
│   ├── train/
│   │   ├── S1/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S2/
│   │   │   └── ...
│   │   └── ...
│   ├── validation/
│   │   ├── S31/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S32/
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── S36/
│       │   └── 1/
│       │       └── EEGdata.csv
│       ├── S37/
│       │   └── ...
│       └── ...
├── SSVEP/
│   ├── train/
│   │   ├── S1/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S2/
│   │   │   └── ...
│   │   └── ...
│   ├── validation/
│   │   ├── S31/
│   │   │   └── 1/
│   │   │       └── EEGdata.csv
│   │   ├── S32/
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── S36/
│       │   └── 1/
│       │       └── EEGdata.csv
│       ├── S37/
│       │   └── ...
│       └── ...
├── train.csv
├── validation.csv
├── test.csv
└── sample_submission.csv
```

## Data Files within Session Directories
- **`EEGdata.csv`**: Contains the raw EEG recordings for all 10 trials in a session, concatenated sequentially.
    - **Columns**: `Time`, 8 EEG channels (`FZ`, `C3`, `CZ`, `C4`, `PZ`, `PO7`, `OZ`, `PO8`), motion sensors (`AccX`, `AccY`, `AccZ`, `Gyro1`, `Gyro2`, `Gyro3`), `Battery` level, `Counter`, and `Validation` flag.
    - **Trial Samples**: Each MI trial has 2250 samples (9s * 250Hz). Each SSVEP trial has 1750 samples (7s * 250Hz).

## CSV Files for Kaggle Competition
These files are located in the root of the `mtc-aic3_dataset` directory and provide a structured way to access the data for the competition.

- **`train.csv`**: Training data with corresponding labels (4800 entries).
    - **Columns**: `id` (unique row ID), `subject_id`, `task` (MI or SSVEP), `trial_session` (session number), `trial` (trial number within session, 1-10), `label` (class label).
    - IDs range from 1 to 4800.

- **`validation.csv`**: Validation data with corresponding labels (100 entries).
    - **Columns**: `id`, `subject_id`, `task`, `trial_session`, `trial`, `label`.
    - IDs range from 4801 to 4900.

- **`test.csv`**: Test data **without** labels (100 entries).
    - **Columns**: `id`, `subject_id`, `task`, `trial_session`, `trial`.
    - IDs range from 4901 to 5000.
    - Participants will use this file to generate predictions.

- **`sample_submission.csv`**: A template file showing the correct format for submitting predictions (100 entries).
    - **Columns**: `id`, `label`.
    - `id` values correspond to those in `test.csv`.
    - Participants should replace the placeholder "Left" labels with their model's predictions.

## How to Use the Data for the Competition

1.  **Understanding the Data Structure**:
    *   The primary CSV files (`train.csv`, `validation.csv`, `test.csv`) act as an index to the raw EEG data.
    *   Each row in these CSVs points to a specific trial within a specific session, for a specific subject and task.

2.  **Accessing Raw EEG Data for a Trial**:
    *   Take a row from `train.csv`, `validation.csv`, or `test.csv`.
    *   Use the `subject_id`, `task`, `trial_session` to navigate to the correct session directory (e.g., `mtc-aic3_dataset/MI/train/S1/1/`).
    *   Load the `EEGdata.csv` file from that session directory.
    *   The `trial` column in the main CSV indicates which of the 10 trials in `EEGdata.csv` to use.
    *   To extract the data for trial `n`:
        *   For **MI**: Samples from `(n-1) * 2250` to `(n * 2250) - 1`.
        *   For **SSVEP**: Samples from `(n-1) * 1750` to `(n * 1750) - 1`.

3.  **Training Your Model**:
    *   Iterate through `train.csv`.
    *   For each entry, fetch the corresponding raw EEG segment as described above.
    *   Use the `label` column from `train.csv` as the ground truth for training your classification model.

4.  **Validating Your Model**:
    *   Iterate through `validation.csv`.
    *   Fetch the EEG data and use the `label` column to evaluate your model's performance on unseen data.

5.  **Generating Predictions for Submission**:
    *   Iterate through `test.csv`.
    *   For each entry, fetch the corresponding raw EEG segment.
    *   Use your trained model to predict the class label.
    *   Create a submission file in the same format as `sample_submission.csv`, with the `id` from `test.csv` and your predicted `label`.

## Example Code Snippet (Python)
```python
import pandas as pd
import numpy as np
import os

# Load index files
base_path = '.'  # Replace with the path to the dataset directory if needed
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

# Function to load a trial's EEG data
def load_trial_data(row, base_path='.'):
    # Determine dataset type based on ID range
    id_num = row['id']
    if id_num <= 4800:
        dataset = 'train'
    elif id_num <= 4900:
        dataset = 'validation'
    else:
        dataset = 'test'
    
    # Construct the path to EEGdata.csv
    eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
    
    # Load the entire EEG file
    eeg_data = pd.read_csv(eeg_path)
    
    # Calculate indices for the specific trial
    trial_num = int(row['trial'])
    if row['task'] == 'MI':
        samples_per_trial = 2250  # 9 seconds * 250 Hz
    else:  # SSVEP
        samples_per_trial = 1750  # 7 seconds * 250 Hz
    
    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial - 1
    
    # Extract the trial data
    trial_data = eeg_data.iloc[start_idx:end_idx+1]
    return trial_data

# Example: Load data for the first training example
first_trial = validation_df.iloc[0]  # Using index 0 for the first row
trial_data = load_trial_data(first_trial, base_path=base_path)  # Pass the base_path parameter
print(f"Loaded task: {first_trial['task']}, Trial: {first_trial['trial']} for subject {first_trial['subject_id']}")
# Print label if available (only for training and validation data)
if 'label' in first_trial:
    print(f"Label: {first_trial['label']}")
print(f"Data shape: {trial_data.shape}")
print(f"First 5 rows of trial data:\n{trial_data.head()}")
```

## Competition Task
Your goal is to develop a robust classification model that can accurately predict the class labels (`Left`/`Right` for MI; `Left`/`Right`/`Forward`/`Backward` for SSVEP) for the trials listed in `test.csv`.

Good luck!
