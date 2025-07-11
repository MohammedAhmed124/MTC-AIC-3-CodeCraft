import sys , os

# -----------------------------------------------------------------------------
# Project Path Setup
#    - Add project root to Python path for consistent imports
# -----------------------------------------------------------------------------



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -----------------------------------------------------------------------------
#  Core Imports
#   - Model.MTCFormer
#   - data extractors: extract_data,extract_subject_labels
#   - training utils: predict
#   - augmentation: augment_data
#   - datasets: The custom EEGDataset
# -----------------------------------------------------------------------------



from model.MTCformerV3 import MTCFormer
import torch
import glob
import numpy as np
import pandas as pd
from torch.optim import Adam
from utils.extractors import extract_data,extract_subject_labels
from utils.preprocessing import preprocess_data,preprocess_one_file
from utils.augmentation import augment_data
from utils.training import  predict
from utils.CustomDataset import EEGDataset
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Hey! I'm Mohammed Ahmed Metwally (or just Mohammed A.Metwally 😊), The Team Leader of CodeCraft.

# Super excited to walk you through this script — it's the full inference pipeline
# we built for the MTC-AIC EEG competition. This is where everything comes together:
# loading test data, running it through our trained MTCFormer models, and generating
# predictions for both MI and SSVEP tasks.

# I’ve kept most of the theory and reasoning out of the code to keep it clean —
# you can find all the details and explanations in our system description paper.

# Alright, let’s get into it!
# -----------------------------------------------------------------------------

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))
DATA_FIF_DIR = os.path.join(ROOT_PATH,"data_fif")



# -----------------------------------------------------------------------------
#  Script Argument: --predict_on_best_models
#
# This script accepts a command-line argument called `--predict_on_best_models`.
# It's a flag that controls **which checkpoints** we use for inference.

# ✅✅✅✅✅ By default, it's set to True — meaning:
# If you just run the script normally (without setting anything),
# it will automatically load the best-performing checkpoints that were submitted
# to the competition. These are assumed to already be stored in the root-level
# `/checkpoints/` directory, and the final predictions will be saved as:
# → `submission.csv`

# 🔄🔄🔄🔄🔄 However, if you **explicitly** pass `--predict_on_best_models False`, then:
# The script will instead use the most **recently trained models** produced by
# our training scripts (like `train_mi_model1.py`, `train_ssvep_model.py`, etc.).
# These fresh checkpoints are expected to live under:
# → `/train/checkpoints/`
# And the final predictions will be saved separately as:
# → `submission_regenerated_(non_best).csv`

# This allows easy switching between official and experimental runs.
# -----------------------------------------------------------------------------



import argparse
parser = argparse.ArgumentParser(description="Run inference on EEG data")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "1"):
        return True
    elif v.lower() in ("false", "f", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# Accept --predict_on_best_models as a boolean flag
parser.add_argument(
    "--predict_on_best_models",
    type=str2bool,
    default=True,
    help="Whether to run prediction using best models (True/False)"
)

predict_on_best_models = parser.parse_args().predict_on_best_models

if predict_on_best_models:
    print("Inference will be run on best checkpoint...")
    checkpoint_dicetory = ROOT_PATH
    SUBMISSION_PATH = os.path.join(ROOT_PATH,"submission.csv")
else:
    print("Inference will be run on recently generated checkpoints (not the best ones)...")
    checkpoint_dicetory = os.path.join(ROOT_PATH,"train")
    SUBMISSION_PATH = os.path.join(ROOT_PATH,"submission_regenerated_(non_best).csv")

print(
    checkpoint_dicetory , SUBMISSION_PATH
)




# -----------------------------------------------------------------------------
#  Inference for Motor Imagery (MI) - Ensemble of 2 MTCFormer Models
#
# In this section, we will perform inference on the MI task using 2 different
# MTCFormer models. Each of these models was trained with different settings:
#
# Instead of relying on just one model’s predictions, I combine all 2 using
# a technique called **Rank Averaging**. The idea here is simple but powerful:
#   → Each model gives me probabilities over the classes (shape: N, C)
#   → I rank those probabilities for each model (higher means more confident)
#   → Then I average the ranks across the models, and normalize the result
#
# Why rank averaging?
#   - It helps reduce the impact of any single model being too confident or skewed
#   - It focuses on which class each model thinks is strongest — not the raw values
#   - The base models that were trained had very poor callibration.
#
# In the end, I get a single set of final probabilities (1, C) that’s more stable
# and (hopefully!) more accurate than any single model on its own.
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
#  Step 1: Load MI Test Data for Inference
#
# Before we can preprocess or run any predictions, we first need to load the
# raw `.fif` EEG files for the Motor Imagery (MI) test set.
#
# Here's what's happening:
# - We defined paths to our test data directory (earlier via ROOT-DIR and SCRIPT-DIR)
# - We gather all `.fif` files related to MI test data using glob
# - We load the raw MNE data using our `extract_data()` helper
# - We also extract subject labels, which might be useful 
#   domain-adaption later.
#
# Additionally, we define:
# - A mapping from task descriptions ("Left", "Right") to class IDs (0, 1)
# - And the inverse mapping, in case we need to decode predictions later
# -----------------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_file_paths_mi = glob.glob(os.path.join(DATA_FIF_DIR, "test/MI/*.fif"))


mapping_mi = {
    "Left":0,
    "Right":1
}   
inv_mapping_mi = {
    v:k for k,v in mapping_mi.items()
}
test_data_mi , test_labels_mi , ids_mi = extract_data(test_file_paths_mi , return_id = True)
test_subject_labels = extract_subject_labels(test_data_mi)


# -----------------------------------------------------------------------------
#  Model 1 – Inference for MI using MTCFormer (depth=2, window=600)
#
# This is the first MTCFormer model we’ll use in the ensemble.
# Here's a breakdown of what's happening:
#
#  Model Architecture:
# - We load a model with:
#     • depth = 2 → two convolutional-attention blocks
#     • kernel_size = 5 → small temporal kernels for local pattern detection
#     • input window = 600 time steps
#     • moderate dropout settings for regularization
# 
#
#  Checkpoint Loading:
# - We load the best checkpoint (depends on --predict_on_best_models script argument) from a previous training run
# - `strict=False` allows for safe loading even if buffers or extra keys differ
# - The loaded checkpoint is the best we could score on both validation test set and the leaderboard
#
#  Preprocessing Pipeline:
# - EEG channels used: ['C3', 'C4', 'CZ', 'FZ'] + motion sensors + Validation marker
# - Preprocessing includes:
#     1. Notch filtering at 50 & 100 Hz
#     2. Bandpass filtering between 6–30 Hz (Motor Imagery relevant band)
#     3. Channel selection
#     4. Windowing into 600-sample windows (stride = 35 for dense overlap)
#     5. Normalization (ignoring Validation column to preserve its binary info)
#
#  Data Wrapping:
# - We convert the processed numpy arrays to PyTorch tensors
# - Use `EEGDataset` to organize inputs, weights, dummy labels, and subject info
# - All windows are fed into a DataLoader (no shuffle, full batch inference)
#
#  Inference:
# - We run the model on the preprocessed test data
# - We set `probability=True` to get softmax outputs (class probabilities)
# - The final output is stored in `preds_mi_one` for later ensembling
# -----------------------------------------------------------------------------


model_mi_1 = MTCFormer(depth=2,
                    kernel_size=5,
                    n_times=600,
                    chs_num=7,
                    eeg_ch_nums=4,
                    class_num=2,
                    class_num_domain=30,
                    modulator_dropout=0.3,
                    mid_dropout=0.5,
                    output_dropout=0.5,
                    weight_init_mean=0,
                    weight_init_std=0.5,
                    ).to(device)



optimizer = Adam(model_mi_1.parameters(), lr=0.002)

checkpoint_path = os.path.join(
    checkpoint_dicetory,
    "checkpoints",
    "model_1_mi_checkpoint",
    "best_model_.pth"
    )

checkpoint = torch.load(checkpoint_path, weights_only=False)

model_mi_1.load_state_dict(checkpoint['model_state_dict'] , strict=False)

print("Preprocessing data for model 1 ..... ")
cols_to_pick = [
        'C3',
        'C4',
        'CZ',
        'FZ',
        'Acc_norm',
        'gyro_norm',
        'Validation'
          ]

params = {
    "cols_to_pick":cols_to_pick,
    "l_freq": 6,
    "h_freq": 30,
    "notch_freqs": [50, 100],
    "notch_width": 1.0,
    "window_size": 600,
    "window_stride": 35
}

test_data,weights_test, _ ,subject_label_test_, WINDOW_LEN= preprocess_data(
    test_data_mi,
    labels = test_labels_mi,
    subject_labels = test_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
    )


test_mi_torch = torch.from_numpy(test_data).to(torch.float32)
weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_labels_placeholder = torch.zeros(test_mi_torch.shape[0], dtype=torch.long)
test_mi_torch_subject = torch.from_numpy(subject_label_test_).to(torch.long)


test_dataset = EEGDataset(
    data_tensor = test_mi_torch,
    weigths = weights_test_torch,
    label_tensor = test_labels_placeholder,
    subject_labels = test_mi_torch_subject
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle=False
    ) 


preds_mi_one = predict(
    model_mi_1,
    window_len=WINDOW_LEN,
    loader=test_loader,
    num_samples_to_predict=50,
    device = device,
    logits=True,
    probability=False,
    K=1
    )










# -----------------------------------------------------------------------------
#  Model 2 – Inference for MI using MTCFormer (depth=2, window=600)
#
# This is the second and final model we'll use in our Motor Imagery (MI) ensemble.
# Architecturally, it's similar to Model 1 — same depth and window size — but it
# was trained under different training conditions,
# which brings diversity to the ensemble and helps improve robustness. (does not use adversarial sample generation)
#
#  Model Architecture:
# - depth = 2 → moderate depth, capturing key spatiotemporal EEG patterns
# - kernel_size = 5 → short-term temporal filters
# - input window = 600 time steps
# - Dropout values are consistent with previous models
#
#  Checkpoint Loading:
# - Loads pre-trained weights from its own checkpoint folder
# - `strict=False` ensures flexible loading across different runs
#
#  Preprocessing Pipeline:
# - Channels used: ['C3', 'C4', 'CZ', 'FZ'] + accelerometer + gyroscope + Validation
# - Steps:
#     1. Notch filter at 50 & 100 Hz
#     2. Bandpass filter from 6–30 Hz (targeting the MI frequency band)
#     3. Channel picking
#     4. Windowing into 600-sample frames (stride = 35)
#     5. Normalization (skipping 'Validation' column to preserve semantics)
#
#  Data Wrapping:
# - Tensors are created from the processed data
# - `EEGDataset` prepares the test set with dummy labels
# - A full-batch DataLoader is created for efficient inference
#
#  Inference:
# - Softmax probabilities are computed by passing the test data through the model
# - The result is stored in `preds_mi_three` for use in the final rank-based ensemble
# -----------------------------------------------------------------------------

model_mi_two = MTCFormer(depth=2,
                    kernel_size=5,
                    n_times=600,
                    chs_num=7,
                    eeg_ch_nums=4,
                    class_num=2,
                    class_num_domain=30,
                    modulator_dropout=0.3,
                    mid_dropout=0.5,
                    output_dropout=0.5,
                    weight_init_mean=0,
                    weight_init_std=0.5,
                    ).to(device)



optimizer = Adam(model_mi_two.parameters(), lr=0.002)

checkpoint_path = os.path.join(
    checkpoint_dicetory,
    "checkpoints",
    "model_2_mi_checkpoint",
    "best_model_.pth"
    )

checkpoint = torch.load(checkpoint_path, weights_only=False)

model_mi_two.load_state_dict(checkpoint['model_state_dict'] , strict=False)

print("Preprocessing data for model 3...... ")
cols_to_pick = [
        'C3',
        'C4',
        'CZ',
        'FZ',
        'Acc_norm',
        'gyro_norm',
        'Validation'
          ]

params = {
    "cols_to_pick":cols_to_pick,
    "l_freq": 6,
    "h_freq": 30,
    "notch_freqs": [50, 100],
    "notch_width": 1.0,
    "window_size": 600,
    "window_stride": 35
}

test_data,weights_test, _ ,subject_label_test_, WINDOW_LEN= preprocess_data(
    test_data_mi,
    labels = test_labels_mi,
    subject_labels = test_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
    )


test_mi_torch = torch.from_numpy(test_data).to(torch.float32)
weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_labels_placeholder = torch.zeros(test_mi_torch.shape[0], dtype=torch.long)
test_mi_torch_subject = torch.from_numpy(subject_label_test_).to(torch.long)


test_dataset = EEGDataset(
    data_tensor = test_mi_torch,
    weigths = weights_test_torch,
    label_tensor = test_labels_placeholder,
    subject_labels = test_mi_torch_subject
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle=False
    ) 
preds_mi_two = predict(
    model_mi_two,
    window_len=WINDOW_LEN,
    loader=test_loader,
    num_samples_to_predict=50,
    device = device,
    logits = True,
    probability=False,
    K=0.2
    )



# -----------------------------------------------------------------------------
#  Final Step – Rank Averaging Ensemble for MI Predictions
#
# Now that we’ve obtained predictions from all two MTCFormer models,
# it’s time to combine them into a single, robust prediction using
# **Rank Averaging**, a powerful ensemble strategy.
#
#  How Rank Averaging Works:
# - Each model outputs a probability distribution over the classes (N, C)
# - For each sample:
#     → We **rank** the predicted probabilities from each model (higher = more confident)
#     → We **average** these ranks across models, optionally using weights
#     → The averaged ranks are then normalized into final pseudo-probabilities
#     → The class with the highest average rank is selected as the final prediction
#
#  Why did we choose Rank Averaging?
# - More robust than raw probability averaging, especially when models have different confidence scales
# - Reduces the impact of any single overconfident (or underconfident) model
#
#  Ensemble Setup:
# - We provide a list of model outputs (`probs_list`)
# - We assign relative importance via `weights` (e.g. giving model 2 a bit less weight)
#
#  Final Output:
# - The resulting predictions are stored in `final_mi_predictions`
# - These are then mapped back to human-readable labels (e.g. "Left", "Right")
# - The final `preds_mi_csv` DataFrame contains IDs and predicted labels
#
#  This DataFrame will later be **concatenated with SSVEP predictions**
#     to form the complete submission for the competition.
# -----------------------------------------------------------------------------

from utils.rank_ensemble import RankAveragingEnsemble

# probs_list =  [
#     preds_mi_one,
#     preds_mi_two,
# ]
# weights = [  1  , 1 ]

# final_mi_predictions = RankAveragingEnsemble(
#     prob_list=probs_list,
#     weights=weights
# )

print(preds_mi_one)
print(preds_mi_two)
W1 = 1
W2 = 1
final_mi_predictions = np.argmax(np.sqrt(W1*preds_mi_two*W2*preds_mi_one),axis=1)

final_mi_predictions = np.argmax((preds_mi_two+preds_mi_one)/2,axis=1)

preds_mi_csv = pd.DataFrame({
    "id":ids_mi,
    "label": pd.Series(final_mi_predictions).map(inv_mapping_mi).values
})






# -----------------------------------------------------------------------------
#  SSVEP Inference – Using a Single MTCFormer Model
#
# Unlike the MI section where we built an ensemble, here we chose to keep
# things simple and stick with just one well-performing MTCFormer model.
#
# Why only one model?
# - The SSVEP task was more straightforward, and the single model performed well
# - We didn't want to increase system complexity without clear benefit
# - Time constraints also limited further model exploration or ensembling
#
#  Step 1: Load Test Data
# - We gather all `.fif` files from the SSVEP test directory
# - Raw EEG data is extracted using `extract_data()`
# - Subject-level information is also extracted (used for domain-based reasoning)
# -----------------------------------------------------------------------------

test_file_paths_ssvep = glob.glob(os.path.join(DATA_FIF_DIR, "test/SSVEP/*.fif"))

test_data_ssvep, test_labels_ssvep , ids_ssvep = extract_data(test_file_paths_ssvep , return_id = True)
test_subject_labels = extract_subject_labels(test_data_ssvep)


mapping_ssvep = {
    "Backward":0,
    "Forward":1,
    "Left":2,
    "Right":3
}

inv_mapping_ssvep = {
    v:k for k,v in mapping_ssvep.items()
}



# -----------------------------------------------------------------------------
# Inference steps
#
# After loading the SSVEP test data, we proceed with setting up and running
# inference using a **single MTCFormer model**. Here's what this section does:
#
#  Label Mapping:
# - We define a mapping between the four SSVEP classes and their string labels:
#     → "Forward", "Backward", "Left", and "Right"
# - Also create the inverse mapping so we can decode final predictions later
#
#  Model Architecture:
# - depth = 1 → a lightweight, shallow model (faster, fewer parameters)
# - kernel_size = 10 → wide enough to capture frequency-driven SSVEP responses
# - input window = 500 time steps (tailored to match stimulation frequency cycles)
# - High dropout (0.7) used throughout →  to  avoid overfitting
# - 4 output classes (multi-directional SSVEP)
#
#  Checkpoint Loading:
# - The model is loaded from its dedicated checkpoint (`best_model_.pth`)
# - `strict=False` allows smooth loading even with non-essential buffer differences
#
#  Preprocessing Pipeline:
# - Channels selected: ['OZ', 'PO7', 'PO8', 'PZ'] (posterior electrodes for SSVEP)
#   along with motion sensors and a binary `Validation` marker
# - Steps:
#     1. Notch filter at 50 & 100 Hz
#     2. Bandpass filter between 8–14 Hz (chosen to match stimulation frequency range)
#     3. Channel picking
#     4. Windowing: 500-sample windows with a stride of 50 (capturing multiple cycles)
#     5. Normalization (excluding Validation column from scaling)
#
#  Data Wrapping:
# - Preprocessed numpy arrays are converted to PyTorch tensors
# - As with MI, we wrap the data using `EEGDataset`, passing dummy labels
# - Full-batch DataLoader is used (no shuffling)
#
#  Inference:
# - The model processes the test windows
# - Predictions are obtained using `predict()` with `probability=False`
# - The output `final_preds_ssvep` contains the predicted **class indices**
# -----------------------------------------------------------------------------


model_ssvep = MTCFormer(
    depth=1,
    kernel_size=10,
    n_times=500,
    chs_num=7,
    eeg_ch_nums=4,
    class_num=4,
    class_num_domain=30,
    modulator_kernel_size=10,
    domain_dropout=0.7,
    modulator_dropout=0.7,
    mid_dropout=0.7,
    output_dropout=0.7,
    weight_init_std=0.05,
    weight_init_mean=0.0,
).to(device)

optimizer = Adam(model_ssvep.parameters(), lr=0.002)

checkpoint_path = os.path.join(
    checkpoint_dicetory,
    "checkpoints",
    "model_ssvep_checkpoint",
    "best_model_.pth"
    )

checkpoint = torch.load(checkpoint_path, weights_only=False)

model_ssvep.load_state_dict(checkpoint['model_state_dict'] , strict=False)


cols_to_pick = [
        'OZ',
        'PO7',
        'PO8',
        'PZ',
        'Acc_norm',
        'gyro_norm',
        'Validation'
          ]


params = {
    "cols_to_pick":cols_to_pick,
    "l_freq": 8,
    "h_freq": 14,
    "notch_freqs": [50, 100],
    "notch_width": 1.0,
    "window_size": 500,
    "window_stride": 50
}

test_data,weights_test, _ ,subject_label_test_, WINDOW_LEN= preprocess_data(
    test_data_ssvep,
    labels = test_labels_ssvep,
    subject_labels = test_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
    )



weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_ssvep_torch = torch.from_numpy(test_data).to(torch.float32)
test_labels_placeholder = torch.zeros(test_ssvep_torch.shape[0], dtype=torch.long)
test_ssvep_torch_subject = torch.from_numpy(subject_label_test_).to(torch.long)

test_dataset = EEGDataset(
    data_tensor=test_ssvep_torch,
    weigths=weights_test_torch,
    label_tensor=test_labels_placeholder,
    subject_labels=test_ssvep_torch_subject
    )

test_loader   = DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle=False
    )



final_preds_ssvep= predict(
    model_ssvep,
    window_len=WINDOW_LEN,
    loader=test_loader,
    num_samples_to_predict=50,
    device = device,
    probability=False,
    num_classes=4
    )


# -----------------------------------------------------------------------------
#  Final Submission Assembly
#
# - Convert SSVEP predictions to a DataFrame and map class indices to labels
# - Concatenate MI and SSVEP prediction DataFrames
# - Sort by ID and save as the final submission CSV
# -----------------------------------------------------------------------------

preds_ssvep_csv = pd.DataFrame({
    "id":ids_ssvep,
    "label": pd.Series(final_preds_ssvep).map(inv_mapping_ssvep).values
})



submission = pd.concat([
    preds_mi_csv,
    preds_ssvep_csv
]).sort_values(
    by="id"
).reset_index(
    drop=True
)


submission.to_csv(SUBMISSION_PATH,index=False)


# -----------------------------------------------------------------------------
# 🙏 Thanks for reading!
#
# This marks the end of our EEG classification pipeline.
# We hope this work contributes meaningfully to the competition and beyond.
# Good luck, and thank you for your time and attention!
# -----------------------------------------------------------------------------

