import sys
import os
import glob
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader



# -----------------------------------------------------------------------------
# Hey! I'm Mohammed Ahmed Metwally (or just Mohammed A.Metwally 😊), the Team Leader of CodeCraft.

# In this script, we’re jumping into the training process for **Model 1 (MI task)** .

# This script handles everything: preprocessing the EEG data, converting it to PyTorch 
# tensors, building the model, and training it from scratch.

# The configuration and hyperparameters here were chosen based on our early MI experiments — 


# the theory behind the model design, preprocessing steps, or training tricks,
# is mostly available in our system description paper — we kept this code focused and practical.

# Let’s dive in 
# -----------------------------------------------------------------------------




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.extractors import extract_trial , extract_subject_labels , extract_data


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# 1. We Extract MI task EEG data and labels from MNE .fif files
# 
#    - Set paths to training, validation, and test .fif files under data_fif/
#    - Read each .fif file using MNE and extract raw EEG data and labels
#    - Labels are extracted from the `description` field in raw annotations
#    - Class mapping is applied: "Left" → 0, "Right" → 1
#    - Also extract subject-level labels (e.g., subject ID) from the raw objects
# -----------------------------------------------------------------------------

print("extracting data for futher preprocessing...",end = "\n\n")


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))
DATA_FIF_DIR = os.path.join(ROOT_PATH,"data_fif")


test_file_paths_mi = glob.glob(os.path.join(DATA_FIF_DIR, "test/MI/*.fif"))
train_file_paths_mi = glob.glob(os.path.join(DATA_FIF_DIR, "train/MI/*.fif"))
val_file_paths_mi = glob.glob(os.path.join(DATA_FIF_DIR, "validation/MI/*.fif"))





mapping_mi = {
    "Left":0,
    "Right":1
}   

test_data_mi , test_labels_mi , ids_mi = extract_data(test_file_paths_mi , return_id = True)
train_data_mi , train_labels_mi , ids = extract_data(train_file_paths_mi , return_id = True)
val_data_mi , val_labels_mi , ids = extract_data(val_file_paths_mi , return_id = True)




train_labels_mi_mapped = np.array([mapping_mi[x] for x in train_labels_mi])
val_labels_mi_mapped = np.array([mapping_mi[x] for x in val_labels_mi])


train_subject_labels = extract_subject_labels(train_data_mi)
val_subject_labels = extract_subject_labels(val_data_mi)
test_subject_labels = extract_subject_labels(test_data_mi)


# -----------------------------------------------------------------------------
# [Optional] Test Pipeline Mode: --test_pipeline
#
#    - When this flag is passed, the script enables a lightweight debug mode
#      intended for fast testing and development.
#
#    - Instead of using the full dataset, it truncates the number of samples
#      from each split train to only the **first 50 trials**.
#
#    - This affects:
#        - Raw EEG data (`train_data_mi`, etc.)
#        - Corresponding labels (`train_labels_mi_mapped`, etc.)
#        - Subject ID labels (`train_subject_labels`, etc.)
#
#    - This mode is useful for verifying pipeline correctness quickly, without
#      waiting for full data to load or process.
#
# -----------------------------------------------------------------------------



import argparse
# --- argparse block ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_pipeline",
    action="store_true",
    help="Run pipeline on small subset of data (50 samples)"
    )
args = parser.parse_args()
TEST_MODE = args.test_pipeline




if TEST_MODE:
    print("[TEST PIPELINE ENABLED] Truncating to 50 samples per split.\n")
    train_subject_labels = train_subject_labels[:50]
    train_data_mi = train_data_mi[:50]
    train_labels_mi_mapped = train_labels_mi_mapped[:50]



# -----------------------------------------------------------------------------
# 2. Preprocess EEG data: we apply sequential preprocessing steps to each recording here
#
#    For each raw EEG trial, we apply the following pipeline:
#
#    1. **Notch Filtering**:
#       - Remove powerline noise at 50Hz and its harmonic at 100Hz.
#       - Frequencies removed: [50Hz, 100Hz] with notch width = 1.0 Hz.
#
#    2. **Bandpass Filtering**:
#       - Retain signal components in the frequency band [6Hz, 30Hz].
#       - Removes both slow drifts and high-frequency muscle artifacts.
#
#    3. **Channel Picking**:
#       - Extract a selected subset of relevant EEG and sensor channels:
#         ['C3', 'C4', 'CZ', 'FZ', 'Acc_norm', 'gyro_norm', 'Validation']
#       - EEG channels capture motor imagery; motion sensors help for artifacts.
#       - The columns were specifcally chosesn based on research.
#
#    4. **Windowing**:
#       - Extract fixed-length windows from each trial with stride. primarily to increase the number of training examples.
#       - Window size = 600 samples, stride = 35 samples.
#       - Allows multiple overlapping snapshots per trial to increase data.
#
#    5. **Normalization**:
#       - Each channel is standardized (mean=0, std=1) per window.
#       - The 'Validation' column is excluded from normalization
#         to preserve its binary nature (indicates valid vs. artifact).
#
#    Outputs:
#      - Preprocessed windowed data
#      - Class labels per window
#      - Subject labels per window (for subject-wise CV)
#      - Window weights (optional, for loss weighting or sample quality)
# -----------------------------------------------------------------------------


from utils.preprocessing import preprocess_data,preprocess_one_file
print("Preprocessing data, This may take a while... ",end = "\n\n")


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
train_data,weights_train,windowed_train_labels,subject_label_train_, WINDOW_LEN = preprocess_data(
    train_data_mi,
    labels =train_labels_mi_mapped,
    subject_labels = train_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
)


val_data,weights_val,windowed_val_labels,subject_label_val_, WINDOW_LEN = preprocess_data(
    val_data_mi,
    labels = val_labels_mi_mapped,
    subject_labels = val_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
)
test_data,weights_test, _ ,subject_label_test_, WINDOW_LEN= preprocess_data(
    test_data_mi,
    labels = test_labels_mi,
    subject_labels = test_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
    )




# -----------------------------------------------------------------------------
# 3. Convert preprocessed EEG windows into PyTorch-ready format
#
#    This stage converts numpy arrays from preprocessing into torch Tensors
#    with correct dtypes and wraps them into PyTorch-compatible Datasets and
#    DataLoaders. This makes the data pipeline ready for training.
#
#    Steps:
#    -------------------------------------------------------------------------
#    1. **Torch Conversion**:
#       - Input data (EEG windows) is cast to `torch.float32`.
#       - Labels and subject IDs are cast to `torch.long` (required for loss).
#       - Sample weights are cast to `torch.float32` for possible loss weighting.
#
#    2. **Test Data Handling**:
#       - Since test labels are unknown, placeholder zeros are used for compatibility.
#
#    3. **Custom Dataset**:
#       - We use a custom `EEGDataset` class that wraps:
#           - EEG windows
#           - Sample weights
#           - Class labels
#           - Subject labels (for subject-level analysis)
#       - Optional online data augmentation (enabled for training only).
#
#    4. **DataLoaders**:
#       - Train loader uses batching and shuffling for training.
#       - Val and Test loaders load the full set in one batch for deterministic evaluation.
#
#    5. **Device Setup**:
#       - Automatically selects GPU (`cuda`) if available, otherwise falls back to CPU.
# -----------------------------------------------------------------------------



from utils.CustomDataset import EEGDataset
from utils.augmentation import augment_data


print("Data Preparation.... Wrapping preprocessed data inside tensor datasets....",end = "\n\n")


if TEST_MODE:
    batch_size=10
else:
    batch_size=100

# Convert numpy arrays to PyTorch tensors with correct dtypes
orig_labels_val_torch = torch.from_numpy(val_labels_mi_mapped).to(torch.long) # Original labels for validation aggregation

train_mi_torch = torch.from_numpy(train_data).to(torch.float32)
train_mi_labels_torch = torch.from_numpy(windowed_train_labels).to(torch.long)
weights_train_torch = torch.from_numpy(weights_train).to(torch.float32) # Ensure float32
train_mi_torch_subject = torch.from_numpy(subject_label_train_).to(torch.long)


val_mi_torch = torch.from_numpy(val_data).to(torch.float32)
val_mi_labels_torch = torch.from_numpy(windowed_val_labels).to(torch.long)
weights_val_torch = torch.from_numpy(weights_val).to(torch.float32) # Ensure float32
val_mi_torch_subject = torch.from_numpy(subject_label_val_).to(torch.long)

test_mi_torch = torch.from_numpy(test_data).to(torch.float32)
weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_labels_placeholder = torch.zeros(test_mi_torch.shape[0], dtype=torch.long)
test_mi_torch_subject = torch.from_numpy(subject_label_test_).to(torch.long)


# Create TensorDatasets
train_dataset = EEGDataset(train_mi_torch, weights_train_torch, train_mi_labels_torch , train_mi_torch_subject,augment=True,augmentation_func=augment_data)
val_dataset = EEGDataset(val_mi_torch, weights_val_torch, val_mi_labels_torch , val_mi_torch_subject)
test_dataset = EEGDataset(test_mi_torch, weights_test_torch, test_labels_placeholder , test_mi_torch_subject)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) # Full batch for validation
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False) # Full batch for test


device_to_work_on = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_to_work_on)



# -----------------------------------------------------------------------------
# 4. Train MTCFormerV3 on Motor Imagery (MI) with long patience and mild adversarial training
#
#    This setup trains a moderately deep MTCFormer model on MI EEG data,
#    using domain adaptation and **soft adversarial defense**.
#    It features high training patience for thorough convergence monitoring.
#
#     Model Configuration:
#    -------------------------------------------------------------------------
#    - depth = 2:
#        - Two convolutional-attention layers for balanced depth and stability
#    - kernel_size = 5:
#        - Small kernel for capturing local temporal structure
#    - n_times = 600:
#        - 600-sample input windows (e.g., 3 seconds at 200Hz)
#    - chs_num = 7:
#        - Total channels: EEG + motion + Validation marker
#    - eeg_ch_nums = 4:
#        - EEG channels only
#    - class_num = 2:
#        - Binary classification (Left vs Right Motor Imagery)
#    - class_num_domain = 30:
#        - Subject/domain IDs for domain loss
#    - Dropouts:
#        - modulator_dropout = 0.3
#        - mid_dropout = 0.5
#        - output_dropout = 0.5
#    - Weight Initialization:
#        - mean = 0, std = 0.5
#
#    ⚙️ Optimizer and Learning Rate Schedule:
#    -------------------------------------------------------------------------
#    - Optimizer: Adam with learning rate = 0.002
#    - LR Scheduler: MultiStepLR that decays LR at epoch 70 (gamma = 0.1)
#
#    ✅ Domain Adaptation:
#    - domain_lambda = 0.01 → enables domain classification loss with low weight
#
#    ✅ Adversarial Training (Soft):
#    - adversarial_training = True
#    - adversarial_steps = 1 → uses FGSM (Fast Gradient Sign Method)
#    - adversarial_epsilon = 0.05 → maximum allowed perturbation
#    - adversarial_alpha = 0.005 → step size for perturbation generation
#
#     Training Strategy:
#    - n_epochs = 200
#    - patience = 100 
#    - save_best_only = True → only saves model on validation improvement
#    - save_path = train.py_checkpoints/MI_Checkpoints/model1
# -----------------------------------------------------------------------------

from torch.optim.lr_scheduler import *
from model.MTCformerV3 import MTCFormer
from utils.training import train_model,predict

model_former_curr = MTCFormer(depth=2,
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

optimizer = Adam(model_former_curr.parameters(), lr=0.002)
criterion = CrossEntropyLoss(reduction="none")
scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

save_path=os.path.join(SCRIPT_PATH,"checkpoints","model_1_mi_checkpoint")
train_model(model_former_curr,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        window_len=WINDOW_LEN,
        original_val_labels=orig_labels_val_torch,
        n_epochs=250,
        patience=100,
        scheduler=scheduler,
        domain_lambda=0.01,
        lambda_scheduler_fn=None,
        adversarial_steps =1,
        adversarial_epsilon=0.05,
        adversarial_alpha = 0.005,
        adversarial_training=True,
        save_best_only=True,
        save_path = save_path,
        n_classes=2,
        device = device
    )



#Best Checkpoint for this training session (not the absolute best checkpoint) is saved to  project_directory/train/checkpoints/model_1_mi_checkpoint/best_model_.pth
# -----------------------------------------------------------------------------
# 🙏 Thanks for reading!
#
# This marks of the training pipeline for model1 MI.
# We hope this work contributes meaningfully to the competition and beyond.
# Good luck, and thank you for your time and attention!
# -----------------------------------------------------------------------------

