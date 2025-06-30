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

# In this script, we’re jumping into the training process for **Our Single MTCFormer (SSVEP task)** .

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
# 1. We Extract SSVEP task EEG data and labels from MNE .fif files
# 
#    - Set paths to training, validation, and test .fif files under data_fif/
#    - Read each .fif file using MNE and extract raw EEG data and labels
#    - Labels are extracted from the `description` field in raw annotations
#    - Class mapping is applied: "Backward" → 0, "Forward" → 1, "Left"→ 2, "Right"→ 3
#    - Also extract subject-level labels (e.g., subject ID) from the raw objects
# -----------------------------------------------------------------------------

print("extracting data for futher preprocessing...",end = "\n\n")

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))
DATA_FIF_DIR = os.path.join(ROOT_PATH,"data_fif")


test_file_paths_ssvep = glob.glob(os.path.join(DATA_FIF_DIR, "test/SSVEP/*.fif"))
train_file_paths_ssvep = glob.glob(os.path.join(DATA_FIF_DIR, "train/SSVEP/*.fif"))
val_file_paths_ssvep = glob.glob(os.path.join(DATA_FIF_DIR, "validation/SSVEP/*.fif"))



test_data_ssvep, test_labels_ssvep , ids_ssvep = extract_data(test_file_paths_ssvep , return_id = True)
train_data_ssvep , train_labels_ssvep , ids = extract_data(train_file_paths_ssvep , return_id = True)
val_data_ssvep , val_labels_ssvep , ids = extract_data(val_file_paths_ssvep , return_id = True)



mapping_ssvep = {
    "Backward":0,
    "Forward":1,
    "Left":2,
    "Right":3
}

train_labels_ssvep_mapped = np.array([mapping_ssvep[x] for x in train_labels_ssvep])
val_labels_ssvep_mapped = np.array([mapping_ssvep[x] for x in val_labels_ssvep])


train_subject_labels = extract_subject_labels(train_data_ssvep)
val_subject_labels = extract_subject_labels(val_data_ssvep)
test_subject_labels = extract_subject_labels(test_data_ssvep)



# -----------------------------------------------------------------------------
# 2. Preprocess SSVEP EEG data: apply a sequence of preprocessing operations
#
#    For each raw EEG trial (SSVEP task), we apply the following steps:
#
#    1. **Notch Filtering**:
#       - Suppress powerline noise artifacts at 50Hz and 100Hz.
#       - Notch frequencies = [50, 100] Hz, with width = 1.0 Hz.
#
#    2. **Bandpass Filtering**:
#       - Restrict the EEG signal to the range [8Hz, 14Hz].
#       - These frequencies are chosen specifically to match the known
#         SSVEP stimulation frequencies used in the competition setup.
#
#    3. **Channel Selection**:
#       - Pick occipital and parietal channels associated with visual response:
#         ['OZ', 'PO7', 'PO8', 'PZ', 'Acc_norm', 'gyro_norm', 'Validation']
#
#    4. **Windowing**:
#       - Extract sliding windows of fixed length across each trial.
#       - Window size = 500 samples, stride = 50 samples.
#       - This produces overlapping temporal snapshots from each recording.
#
#    5. **Normalization**:
#       - Standardize all numeric channels in each window (mean=0, std=1).
#       - 'Validation' column is excluded from normalization to preserve its
#         binary integrity (often used to flag artifact-contaminated windows).
#
#    The result is a windowed, preprocessed dataset ready for training.
# -----------------------------------------------------------------------------


from utils.preprocessing import preprocess_data,preprocess_one_file
print("Preprocessing data, This may take a while... ",end = "\n\n")


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
train_data,weights_train,windowed_train_labels,subject_label_train_, WINDOW_LEN = preprocess_data(
    train_data_ssvep,
    labels =train_labels_ssvep_mapped,
    subject_labels = train_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
)


val_data,weights_val,windowed_val_labels,subject_label_val_, WINDOW_LEN = preprocess_data(
    val_data_ssvep,
    labels = val_labels_ssvep_mapped,
    subject_labels = val_subject_labels,
    preprocess_func=preprocess_one_file,
    params = params,
    n_jobs=4
)
test_data,weights_test, _ ,subject_label_test_, WINDOW_LEN= preprocess_data(
    test_data_ssvep,
    labels = test_labels_ssvep,
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

batch_size = 100
orig_labels_train_torch = torch.from_numpy(train_labels_ssvep_mapped).to(torch.long)
orig_labels_val_torch = torch.from_numpy(val_labels_ssvep_mapped).to(torch.long)

train_ssvep_torch = torch.from_numpy(train_data).to(torch.float32)
train_ssvep_labels_torch = torch.from_numpy(windowed_train_labels).to(torch.long)
train_ssvep_torch_subject = torch.from_numpy(subject_label_train_).to(torch.long)
weights_train_torch = torch.from_numpy(weights_train).to(torch.float32)


val_ssvep_torch = torch.from_numpy(val_data).to(torch.float32)
val_ssvep_labels_torch = torch.from_numpy(windowed_val_labels).to(torch.long)
val_ssvep_torch_subject = torch.from_numpy(subject_label_val_).to(torch.long)
weights_val_torch = torch.from_numpy(weights_val).to(torch.float32)




weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_ssvep_torch = torch.from_numpy(test_data).to(torch.float32)
test_labels_placeholder = torch.zeros(test_ssvep_torch.shape[0], dtype=torch.long)
test_ssvep_torch_subject = torch.from_numpy(subject_label_test_).to(torch.long)





train_dataset = EEGDataset(train_ssvep_torch,weights_train_torch,train_ssvep_labels_torch, train_ssvep_torch_subject , augment=False)
val_dataset =EEGDataset(val_ssvep_torch,weights_val_torch,val_ssvep_labels_torch,val_ssvep_torch_subject)
test_dataset = EEGDataset(test_ssvep_torch,weights_test_torch,test_labels_placeholder,test_ssvep_torch_subject)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader   = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# -----------------------------------------------------------------------------
# 4. Initialize and train the MTCFormerV3 model on preprocessed SSVEP data
#
#    In this stage, we:
#    - Load the MTCFormerV3 architecture (from model.MTCformerV3)
#    - Instantiate it with the chosen hyperparameters
#    - Set up the optimizer, scheduler, and loss function
#    - Train it using the previously constructed DataLoaders
#
#     Model Architecture: `MTCFormer`
#    -------------------------------------------------------------------------
#    - depth=1:
#        - Sets the number of convolutional-attention blocks stacked in the model.
#        - Controls the depth of temporal feature extraction.
#    - kernel_size=10:
#        - Temporal convolutional kernel size for initial feature extraction.
#    - n_times=500:
#        - Length of the input sequence (windowed signal).
#    - chs_num=7:
#        - Total number of input channels (EEG + sensors).
#    - eeg_ch_nums=4:
#        - Number of EEG-only channels (used for channel separation logic).
#    - class_num=4:
#        - Number of output classes for the main task (e.g., 4 SSVEP targets).
#    - class_num_domain=30:
#        - Number of domain labels (e.g., subjects) — used **only** if domain adaptation is active.
#    - modulator_kernel_size=10:
#        - Kernel size for modulator block (learns temporal adaptation).
#    - Various dropout settings:
#        - domain_dropout, modulator_dropout, mid_dropout, output_dropout = 0.7
#        - Encourage regularization at different parts of the model.
#    - weight_init_std = 0.05, weight_init_mean = 0.0:
#        - Sets the normal distribution used for weight initialization.
#
#     Training Setup:
#    -------------------------------------------------------------------------
#    - Optimizer: Adam with learning rate = 0.001
#    - Loss: CrossEntropyLoss (per-sample) for flexibility with weights or masking
#    - Scheduler: MultiStepLR with a decay at epoch 250 by factor of 0.1
#
#    ❌ Domain Adaptation & Adversarial Training:
#    -------------------------------------------------------------------------
#    - `domain_lambda = 0.0`: Domain adaptation is **disabled**
#    - `adversarial_training = False`: No adversarial defense applied
#    - This makes the training focus purely on classification with no auxiliary loss
#
#    ✅ Early Stopping & Checkpointing:
#    - Patience: 50 epochs without improvement
#    - Best model saved to: `train.py_checkpoints/SSVEPCheckpoints`
# -----------------------------------------------------------------------------

from model.MTCformerV3 import MTCFormer
from torch.optim.lr_scheduler import *
from utils.training import train_model , predict
model_former = MTCFormer(
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
learning_rate=0.001
optimizer = Adam(model_former.parameters(), lr=learning_rate) # Use the reduced learning_rate
criterion = CrossEntropyLoss(reduction="none")

scheduler = MultiStepLR(optimizer, milestones=[250], gamma=0.1)


save_path = os.path.join(SCRIPT_PATH,"checkpoints","model_ssvep_checkpoint")
train_model(model_former,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            window_len=WINDOW_LEN,
            original_val_labels=orig_labels_val_torch,
            n_epochs=220,
            patience=50,
            scheduler=scheduler,
            save_path=save_path,
            domain_lambda=0.0,
            lambda_scheduler_fn=None,
            adversarial_training=False,
            n_classes=4,
            device=device,
            save_best_only=True
    )


#Best Checkpoint for this training session (not the absolute best checkpoint) is saved to  project_directory/train/checkpoints/model_ssvep_checkpoint/best_model_.pth
# -----------------------------------------------------------------------------
# 🙏 Thanks for reading!
#
# This marks of the training pipeline for SSVEP Single model.
# We hope this work contributes meaningfully to the competition and beyond.
# Good luck, and thank you for your time and attention!
# -----------------------------------------------------------------------------
