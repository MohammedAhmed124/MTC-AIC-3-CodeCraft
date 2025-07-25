import sys
import os
import glob
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader





# -----------------------------------------------------------------------------
# Hey! This is CodeCraft Team.

# In this script, we’re jumping into the training process for **Our Single MTCFormer (SSVEP task)** .

# This script handles everything: preprocessing the EEG data, converting it to PyTorch 
# tensors, building the model, and training it from scratch.

# The configuration and hyperparameters here were chosen based on our early MI experiments — 


# the theory behind the model design, preprocessing steps, or training tricks,
# is mostly available in our system description paper — we kept this code focused and practical.

# Let’s dive in 
# -----------------------------------------------------------------------------



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.loader import Loader


import warnings
warnings.filterwarnings("ignore", category=UserWarning)




print("extracting data for futher preprocessing...",end = "\n\n")


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))
DATA_DIR = os.path.join(ROOT_PATH,"data")


mapping_ssvep = {
    "Backward":0,
    "Forward":1,
    "Left":2,
    "Right":3
}
inverse_mapping_mi = {
    v:k for k , v in mapping_ssvep.items()
}   




# This part handle data extraction.
# First, we select only the important EEG channels for SSVEP (PZ, PO7, PO8, OZ),
# and grab the motion data: AccX/Y/Z and Gyro1/2/3. 
# 
# We compute the L2 norm of both accelerometer and gyroscope signals across time,
# which gives us a single motion signal per type (acc/gyro), instead of 3 axes.
# This is useful because we care about motion intensity overall, not direction.
# we then exclude Acc (x , y , z) and gyro (1 , 2 , 3) and only keep the norms.
#
#
# We also keep the 'Validation' signal – this indicates how trustworthy the EEG
# recording is at each time point (low values = bad signal quality).
# 
# We then cut out any trials that seem too noisy:
# - If the average validation score is too low, we skip it.
# - If the subject was moving a lot (high acc or gyro norm), we also skip it.
# 
# For the valid trials:
# - We extract the EEG and motion signals as input data.
# - We grab metadata: subject index, trial ID, and SSVEP task label (left/right/Backward/Forward).
# 
# Output format:
# - trials: numpy array of shape (n_trials, n_channels, n_timepoints)
# - labels: list of ground-truth SSVEP classes
# - subject_ids: numeric subject indices (for domain adaptation)
# - trial_ids: for debugging or visualization


loader = Loader(
    base_path=DATA_DIR,
    label_mapping = mapping_ssvep,
    dataset_type="train",
)
quality_filter_tuple = (0.72,6,None) #(minimum validation mean accepted , maximum gyroscope mean accepted , maximum accelerometer mean accepted )
train_trials = loader.get_trials_from_df(loader.loaded_df,task_type="SSVEP")

_, train_subjects, _, train_data, train_labels = loader.load_data_parallel(
    train_trials,
    quality_filter=quality_filter_tuple,
    return_numpy=True,
    max_workers=4,
    )


loader = Loader(
    base_path=DATA_DIR,
    label_mapping = mapping_ssvep,
    dataset_type="validation"
)
validation_trials = loader.get_trials_from_df(loader.loaded_df,task_type="SSVEP")
_, val_subjects, _, val_data, val_labels = loader.load_data_parallel(
    validation_trials,
    quality_filter=quality_filter_tuple,
    return_numpy=True,
    max_workers=4,
    )







# -----------------------------------------------------------------------------
# [Optional] Test Pipeline Mode: --test_pipeline
#
#    - When this flag is passed, the script enables a lightweight debug mode
#      intended for fast testing and development.
#
#    - Instead of using the full dataset, it truncates the number of samples
#      from each split train to only the **first 50 trials**.
#
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
    train_subjects = train_subjects[:50]
    train_data = train_data[:50 , : , :]
    train_labels = train_labels[:50]



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# . Preprocess EEG data: we apply a step-by-step pipeline to every trial.
#
#    Here’s what happens to each raw EEG trial:
#
#    1. **Bandpass Filtering**:
#       - We apply a bandpass filter to keep only the frequencies we care about
#         (e.g., 8Hz to 14Hz).
#
#    2. **Windowing**:
#       - Each trial is split into overlapping windows (e.g., 500 samples long, stride 50).
#       - This gives us more training samples and allows the model to learn from different
#         parts of the signal.
#
#    3. **Window Weighting**:
#       - Each trial is assigned a weight based on signal quality.
#       - Specifically: weight = mean of the 'Validation' signal in the trial.
#         (Higher = better quality; lower = more artifacts).
#
#    4. **Cropping**:
#       - We crop the trial using a `crop_range` (e.g., 1.5s to 6s).
#       - This removes the edges of the trial, which usually contain idle or noisy parts
#         unrelated to the motor imagery task.
#
#    5. **Normalization**:
#       - Each channel in each window is normalized to zero mean and unit variance.
#       - The 'Validation' column is excluded from normalization, to avoid distorting its binary nature.
#
#    Outputs:
#      - Preprocessed and windowed EEG data
#      - Class labels (for SSVEP tasks) per window
#      - Subject IDs per window (useful for domain adaptation)
#      - Window weights (used later to weigh loss or exclude bad windows)
# -----------------------------------------------------------------------------


from utils.preprocessing import SignalPreprocessor
print("Preprocessing data, This may take a while... ",end = "\n\n")



preprocessor = SignalPreprocessor(
    fs=250,                                                 
    bandpass_low=8,                     
    bandpass_high=14,                  
    n_cols_to_filter=4,                   
    window_size=500,                      
    window_stride=50,                    
    idx_to_ignore_normalization=-1,        
    crop_range=(1.5 , 6)            
)


preprocessed_train_data , preprocessed_train_labels , preprocessed_train_subject_ids , weights_train = preprocessor.apply_preprocessing(train_data, train_labels , train_subjects)

preprocessed_val_data , preprocessed_val_labels , preprocessed_val_subject_ids , weights_val = preprocessor.apply_preprocessing(val_data, val_labels , val_subjects)

num_windows_per_trial = preprocessor.num_windows_per_trial


print(preprocessed_train_data.shape)






# -----------------------------------------------------------------------------
#  Convert preprocessed EEG windows into PyTorch-ready format
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


trial_level_labels_val = torch.from_numpy(val_labels).to(torch.long) 

preprocessed_train_data = torch.from_numpy(preprocessed_train_data).to(torch.float32)
preprocessed_train_labels = torch.from_numpy(preprocessed_train_labels).to(torch.long)
weights_train = torch.from_numpy(weights_train).to(torch.float32) 
preprocessed_train_subject_ids = torch.from_numpy(preprocessed_train_subject_ids).to(torch.long)


preprocessed_val_data = torch.from_numpy(preprocessed_val_data).to(torch.float32)
preprocessed_val_labels = torch.from_numpy(preprocessed_val_labels).to(torch.long)
weights_val = torch.from_numpy(weights_val).to(torch.float32) 
preprocessed_val_subject_ids = torch.from_numpy(preprocessed_val_subject_ids).to(torch.long)



train_dataset = EEGDataset(preprocessed_train_data, weights_train, preprocessed_train_labels , preprocessed_train_subject_ids,augment=True,augmentation_func=augment_data)
val_dataset = EEGDataset(preprocessed_val_data, weights_val, preprocessed_val_labels , preprocessed_val_subject_ids)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) # Full batch for validation



device_to_work_on = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_to_work_on)



# -----------------------------------------------------------------------------
# Train MTCFormerV3 for SSVEP EEG Classification with Optional Adversarial Training
#
# This configuration trains a deep MTCFormer model (5 layers) on 4-class SSVEP data,
# utilizing various dropout strategies and offering the option to include 
# adversarial training and domain adaptation dynamically using a scheduler.
#
#     Model Configuration:
# -----------------------------------------------------------------------------
# - depth = 5:
#     - Deep network with five gated attention-convolutional blocks
# - kernel_size = 50:
#     - Temporal convolution kernel captures signal patterns up to 250 ms
# - modulator_kernel_size = 30:
#     - Controls attention modulator's receptive field (150 ms)
# - n_times = 500:
#     - Input window size = 500 samples (e.g., 2.5 seconds at 200Hz)
# - chs_num = 7:
#     - Number of total input channels (EEG + motion/validation)
# - eeg_ch_nums = 4:
#     - EEG-only channels used for spatial filtering and early fusion
# - class_num = 4:
#     - 4-class classification task (e.g., SSVEP target IDs)
# - class_num_domain = 30:
#     - 30 domain labels (e.g., subject or session IDs)
# - Dropouts:
#     - domain_dropout, modulator_dropout, mid_dropout, output_dropout = 0.3
#     - Applied at different stages for regularization
# - k = 100:
#     - corresponds to low rank matrix decompistion compression dimention (lower = smaller model)
# - projection_dimention = 2:
#     - Corresponds to how rich the output dimention of the of the PointWise convolution.
# - seed = 5445:
#     - Ensures reproducibility
#
#     Optimizer and Learning Rate Schedule:
# -----------------------------------------------------------------------------
# - Optimizer: Adam
#     - Learning rate = 0.001
#     - Weight decay = 0 (no L2 regularization)
# - Scheduler: MultiStepLR
#     - Learning rate decays by factor of 0.1 at epoch 250
#
#     Domain Adaptation and Adversarial Training Schedule:
# -----------------------------------------------------------------------------
# - Controlled via `scheduler_fn(epoch)`
# - From epoch 0–39:
#     - domain_lambda = 0.0 (no domain loss)
#     - adversarial_training = False
#     - adversarial_factor = 0.5
# - From epoch 40+:
#     - domain_lambda = 0.3 (enable domain adaptation)
#     - adversarial_training = False (no adversarial training)
#     - adversarial_factor = 0.8
# - FGSM-like adversarial perturbation:
#     - adversarial_steps = 1
#     - adversarial_epsilon = 0.1
#     - adversarial_alpha = 0.005
#



from model.MTCformerV3 import MTCFormer
from torch.optim.lr_scheduler import *
from utils.training import train_model 
model_former = MTCFormer(
    depth=5,
    kernel_size=50,
    n_times=500,
    chs_num=7,
    eeg_ch_nums=4,
    class_num=4,
    class_num_domain=30,
    modulator_kernel_size=30,
    domain_dropout=0.3,
    modulator_dropout=0.3,
    mid_dropout=0.3,
    output_dropout=0.3,
    k=100,
    projection_dimention=2,
    seed = 5445
).to(device)
learning_rate=0.001


save_path = os.path.join(SCRIPT_PATH,"checkpoints","model_ssvep_checkpoint")

training_loop_params = {
    "criterion": CrossEntropyLoss(reduction="none"),


    "optimizer_class": Adam,
    "optimizer_config": {
        "lr": 0.001,
        "weight_decay":0
    },


    "scheduler_class": None,
    "scheduler_config":  {
            "T_max": 121,
            "eta_min": 0.00019632120929158656
        },

    "window_len": num_windows_per_trial,
    "n_epochs": 600,
    "patience": 600 ,
    "domain_lambda": 0.2,
    "lambda_scheduler_fn": None,
    "adversarial_steps": 1,
    "adversarial_epsilon": 0.05,
    "adversarial_alpha": 0.005,
    "adversarial_training": False,
    "save_best_only": True,
    "save_path": save_path,
    "n_classes": 4,
    "device": device,
    "update_loader": (20, 100),
    "scheduler_fn": None  # Optional: your own dynamic LR adjustment function
}

train_model(model_former,
            train_loader=train_loader,
            val_loader=val_loader,
            original_val_labels=trial_level_labels_val,
            **training_loop_params
    )


#Best Checkpoint for this training session (not the absolute best checkpoint) is saved to  project_directory/train/checkpoints/model_ssvep_checkpoint/best_model_.pth
# -----------------------------------------------------------------------------
# 🙏 Thanks for reading!
#
# This marks of the training pipeline for SSVEP Single model.
# We hope this work contributes meaningfully to the competition and beyond.
# Good luck, and thank you for your time and attention!
# -----------------------------------------------------------------------------
