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

# In this script, we’re jumping into the training process for **Model  (MI task)** .

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

mapping_mi = {
    "Left":0,
    "Right":1
}
inverse_mapping_mi = {
    v:k for k , v in mapping_mi.items()
}   




# This part handle data extraction.
# First, we select only the important EEG channels for Motor Imagery (C3, C4, CZ, FZ),
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
# - We grab metadata: subject index, trial ID, and MI task label (left/right).
# 
# Output format:
# - trials: numpy array of shape (n_trials, n_channels, n_timepoints)
# - labels: list of ground-truth MI classes (e.g., 0 for left, 1 for right)
# - subject_ids: numeric subject indices (for domain adaptation, etc.)
# - trial_ids: for debugging or visualization


loader = Loader(
    base_path=DATA_DIR,
    label_mapping = mapping_mi,
    dataset_type="train",
)
quality_filter_tuple = (0.72,6,None) #(minimum validation mean accepted , maximum gyroscope mean accepted , maximum accelerometer mean accepted )
train_trials = loader.get_trials_from_df(loader.loaded_df,task_type="MI")

_, train_subjects, _, train_data, train_labels = loader.load_data_parallel(
    train_trials,
    quality_filter=quality_filter_tuple,
    return_numpy=True,
    max_workers=4,
    )


loader = Loader(
    base_path=DATA_DIR,
    label_mapping = mapping_mi,
    dataset_type="validation"
)
validation_trials = loader.get_trials_from_df(loader.loaded_df,task_type="MI")
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
# . Preprocess EEG data: we apply a step-by-step pipeline to every trial.
#
#    Here’s what happens to each raw EEG trial:
#
#    1. **Bandpass Filtering**:
#       - We apply a bandpass filter to keep only the frequencies we care about
#         (e.g., 6Hz to 24Hz).
#
#    2. **Windowing**:
#       - Each trial is split into overlapping windows (e.g., 600 samples long, stride 35).
#       - This gives us more training samples and allows the model to learn from different
#         parts of the signal.
#
#    3. **Window Weighting**:
#       - Each trial is assigned a weight based on signal quality.
#       - Specifically: weight = mean of the 'Validation' signal in the trial.
#         (Higher = better quality; lower = more artifacts).
#
#    4. **Cropping**:
#       - We crop the trial using a `crop_range` (e.g., 2.5s to 7s).
#       - This removes the edges of the trial, which usually contain idle or noisy parts
#         unrelated to the motor imagery task.
#
#    5. **Normalization**:
#       - Each channel in each window is normalized to zero mean and unit variance.
#       - The 'Validation' column is excluded from normalization, to avoid distorting its binary nature.
#
#    Outputs:
#      - Preprocessed and windowed EEG data
#      - Class labels (for MI tasks) per window
#      - Subject IDs per window (useful for domain adaptation)
#      - Window weights (used later to weigh loss or exclude bad windows)
# -----------------------------------------------------------------------------



from utils.preprocessing import SignalPreprocessor
print("Preprocessing data, This may take a while... ",end = "\n\n")



preprocessor = SignalPreprocessor(
    fs=250,                                                
    bandpass_low=6.0,                     
    bandpass_high=24.0,                  
    n_cols_to_filter=4,                   
    window_size=600,                      
    window_stride=35,                    
    idx_to_ignore_normalization=-1,        
    crop_range=(2.5 , 7)              
)


preprocessed_train_data , preprocessed_train_labels , preprocessed_train_subject_ids , weights_train = preprocessor.apply_preprocessing(train_data, train_labels , train_subjects)

preprocessed_val_data , preprocessed_val_labels , preprocessed_val_subject_ids , weights_val = preprocessor.apply_preprocessing(val_data, val_labels , val_subjects)

num_windows_per_trial = preprocessor.num_windows_per_trial

print(preprocessed_train_data.shape)
# -----------------------------------------------------------------------------
# . Convert preprocessed EEG windows into PyTorch-ready format
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
    batch_size=1000


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
# 4. Train MTCFormerV3 on Motor Imagery (MI) with long patience and scheduled adversarial training
#
#    This setup trains a deeper MTCFormer model on MI EEG data,
#    using domain adaptation and **dynamically weighted adversarial defense**.
#    It features long training patience and uses cosine learning rate annealing.
#
#     Model Configuration:
#    -----------------------------------------------------------------------------
#    - depth = 3:
#        - Three convolutional-attention layers for deeper feature extraction
#    - kernel_size = 50:
#        - Large temporal kernel to capture long-range dependencies
#    - modulator_kernel_size = 30:
#        - Temporal receptive field for the attention modulator
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
#        - modulator_dropout ≈ 0.49
#        - mid_dropout = 0.5
#        - output_dropout ≈ 0.43
#    - Weight Initialization:
#        - mean = 0, std = 0.5
#
#     Optimizer and Learning Rate Schedule:
#    -----------------------------------------------------------------------------
#    - Optimizer: Adam with learning rate = 0.002
#    - Weight decay: ≈ 1.66e-5
#    - LR Scheduler: CosineAnnealingLR
#        - T_max = 121
#        - eta_min ≈ 0.0002
#
#     Domain Adaptation:
#    - domain_lambda = 0.05
#
#     Adversarial Training (Dynamic):
#    - adversarial_training = True
#    - adversarial_steps = 1 → uses FGSM
#    - adversarial_epsilon = 0.1
#    - adversarial_alpha = 0.005
#    - adversarial_factor varies by epoch (0.5 → 0.6 after epoch 30)
#
#     Training Strategy:
#    - n_epochs = 600
#    - patience = 100 
#    - save_best_only = False
#    - save_path = checkpoints/model_1_mi_checkpoint
#    - TensorBoard logging enabled
# -----------------------------------------------------------------------------


from torch.optim.lr_scheduler import *
from model.MTCformerV3 import MTCFormer
from utils.training import train_model
model_former_curr = MTCFormer(
    depth=3,
    kernel_size=50,
    modulator_kernel_size=30,
    n_times=600,
    chs_num=7,
    eeg_ch_nums=4,
    class_num=2,
    class_num_domain=30,
    modulator_dropout=0.48929137963218305,
    mid_dropout=0.5,
    output_dropout=0.42685917257840517,
    k=100,
    projection_dimention=2,
    seed=4224
    ).to(device)




def scheduler_fn(epoch):
    if epoch >=30:
        domain_lambda=0.05
        adversarial_training=True
        adversarial_steps=1
        adversarial_epsilon=0.1
        adversarial_alpha= 0.005
        adversarial_factor=0.6

    else:
        domain_lambda=0.05
        adversarial_training=True
        adversarial_steps=1
        adversarial_epsilon=0.1
        adversarial_alpha= 0.005
        adversarial_factor=0.5


    return (
        domain_lambda,
        adversarial_training,
        adversarial_steps,
        adversarial_epsilon,
        adversarial_alpha,
        adversarial_factor
    )

save_path=os.path.join(SCRIPT_PATH,"checkpoints","model_1_mi_checkpoint")
training_loop_params = {
    "criterion": CrossEntropyLoss(reduction="none"),


    "optimizer_class": Adam,
    "optimizer_config": {
        "lr": 0.002,
        "weight_decay":1.6581884226239174e-05
    },


    "scheduler_class": CosineAnnealingLR,
    "scheduler_config":  {
            "T_max": 121,
            "eta_min": 0.00019632120929158656
        },

    "window_len": num_windows_per_trial,
    "n_epochs": 600,
    "patience": 100 ,
    "domain_lambda": 0.05,
    "lambda_scheduler_fn": None,
    "adversarial_steps": 1,
    "adversarial_epsilon": 0.1,
    "adversarial_alpha": 0.005,
    "adversarial_training": True,
    "save_best_only": True,
    "save_path": save_path,
    "n_classes": 2,
    "device": device,
    "update_loader": (10, 100),
    "scheduler_fn": scheduler_fn ,
    "tensorboard":True,
}

train_model(model_former_curr,
        train_loader=train_loader,
        val_loader=val_loader,
        original_val_labels=trial_level_labels_val,
        **training_loop_params
    )



#Best Checkpoint for this training session (not the absolute best checkpoint) is saved to  project_directory/train/checkpoints/model_1_mi_checkpoint/best_model_.pth
# -----------------------------------------------------------------------------
# 🙏 Thanks for reading!
#
# This marks of the training pipeline for MI model.
# We hope this work contributes meaningfully to the competition and beyond.
# Good luck, and thank you for your time and attention!
# -----------------------------------------------------------------------------



