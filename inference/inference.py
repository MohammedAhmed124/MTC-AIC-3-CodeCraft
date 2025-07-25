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
from utils.augmentation import augment_data
from utils.training import  predict_optimized
from utils.CustomDataset import EEGDataset
from torch.utils.data import DataLoader
from utils.loader import Loader
from utils.preprocessing import SignalPreprocessor
# -----------------------------------------------------------------------------
# Hey! I'm Mohammed Ahmed Metwally.

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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





##----- MI pipeline ------##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




from torch.optim.lr_scheduler import *
from model.MTCformerV3 import MTCFormer
from utils.training import train_model
model_mi_1 = MTCFormer(depth=3,
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



checkpoint_path = os.path.join(
    checkpoint_dicetory,
    "checkpoints",
    "model_1_mi_checkpoint",
    "best_model_.pth"
    )

checkpoint = torch.load(checkpoint_path, weights_only=False)

model_mi_1.load_state_dict(checkpoint['model_state_dict'] , strict=False)


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

loader = Loader(
    base_path=DATA_DIR,
    label_mapping = mapping_mi,
    dataset_type="test",
)

test_trials = loader.get_trials_from_df(loader.loaded_df,task_type="MI")

ids_mi, test_subjects, _, test_data, test_labels = loader.load_data_parallel(
    test_trials,
    quality_filter=None,
    return_numpy=True,
    max_workers=4,
    )

print("Preprocessing data for model 1 ..... ")


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

preprocessed_test_data , preprocessed_test_labels , preprocessed_test_subject_ids , weights_test = preprocessor.apply_preprocessing(test_data, test_labels , test_subjects)

num_windows_per_trial = preprocessor.num_windows_per_trial


test_mi_torch = torch.from_numpy(preprocessed_test_data).to(torch.float32)
weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_labels_placeholder = torch.zeros(preprocessed_test_labels.shape[0], dtype=torch.long)
test_mi_torch_subject = torch.from_numpy(preprocessed_test_subject_ids).to(torch.long)


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

preds_mi_one = predict_optimized(
        model_mi_1,
        windows_per_trial=num_windows_per_trial,
        loader=test_loader,
        probability=False,
        device=device,
        )


preds_mi_csv = pd.DataFrame({
    "id":ids_mi,
    "label": pd.Series(preds_mi_one).map(inverse_mapping_mi).values
})






##------ SSVEP pipeline-------##

from model.MTCformerV2 import MTCFormer
from torch.optim.lr_scheduler import *
from utils.training import train_model 
model_ssvep = MTCFormer(
    depth=2,
    kernel_size=50,
    n_times=500,
    chs_num=7,
    eeg_ch_nums=4,
    class_num=4,
    class_num_domain=30,
    modulator_kernel_size=30,
    domain_dropout=0.4,
    modulator_dropout=0.4,
    mid_dropout=0.4,
    output_dropout=0.6,
    k=100,
    projection_dimention=2,
    seed = 5445
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



SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))
DATA_DIR = os.path.join(ROOT_PATH,"data")


mapping_ssvep = {
    "Backward":0,
    "Forward":1,
    "Left":2,
    "Right":3
}
inv_mapping_ssvep = {v:k for k,v in mapping_ssvep.items()}


loader = Loader(
    base_path=DATA_DIR,
    label_mapping = mapping_ssvep,
    dataset_type="test",
)

test_trials = loader.get_trials_from_df(loader.loaded_df,task_type="SSVEP")

ids_ssvep, test_subjects, _, test_data, test_labels = loader.load_data_parallel(
    test_trials,
    quality_filter=None,
    return_numpy=True,
    max_workers=4,
    )


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



preprocessed_test_data , preprocessed_test_labels , preprocessed_test_subject_ids , weights_test = preprocessor.apply_preprocessing(test_data, test_labels , test_subjects)
num_windows_per_trial = preprocessor.num_windows_per_trial



weights_test_torch = torch.from_numpy(weights_test).to(torch.float32)
test_ssvep_torch = torch.from_numpy(preprocessed_test_data).to(torch.float32)
test_labels_placeholder = torch.zeros(test_ssvep_torch.shape[0], dtype=torch.long)
test_ssvep_torch_subject = torch.from_numpy(preprocessed_test_subject_ids).to(torch.long)

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



final_preds_ssvep = predict_optimized(
    model_ssvep,
    windows_per_trial=num_windows_per_trial,
    loader=test_loader,
    device = device,
    probability=False
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

