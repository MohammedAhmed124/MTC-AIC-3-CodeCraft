import sys
import os
import glob
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.extractors import extract_trial , extract_subject_labels , extract_data


import warnings
warnings.filterwarnings("ignore", category=UserWarning)



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



import logging
from utils.preprocessing import preprocess_data,preprocess_one_file
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
    "window_size": 1200,
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






from utils.CustomDataset import EEGDataset
from utils.augmentation import augment_data
batch_size=250

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from model.MTCformerV3 import MTCFormer
from utils.training import train_model , predict
from torch.optim.lr_scheduler import *


model_former = MTCFormer(depth=3,
                    kernel_size=10,
                    n_times=1200,
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


optimizer = Adam(model_former.parameters(), lr=0.002)
criterion = CrossEntropyLoss(reduction="none")
scheduler = MultiStepLR(optimizer, milestones=[60], gamma=0.1)

save_path = os.path.join(SCRIPT_PATH,"train.py_checkpoints","MI_Checkpoints","model2")
best_epoch = train_model(model_former,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                window_len=WINDOW_LEN,
                original_val_labels=orig_labels_val_torch,
                n_epochs=500,
                patience=25,
                scheduler=scheduler,
                domain_lambda=0.01,
                lambda_scheduler_fn=None,
                adversarial_training=True,
                adversarial_alpha=0.01,
                adversarial_epsilon=0.01,
                adversarial_factor=0.4,
                adversarial_steps=1,
                save_path = save_path,
                device = device,
                save_best_only=True,
                n_classes=2
                )


