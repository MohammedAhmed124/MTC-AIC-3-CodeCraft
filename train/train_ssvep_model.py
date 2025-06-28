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



from utils.preprocessing import preprocess_data,preprocess_one_file
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







from utils.CustomDataset import EEGDataset
from utils.augmentation import augment_data


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
            save_path="train.py_checkpoints/SSVEPCheckpoints",
            domain_lambda=0.0,
            lambda_scheduler_fn=None,
            adversarial_training=False,
            n_classes=4,
            device=device,
            save_best_only=True
    )