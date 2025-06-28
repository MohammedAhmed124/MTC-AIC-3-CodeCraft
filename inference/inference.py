import sys , os

# -----------------------------------------------------------------------------
# 1. Project Path Setup
#    - Add project root to Python path for consistent imports
# -----------------------------------------------------------------------------



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -----------------------------------------------------------------------------
# 2. Core Imports
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





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))
DATA_FIF_DIR = os.path.join(ROOT_PATH,"data_fif")


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
    ROOT_PATH,
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
    probability=True
    )









model_mi_2 = MTCFormer(depth=3,
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



optimizer = Adam(model_mi_2.parameters(), lr=0.002)

checkpoint_path = os.path.join(
    ROOT_PATH,
    "checkpoints",
    "model_2_mi_checkpoint",
    "best_model_.pth"
    )

checkpoint = torch.load(checkpoint_path, weights_only=False)

model_mi_2.load_state_dict(checkpoint['model_state_dict'] , strict=False)

print("Preprocessing data for model 2...... ")
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
    model_mi_2,
    window_len=WINDOW_LEN,
    loader=test_loader,
    num_samples_to_predict=50,
    device = device,
    probability=True
    )





model_mi_three = MTCFormer(depth=2,
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



optimizer = Adam(model_mi_three.parameters(), lr=0.002)

checkpoint_path = os.path.join(
    ROOT_PATH,
    "checkpoints",
    "model_3_mi_checkpoint",
    "best_model_.pth"
    )

checkpoint = torch.load(checkpoint_path, weights_only=False)

model_mi_three.load_state_dict(checkpoint['model_state_dict'] , strict=False)

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
preds_mi_three = predict(
    model_mi_three,
    window_len=WINDOW_LEN,
    loader=test_loader,
    num_samples_to_predict=50,
    device = device,
    probability=True
    )



from utils.rank_ensemble import RankAveragingEnsemble

probs_list =  [
    preds_mi_one,
    preds_mi_two,
    preds_mi_three
]
weights = [  1  ,   0.5   ,   1  ]

final_mi_predictions = RankAveragingEnsemble(
    prob_list=probs_list,
    weights=weights
)


preds_mi_csv = pd.DataFrame({
    "id":ids_mi,
    "label": pd.Series(final_mi_predictions).map(inv_mapping_mi).values
})







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
    ROOT_PATH,
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

submission_save_path = os.path.join(ROOT_PATH , "submission.csv")
submission.to_csv(submission_save_path,index=False)