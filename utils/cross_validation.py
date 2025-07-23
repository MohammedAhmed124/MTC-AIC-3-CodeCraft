from sklearn.model_selection import GroupKFold
import copy
from utils.training import train_model , predict
import torch
import numpy as np
from utils.CustomDataset import EEGDataset
from torch.utils.data import DataLoader
import os , sys

def find_original_indices(windows_indices , WINDOW_LEN, max_trials_count , return_size=None):

    final_indices = np.zeros(return_size)
    j = 0
    for i in range(max_trials_count):
        # if WINDOW_LEN * i not in windows_indices:
        #     print(f"Warning: trial {i} not found in windows_indices")

        if WINDOW_LEN*i in windows_indices:
            j+=1
            index_placement = np.where(windows_indices == WINDOW_LEN*i)[0][0]/WINDOW_LEN
            assert index_placement.is_integer()
            final_indices[int(index_placement)] = i

    print(f"_------found----- ", j , "  trials")
    return np.asarray(final_indices , dtype=np.int32)

def LeaveKSubjectOutCV(
        model_class,
        model_params,
        X,
        y,
        weights,
        before_windowing_y,
        subjects,
        test_loader=None,
        val_loader=None,
        augmentation_func = None,
        training_loop_params = None,
        n_splits = 5,
        random_state = 42,
        batch_size = 100,
        models_folder_path="checkpoints/model_1_mi",
        device = "cuda"
        ):
    
    if isinstance(device,str):
        device = torch.device(device)

    if not training_loop_params:
        training_loop_params={}

    GroupFold = GroupKFold(n_splits=n_splits )

    n_classes = training_loop_params["n_classes"]
    window_len =training_loop_params["window_len"]
    n_samples_val = int(val_loader.dataset.__len__()/window_len)
    n_samples_test = int(test_loader.dataset.__len__()/window_len)
    n_samples_train = int(len(weights))
    n_samples_train_trial_level = int(n_samples_train/window_len)



    OutOfFoldPredictions = np.zeros((n_samples_train_trial_level, n_classes))
    validation_set_predictions = np.zeros((n_samples_val, n_classes))
    test_set_predictions = np.zeros((n_samples_test , n_classes))


    for fold_id , (train_idx , test_idx ) in enumerate(GroupFold.split(X, y , groups=subjects)):



        X_train, X_test = X[train_idx].to(device), X[test_idx].to(device)
        y_train, y_test = y[train_idx].to(device), y[test_idx].to(device)
        subject_train,subject_test = subjects[train_idx].to(device) , subjects[test_idx].to(device)
        weights_train , weights_test = weights[train_idx].to(device) , weights[test_idx].to(device)

        n_samples_X_test = int(len(test_idx)/window_len)


        trial_level_indices = find_original_indices(test_idx , window_len, n_samples_train_trial_level , return_size=n_samples_X_test)


        
        before_windowing_y_test = before_windowing_y[trial_level_indices]

        model = model_class(**model_params).to(device)





        X_train_dataset = EEGDataset(
            X_train,
            weights_train,
            y_train,
            subject_train,
            augment=True if augmentation_func else False,
            augmentation_func=augmentation_func
            )
        X_test_dataset = EEGDataset(
            X_test,
            weights_test,
            y_test ,
            subject_test
            )

        X_train_loader = DataLoader(
            X_train_dataset,
            batch_size=batch_size,
            shuffle=True
            )
        X_test_loader = DataLoader(
            X_test_dataset,
            batch_size=len(X_test_dataset),
            shuffle=False
            )
        
        training_loop_params["save_path"] = os.path.join(models_folder_path,f"fold_{fold_id+1}")
        train_model(
        model,
        train_loader=X_train_loader,
        val_loader=X_test_loader,
        original_val_labels=before_windowing_y_test,
        **training_loop_params
        )

        cpu_device = torch.device("cpu")
        cloned_model = model.to(cpu_device)
        preds_prob_fold = predict(
        model,
        window_len=training_loop_params["window_len"],
        loader=X_test_loader,
        num_samples_to_predict=n_samples_X_test,
        device =cpu_device,
        probability=True
        )

        preds_prob_validation = predict(
        model,
        window_len=training_loop_params["window_len"],
        loader=val_loader,
        num_samples_to_predict=n_samples_val,
        device = cpu_device,
        probability=True
        )


        preds_prob_test = predict(
        model,
        window_len=training_loop_params["window_len"],
        loader=test_loader,
        num_samples_to_predict=n_samples_test,
        device = cpu_device,
        probability=True
        )

        OutOfFoldPredictions[trial_level_indices] = preds_prob_fold

        validation_set_predictions+=preds_prob_validation

        test_set_predictions+=preds_prob_test


    
    validation_predictions=validation_set_predictions/n_splits
    test_predictions = test_set_predictions/n_splits

    return OutOfFoldPredictions , validation_predictions , test_predictions
        


        


