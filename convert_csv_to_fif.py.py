import os
from tqdm import tqdm
import mne
import pandas as pd
import numpy as np
from utils.extractors import extract_trial
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--competitions_data_directory",
    type=str,
    default="data",
    help="Path to the directory containing the competition data"
)
args = parser.parse_args()

competitions_data_directory = args.competitions_data_directory


# ------------------------------------------------------------------------------
# Utility: remove all subdirectories from a given path (used to reset output dir)
# ------------------------------------------------------------------------------

os.makedirs("data_fif", exist_ok=True)

def remove_all_subdirectories(path):
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)

remove_all_subdirectories("data_fif")




# ------------------------------------------------------------------------------
# Read label CSVs which contain metadata for each EEG trial
# ------------------------------------------------------------------------------

train_csv_path = os.path.join(competitions_data_directory,"train.csv")
val_csv_path = os.path.join(competitions_data_directory,"validation.csv")
test_csv_path = os.path.join(competitions_data_directory,"test.csv")

train_labels = pd.read_csv(train_csv_path)
validation_labels = pd.read_csv(val_csv_path)
test_labels = pd.read_csv(test_csv_path)



# ------------------------------------------------------------------------------
# Main function: save each EEG trial as a clean .fif file
# ------------------------------------------------------------------------------
def save_data_as_fif(df,base_path="data_fif/", data_type="train"):

    """
    Processes EEG trials from a competition dataset and saves them as clean .fif files 
    (used by MNE) with relevant metadata and quality annotations.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing trial metadata. Each row describes one EEG trial and 
        includes the following fields:
        - id : Unique trial ID
        - subject_id : Identifier of the subject who performed the trial
        - task : Either "MI" (Motor Imagery) or "SSVEP" (Steady-State Visual Evoked Potential)
        - trial_session : Session index in which the trial was recorded
        - trial : Trial index within the session
        - label : (train/val only) Ground truth label for the trial

    base_path : str, optional
        Directory where the `.fif` files will be saved. It will create a subfolder per 
        task and data split (e.g., `data_fif/train/MI/...`).

    data_type : str, optional
        One of "train", "validation", or "test". Used to:
        - Skip low-quality trials during training and validation.
        - Avoid assigning labels for test data.
        - Determine the file path structure.

    Workflow Summary
    ----------------
    - For each trial:
        1. Loads the associated full EEG session CSV.
        2. Computes auxiliary features:
           - Acc_norm: L2 norm of [AccX, AccY, AccZ]
           - gyro_norm: L2 norm of [Gyro1, Gyro2, Gyro3]
        3. Extracts the trial-specific segment using `extract_trial()`.
        4. Constructs an MNE `Raw` object with channel names and types.
        5. Stores relevant metadata:
           - Trial ID, quality metrics, and (optionally) the label
        6. Annotates bad signal segments where Validation == 0.
        7. Skips trials with poor quality (low `Validation` or high `gyro_norm`) in train/val.
        8. Saves the result as a `.fif` file to disk.

    Notes
    -----
    - This function uses caching to avoid re-loading the same CSV for trials from the 
      same session.
    - Annotations are useful for downstream denoising during later model training.
    - The final `.fif` files are MNE-compatible and contain these final selected channels: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'Acc_norm','gyro_norm','Validation'].

    Raises
    ------
    FileNotFoundError
        If the EEG data CSV for a trial is missing.

    ValueError
        If trial metadata is incomplete or malformed.
    """
    mne.set_log_level("ERROR")
    path_mi= os.path.join(base_path,data_type,"MI")
    path_ssvep = os.path.join(base_path,data_type,"SSVEP")
    os.makedirs(path_mi, exist_ok=True)
    os.makedirs(path_ssvep, exist_ok=True)
    cache = [(0,0)]
    for row_id in tqdm(range(df.shape[0]) , total = len(df)):
        row = df.loc[row_id]
        id_ = row.id
        subject_id = row.subject_id
        task = row.task
        trial_session = row.trial_session
        trial =row.trial
        label = row.label if not data_type=="test" else None
        file_path = os.path.join(
            competitions_data_directory,
            task,
            data_type,
            subject_id,
            str(trial_session),
            "EEGdata.csv"
            )
        
        if file_path == cache[0][1]:
            csv_file = cache[0][0]
        else:
            csv_file = pd.read_csv(file_path).drop(["Time" , "Counter" ,"Battery"], axis = 1)
            csv_file["Acc_norm"] = np.linalg.norm(csv_file[['AccX','AccY', 'AccZ']],axis =1 )
            csv_file["gyro_norm"] = np.linalg.norm(csv_file[['Gyro1','Gyro2', 'Gyro3']],axis =1 )
            csv_file = csv_file.drop(columns= ['AccX','AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3'])
            csv_file = csv_file[['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'Acc_norm','gyro_norm','Validation']]
            cache[0] = (csv_file , file_path)
        
        ch_names = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'Acc_norm','gyro_norm','Validation']
        ch_types = ["eeg"]*8 + ['misc'] * 2+ ['stim']


        raw_data = extract_trial((trial-1), csv_file, task =task )

        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)
        raw = mne.io.RawArray(raw_data.to_numpy().T, info)


        val_mean = float(raw_data["Validation"].mean())
        acc_mean = float(raw_data["Acc_norm"].mean())
        gyro_mean = float(raw_data["gyro_norm"].mean())


        if (val_mean<=0.72 or gyro_mean>6) and data_type in ["train","validation"]:
            continue #skip this trial 


        raw.info['subject_info'] = {
            'id': int(id_),                     
            'his_id': str((val_mean,acc_mean , gyro_mean)),          

            'sex': 0,                         
            'birthday': None,           
        }
        raw.info['description'] = label

        flag_data = raw.copy().pick_channels(["Validation"]).get_data()[0]

        flag_data = raw.copy().pick_channels(["Validation"]).get_data()[0]
        sfreq = 250

        # Invert if 1 means good and 0 means bad, because annotations mark bad segments
        bad_mask = (flag_data == 0)

        # Find transitions in the mask (diff non-zero means start or end of a bad segment)
        changes = np.diff(bad_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases (start or end of data is bad)
        if bad_mask[0]:
            starts = np.insert(starts, 0, 0)
        if bad_mask[-1]:
            ends = np.append(ends, len(bad_mask))

        onsets = starts / sfreq
        durations = (ends - starts) / sfreq

        annotations = mne.Annotations(onset=onsets, duration=durations, description=['BAD']*len(onsets))
        raw.set_annotations(annotations)

        # raw.drop_channels(['Validation'])
        save_path = os.path.join(path_mi if task=="MI" else path_ssvep,f"{task}_{subject_id}_{trial_session}_{trial}_.fif")

        raw.save(save_path, overwrite=True)
    mne.set_log_level("INFO")
                


save_data_as_fif(train_labels.sort_values(by=["task","trial_session"])
                 ,base_path="data_fif/"
                 , data_type="train")

save_data_as_fif(validation_labels.sort_values(by=["task","trial_session"])
                 ,base_path="data_fif/"
                 , data_type="validation")


save_data_as_fif(test_labels.sort_values(by=["task","trial_session"])
                 ,base_path="data_fif/"
                 , data_type="test")