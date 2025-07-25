from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import polars as pl
import os
import sys
import numpy as np

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "data")
class Loader:

    """
    EEG Trial Data Loader for MI and SSVEP Tasks.

    This class is responsible for:
    - Loading trial-level EEG data from .csv files.
    - applies bad trial filtering.
    - Selecting relevant EEG and motion channels depending on the task type (MI or SSVEP).
    - Structuring the data into (n_trials, n_channels, n_timepoints) format.
    - Extracting labels, subject IDs, and validation flags.

    Attributes:
        dataset_type (str): One of {"train", "validation", "test"}.
        base_path (str): Root directory containing data and metadata CSVs.
        label_mapping (dict): Optional mapping from original labels to integers.
        sample_rates (dict): Sampling rates per task (MI vs. default).
        loaded_df (pl.DataFrame): Metadata DataFrame for the chosen split.
    """

    def __init__(self, base_path: str = base_path, label_mapping: dict = None,dataset_type=None):

        """
        Initializes the Loader.

        Args:
            base_path (str): Path to the `data/` directory.
            label_mapping (dict, optional): Mapping from text labels to ints.
            dataset_type (str): Which split to load ("train", "validation", "test").

        Raises:
            AssertionError: If dataset_type is not one of the allowed values.
        """
        assert dataset_type in ["train" , "validation" , "test"] , "dataset_type should be specified and in [train, validation , test] "
        self.dataset_type = dataset_type
        self.base_path = base_path
        self.label_mapping = label_mapping
        self.sample_rates = {"MI": 2250, "default": 1750}

        if dataset_type=="train":
            self.loaded_df = self._safe_read_csv(os.path.join(self.base_path, "train.csv"))
        elif dataset_type=="validation":
            self.loaded_df = self._safe_read_csv(
                os.path.join(self.base_path, "validation.csv")
            )
        else:
            self.loaded_df = self._safe_read_csv(os.path.join(self.base_path, "test.csv"))

    def _safe_read_csv(self, file_path):
        """
        Reads a CSV file into a Polars DataFrame, raising on error.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pl.DataFrame: Loaded data.
        """
        try:
            return pl.read_csv(file_path)
        except Exception as e:
            raise e

    @lru_cache(maxsize=1000)
    def _load_eeg_file(self, file_path):
        """
        Caches and loads a single EEG CSV file via Polars.

        Args:
            file_path (str): Path to the trial's EEGdata.csv.

        Returns:
            pl.DataFrame: Raw EEG and sensor data for the trial.
        """
        try:
            return pl.read_csv(file_path)
        except Exception as e:
            raise e

    def _determine_dataset_type(self, id_num):
        if id_num <= 4800:
            return "train"
        elif id_num <= 4900:
            return "validation"
        else:
            return "test"

    def _get_sample_rate(self, task):
        """
        Returns the sampling rate for the given task.

        Args:
            task (str): "MI" or other (default).

        Returns:
            int: Sampling rate in Hz.
        """
        return self.sample_rates.get(task, self.sample_rates["default"])

    def _construct_data_path(self, task, subject_id, trial_session) -> Path:
        """
        Builds the file path to a trial's EEGdata.csv.

        Args:
            task (str): "MI" or "SSVEP".
            subject_id (str): Subject identifier.
            trial_session (str): Session number or name.

        Returns:
            str: Full path to the EEGdata.csv file.
        """
        return os.path.join(
            self.base_path,
            task,
            self.dataset_type,
            str(subject_id),
            str(trial_session),
            "EEGdata.csv",
        )

    def get_trials_from_df(self, df,task_type=None):
        """
        Filters the metadata DataFrame for a specific task.

        Args:
            df (pl.DataFrame): Metadata DataFrame (train/validation/test).
            task_type (str): Either "MI" or "SSVEP".

        Returns:
            List[dict]: List of row-dictionaries matching the task.
        """
        assert task_type in ["SSVEP" , "MI"], "task type should be specified and be either SSVEP or MI"
        return [
            dict_items for dict_items in df.to_dicts() 
            if dict_items["task"]==task_type
        ]

    def load_single_trial(self, trial,quality_filter= None):

        """
        Loads and processes one trial based on metadata.

        Steps:
          1. Reads trial metadata (id, task, subject, session, label).
          2. Loads CSV into Polars DataFrame.
          3. Slices the window for the given trial number.
          4. Computes L2 norms for accelerometer and gyroscope.
          5. Selects task-specific EEG channels + shared features.
          6. Optionally filters out low-quality trials.

        Args:
            trial (dict): Metadata dict with keys ["id","task","subject_id","trial_session","trial","label"].
            quality_filter (tuple, optional): (min_val, max_gyro, max_acc) thresholds.

        Returns:
            tuple or None:
                (id_num, subject_id, task, trial_data, label) on success,
                or None if filtered out.
        """

        try:
            id_num = trial["id"]
            task = trial["task"]
            subject_id = trial["subject_id"]
            trial_session = trial["trial_session"]
            trial_num = trial["trial"]
            if self.dataset_type == "test":
                label = trial.get("label" , None)
            else:
                label = trial["label"]

            data_path = self._construct_data_path(task, subject_id, trial_session)

            eeg_data = self._load_eeg_file(data_path)

            sample_rate = self._get_sample_rate(task)

            start_idx = (trial_num - 1) * sample_rate

            trial_data = eeg_data.slice(start_idx, sample_rate)

            shared_cols = ['Acc_norm', 'gyro_norm', 'Validation']

            if task == "MI":
                eeg_cols = ['C3', 'C4', 'CZ', 'FZ']
            elif task == "SSVEP":
                eeg_cols = ['OZ', 'PO7', 'PO8', 'PZ']
            else:
                raise ValueError(f"{task} is not a valid type")

            cols_to_pick = eeg_cols + shared_cols
            polars_cols = [pl.col(col) for col in cols_to_pick]
            

            trial_data = trial_data.with_columns(
                (pl.col("AccX").pow(2) + pl.col("AccY").pow(2) + pl.col("AccZ").pow(2)).sqrt().alias("Acc_norm"),
                (pl.col("Gyro1").pow(2) + pl.col("Gyro2").pow(2) + pl.col("Gyro3").pow(2)).sqrt().alias("gyro_norm"),
            ).select(
                polars_cols
            )

            if quality_filter:
                #ignore bad quality trials
                #the quality filter paramater accepts a tuple (minimum_accepted_validation, maximum_accepted_gyro, maximum accepted acc)
                #if the paramater is passed 'None' Filtering does not occur
                #if one of the tuples elements is passed None, Filtering does not occure on the corresponding metrics value
                ####----Don't use this for Testing data----####
                assert len(quality_filter) == 3, (
                    "quality_filter must be a 3-tuple: "
                    "(minimum_accepted_validation, maximum_accepted_gyro, maximum_accepted_acc)"
                )

                min_val, max_gyro, max_acc = quality_filter

                means = trial_data.select([
                    pl.col("Validation").mean().alias("mean_val"),
                    pl.col("gyro_norm").mean().alias("mean_gyro"),
                    pl.col("Acc_norm").mean().alias("mean_acc")
                ])

                mean_val = means["mean_val"][0]
                mean_gyro = means["mean_gyro"][0]
                mean_acc = means["mean_acc"][0]

                if min_val is not None and mean_val < min_val:
                    return None
                if max_gyro is not None and mean_gyro > max_gyro:
                    return None
                if max_acc is not None and mean_acc > max_acc:
                    return None
                
                        

            return id_num, subject_id, task, trial_data, label

        except Exception as e:
            raise e

    def load_data_parallel(self, trials,quality_filter=None,return_numpy=False, max_workers: int = 4):
        """
        Parallel loading of multiple trials.

        Args:
            trials (list[dict]): List of metadata dicts from get_trials_from_df().
            quality_filter (tuple, optional): Thresholds for filtering trials.
            return_numpy (bool): If True, convert data and labels to NumPy arrays.
            max_workers (int): Number of threads for parallel loading.

        Returns:
            tuple:
                ids, subjects, tasks, datas, labels
                - ids (list or np.ndarray)
                - subjects (list or np.ndarray)
                - tasks (list or np.ndarray)
                - datas (list of pl.DataFrame or np.ndarray)
                - labels (list or np.ndarray)
        """
        ids, subjects, tasks, datas, labels = [], [], [], [], []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.load_single_trial, trial, quality_filter) for trial in trials]

        for trial, future in zip(trials, futures):
            result = future.result()
            if not result:
                continue
            try:
                id_num, subject_id, task, trial_data, label = result
                ids.append(id_num)
                subjects.append(int(subject_id.lstrip("S")) - 1)
                tasks.append(task)
                datas.append(trial_data)
                labels.append(label)
            except Exception as e:
                raise e

            
            
        assert len(ids) == len(subjects)== len(tasks) == len(datas) == len(labels), "Mismatch in returns lengths"
        if return_numpy:
            if self.label_mapping:
                if self.dataset_type != "test":
                    labels = np.asarray([
                        self.label_mapping[x] for x in labels
                        ])
                else:
                    labels = np.asarray(labels)

            else :
                labels = np.asarray(labels)

            subjects = np.asarray(subjects)
            tasks = np.asarray(tasks)
            ids = np.asarray(ids)
            datas = np.asarray([
                trial_data.to_numpy().T for trial_data in datas
            ])
            return ids, subjects, tasks, datas, labels
        else:
            return ids, subjects, tasks, datas, labels


