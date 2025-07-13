from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import lru_cache
from pathlib import Path
import polars as pl
import numpy as np
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Loader:
    SAMPLE_RATES = {"MI": 2250, "SSVEP": 1750}
    EEG_COLUMNS = {"MI": ["C3", "C4", "CZ", "FZ"], "SSVEP": ["OZ", "PO7", "PO8", "PZ"]}
    SHARED_COLUMNS = ["Acc_norm", "gyro_norm", "Validation"]
    VALID_TASKS = {"SSVEP", "MI"}

    def __init__(self, base_path: Union[str, Path]) -> None:
        """
        Initializes the Loader with the base path to the dataset.

        Args:
            base_path (Union[str, Path]): The root directory of the dataset,
                                            expected to contain 'train.csv',
                                            'validation.csv', 'test.csv',
                                            and subdirectories for task-specific
                                            EEG data.
        Raises:
            FileNotFoundError: If any of the manifest CSV files are not found.
            Exception: For other errors during CSV file reading.
        """
        self.base_path = Path(base_path)
        self.train_df = self._safe_read_csv(os.path.join(self.base_path, "train.csv"))
        self.validation_df = self._safe_read_csv(
            os.path.join(self.base_path, "validation.csv")
        )
        self.test_df = self._safe_read_csv(os.path.join(self.base_path, "test.csv"))

    def _safe_read_csv(self, file_path: Path) -> pl.DataFrame:
        """
        Safely reads a CSV file into a Polars DataFrame.

        Args:
            file_path (Path): The full path to the CSV file.

        Returns:
            pl.DataFrame: The loaded Polars DataFrame.

        Raises:
            Exception: If an error occurs during file reading.
        """
        try:
            return pl.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _load_eeg_file(self, file_path: Path) -> pl.DataFrame:
        """
        Loads an EEG data CSV file into a Polars DataFrame, with caching.

        This method uses `functools.lru_cache` to cache previously loaded
        EEG files, improving performance for repeated access to the same file.

        Args:
            file_path (Path): The full path to the EEG data CSV file.

        Returns:
            pl.DataFrame: The loaded Polars DataFrame containing EEG data.

        Raises:
            Exception: If an error occurs during file loading.
        """
        try:
            return pl.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading EEG file {file_path}: {e}")
            raise

    def _determine_dataset_type(self, id_num: int) -> str:
        """
        Determines the dataset type (train, validation, or test) based on the ID number.

        Args:
            id_num (int): The ID number of the trial.

        Returns:
            str: "train", "validation", or "test".
        """
        if id_num <= 4800:
            return "train"
        elif id_num <= 4900:
            return "validation"
        else:
            return "test"

    def _get_sample_rate(self, task: str) -> int:
        """
        Retrieves the sample rate for a given task type.

        If the task is not explicitly defined in SAMPLE_RATES, it defaults to 'SSVEP'.

        Args:
            task (str): The type of the task (e.g., "MI", "SSVEP").

        Returns:
            int: The sample rate for the specified task.
        """
        return self.SAMPLE_RATES.get(task, self.SAMPLE_RATES["SSVEP"])

    def _construct_data_path(
        self, task: str, subject_id: str, trial_session: int
    ) -> Path:
        """
        Constructs the full file path to an EEG data CSV file.

        Args:
            task (str): The task type (e.g., "MI", "SSVEP").
            subject_id (str): The subject ID (e.g., "S1").
            trial_session (int): The trial session number.

        Returns:
            Path: The constructed path to the EEGdata.csv file.
        """
        return os.path.join(
            self.base_path,
            task,
            "train",
            str(subject_id),
            str(trial_session),
            "EEGdata.csv",
        )

    def get_trials_from_df(self, df: pl.DataFrame, task: str) -> List[Dict[str, Any]]:
        """
        Filters a DataFrame to get trials for a specific task and converts them to a list of dictionaries.

        Args:
            df (pl.DataFrame): The input DataFrame (e.g., train_df, validation_df, test_df).
            task (str): The task type to filter by (e.g., "MI", "SSVEP").

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                    represents a trial and its metadata.
        """
        filtered_df = df.filter(pl.col("task") == task)
        return filtered_df.to_dicts()

    def _compute_derived_columns(self, trial_data: pl.DataFrame) -> pl.DataFrame:
        """
        Computes 'Acc_norm' (accelerometer norm) and 'gyro_norm' (gyroscope norm)
        columns and adds them to the trial data DataFrame.

        Args:
            trial_data (pl.DataFrame): The Polars DataFrame containing raw sensor data.

        Returns:
            pl.DataFrame: The DataFrame with 'Acc_norm' and 'gyro_norm' columns added.
        """
        return trial_data.with_columns(
            [
                (pl.col("AccX").pow(2) + pl.col("AccY").pow(2) + pl.col("AccZ").pow(2))
                .sqrt()
                .alias("Acc_norm"),
                (
                    pl.col("Gyro1").pow(2)
                    + pl.col("Gyro2").pow(2)
                    + pl.col("Gyro3").pow(2)
                )
                .sqrt()
                .alias("gyro_norm"),
            ]
        )

    def _apply_quality_filter(
        self,
        trial_data: pl.DataFrame,
        quality_filter: Tuple[Optional[float], Optional[float], Optional[float]],
    ) -> bool:
        """
        Applies quality filters based on mean 'Validation', 'gyro_norm', and 'Acc_norm'.

        Args:
            trial_data (pl.DataFrame): The Polars DataFrame for a single trial,
                                        expected to contain 'Validation', 'gyro_norm',
                                        and 'Acc_norm' columns.
            quality_filter (Tuple[Optional[float], Optional[float], Optional[float]]):
                A 3-tuple representing the quality filter criteria:
                (minimum_accepted_validation, maximum_accepted_gyro_norm, maximum_accepted_acc_norm).
                None for any value means no filter is applied for that metric.

        Returns:
            bool: True if the trial passes all specified quality filters, False otherwise.
        """

        min_val, max_gyro, max_acc = quality_filter

        means = trial_data.select(
            [
                pl.col("Validation").mean().alias("mean_val"),
                pl.col("gyro_norm").mean().alias("mean_gyro"),
                pl.col("Acc_norm").mean().alias("mean_acc"),
            ]
        ).row(0)

        mean_val, mean_gyro, mean_acc = means

        if min_val is not None and mean_val < min_val:
            return False
        if max_gyro is not None and mean_gyro > max_gyro:
            return False
        if max_acc is not None and mean_acc > max_acc:
            return False

        return True

    def load_single_trial(
        self,
        trial: Dict[str, Any],
        quality_filter: Optional[
            Tuple[Optional[float], Optional[float], Optional[float]]
        ] = None,
    ) -> Optional[Tuple[int, int, str, pl.DataFrame, Any]]:
        """
        Loads and preprocesses data for a single trial.

        This method extracts a specific segment of EEG and sensor data based on
        trial information, computes derived sensor norms, selects relevant columns,
        and optionally applies quality filtering.

        Args:
            trial (Dict[str, Any]): A dictionary containing metadata for a single trial,
                                    expected to have keys like 'id', 'task', 'subject_id',
                                    'trial_session', 'trial', and 'label'.
            quality_filter (Optional[Tuple[Optional[float], Optional[float], Optional[float]]]):
                An optional 3-tuple for quality filtering:
                (minimum_accepted_validation, maximum_accepted_gyro_norm, maximum_accepted_acc_norm).
                If provided, trials not meeting these criteria will return None.

        Returns:
            Optional[Tuple[int, int, str, pl.DataFrame, Any]]: A tuple containing:
                - id_num (int): The trial ID.
                - subject_id_int (int): The subject ID as an integer (0-indexed).
                - task (str): The task type.
                - trial_data (pl.DataFrame): The preprocessed Polars DataFrame for the trial.
                - label (Any): The label associated with the trial.
                Returns None if the trial fails the quality filter or if an invalid task type is encountered.

        Raises:
            ValueError: If an invalid task type is provided or if `quality_filter`
                        is not a 3-tuple when provided.
            Exception: For other errors during data loading or processing.
        """
        try:
            id_num = trial["id"]
            task = trial["task"]
            subject_id = trial["subject_id"]
            trial_session = trial["trial_session"]
            trial_num = trial["trial"]
            label = trial["label"]

            if task not in self.VALID_TASKS:
                raise ValueError(f"Invalid task type: {task}")

            data_path = self._construct_data_path(task, subject_id, trial_session)

            eeg_data = self._load_eeg_file(data_path)

            sample_rate = self._get_sample_rate(task)
            start_idx = (trial_num - 1) * sample_rate
            trial_data = eeg_data.slice(start_idx, sample_rate)

            trial_data = self._compute_derived_columns(trial_data)

            eeg_col = self.EEG_COLUMNS[task]
            cols_to_select = eeg_col + self.SHARED_COLUMNS

            trial_data = trial_data.select(cols_to_select)

            if quality_filter is not None:
                if len(quality_filter) != 3:
                    raise ValueError(
                        "quality_filter must be a 3-tuple: "
                        "(minimum_accepted_validation, maximum_accepted_gyro, maximum_accepted_acc)"
                    )

                if not self._apply_quality_filter(trial_data, quality_filter):
                    return None

            subject_id_int = int(subject_id.lstrip("S")) - 1

            return id_num, subject_id_int, task, trial_data, label

        except Exception as e:
            logger.error(f"Error loading trial {trial.get('id', 'unknown')}: {e}")
            raise

    def load_data_parallel(
        self,
        trials: List[Dict[str, Any]],
        quality_filter: Optional[
            Tuple[Optional[float], Optional[float], Optional[float]]
        ] = None,
        max_workers: int = 4,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Loads multiple trials in parallel using a ThreadPoolExecutor.

        This method efficiently loads and preprocesses a list of trials,
        applying specified quality filters. Results are gathered and returned
        as NumPy arrays for consistency.

        Args:
            trials (List[Dict[str, Any]]): A list of trial dictionaries,
                                            each containing metadata for a trial.
            quality_filter (Optional[Tuple[Optional[float], Optional[float], Optional[float]]]):
                An optional 3-tuple for quality filtering, passed to `load_single_trial`.
                (minimum_accepted_validation, maximum_accepted_gyro_norm, maximum_accepted_acc_norm).
            max_workers (int): The maximum number of threads to use for parallel loading.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing five NumPy arrays:
                - ids (np.ndarray): Array of trial IDs.
                - subjects (np.ndarray): Array of subject IDs (0-indexed integers).
                - tasks (np.ndarray): Array of task types (strings).
                - datas (np.ndarray): Array of preprocessed trial data, where each element
                                        is a NumPy array representing the data for a trial,
                                        transposed (channels x samples).
                - labels (np.ndarray): Array of trial labels.
        """
        ids, subjects, tasks, datas, labels = [], [], [], [], []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_trial = {
                executor.submit(self.load_single_trial, trial, quality_filter): trial
                for trial in trials
            }

        for future in as_completed(future_to_trial):
            trial = future_to_trial[future]

            try:
                result = future.result()
                if result is None:
                    continue

                id_num, subject_id, task, trial_data, label = result
                ids.append(id_num)
                subjects.append(subject_id)
                tasks.append(task)
                datas.append(trial_data)
                labels.append(label)

            except Exception as e:
                logger.error(f"Failed to load trial {trial.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Successfully loaded {len(ids)} trials")

        ids = np.array(ids)
        subjects = np.array(subjects)
        tasks = np.array(tasks)
        labels = np.array(labels)
        datas = np.array([trial_data.to_numpy().T for trial_data in datas])

        return ids, subjects, tasks, datas, labels


loader = Loader()

train_trials_mi = loader.get_trials_from_df(loader.train_df, "MI")

ids, sub, tasks, datas, labels = loader.load_data_parallel(
    train_trials_mi, max_workers=8
)
