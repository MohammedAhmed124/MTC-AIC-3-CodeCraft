from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import polars as pl
import os


class Loader:
    def __init__(self, base_path: str = "/kaggle/input/mtcaic3"):
        self.base_path = base_path
        self.sample_rates = {"MI": 2250, "default": 1750}

        self.train_df = self._safe_read_csv(os.path.join(self.base_path, "train.csv"))
        self.validation_df = self._safe_read_csv(
            os.path.join(self.base_path, "validation.csv")
        )
        self.test_df = self._safe_read_csv(os.path.join(self.base_path, "test.csv"))

    def _safe_read_csv(self, file_path):
        try:
            return pl.read_csv(file_path)
        except Exception as e:
            raise e

    @lru_cache(maxsize=1000)
    def _load_eeg_file(self, file_path):
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
        return self.sample_rates.get(task, self.sample_rates["default"])

    def _construct_data_path(self, task, subject_id, trial_session) -> Path:
        return os.path.join(
            self.base_path,
            task,
            "train",
            str(subject_id),
            str(trial_session),
            "EEGdata.csv",
        )

    def get_trials_from_df(self, df):
        return df.to_dicts()

    def load_single_trial(self, trial):

        try:
            id_num = trial["id"]
            task = trial["task"]
            subject_id = trial["subject_id"]
            trial_session = trial["trial_session"]
            trial_num = trial["trial"]
            label = trial["label"]

            data_path = self._construct_data_path(task, subject_id, trial_session)

            eeg_data = self._load_eeg_file(data_path)

            sample_rate = self._get_sample_rate(task)

            start_idx = (trial_num - 1) * sample_rate

            trial_data = eeg_data.slice(start_idx, sample_rate)

            return id_num, subject_id, task, trial_data, label

        except Exception as e:
            raise e

    def load_data_parallel(self, trials, max_workers: int = 4):
        ids, subjects, tasks, datas, labels = [], [], [], [], []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_trial = {
                executor.submit(self.load_single_trial, trial): trial
                for trial in trials
            }

        for future in as_completed(future_to_trial):
            try:
                id_num, subject_id, task, trial_data, label = future.result()
                ids.append(id_num)
                subjects.append(subject_id)
                tasks.append(task)
                datas.append(trial_data)
                labels.append(label)

            except Exception as e:
                raise e

        return ids, subjects, tasks, datas, labels


loader = Loader()


train_trials = loader.get_trials_from_df(loader.train_df)
test_trials = loader.get_trials_from_df(loader.test_df)
validation_trials = loader.get_trials_from_df(loader.validation_df)


ids, sub, tasks, datas, labels = loader.load_data_parallel(train_trials)
