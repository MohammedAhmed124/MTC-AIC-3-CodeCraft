from utils.extractors import extract_trial
from typing import Dict, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import logging
import shutil
import mne



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    """
    A class to process EEG competition data and convert it to MNE-compatible .fif files.
    """

    # Constants
    SAMPLING_FREQ = 250
    EEG_CHANNELS = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
    ACCELEROMETER_CHANNELS = ["AccX", "AccY", "AccZ"]
    GYROSCOPE_CHANNELS = ["Gyro1", "Gyro2", "Gyro3"]
    COLUMNS_TO_DROP = ["Time", "Counter", "Battery"]

    # Quality thresholds
    MIN_VALIDATION_THRESHOLD = 0.72
    MAX_GYRO_THRESHOLD = 6.0

    def __init__(
        self, competitions_data_directory: str, output_directory: str = "data_fif"
    ):
        """
        Initialize the EEG processor.

        Parameters
        ----------
        competitions_data_directory : str
            Path to the directory containing competition data
        output_directory : str
            Path to the output directory for .fif files
        """
        self.competitions_data_directory = Path(competitions_data_directory)
        self.output_directory = Path(output_directory)
        self.csv_cache: Dict[str, pd.DataFrame] = {}

        # Set MNE log level to reduce verbosity
        mne.set_log_level("ERROR")

        # Create output directory
        self.output_directory.mkdir(exist_ok=True)

        self._reset_output_directory()

    def _reset_output_directory(self) -> None:
        """Remove all subdirectories from the output directory."""
        for item in self.output_directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

    def _load_csv_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test CSV files."""
        try:
            train_labels = pd.read_csv(self.competitions_data_directory / "train.csv")
            validation_labels = pd.read_csv(
                self.competitions_data_directory / "validation.csv"
            )
            test_labels = pd.read_csv(self.competitions_data_directory / "test.csv")
            return train_labels, validation_labels, test_labels
        except FileNotFoundError as e:
            logger.error(f"CSV file not found: {e}")
            raise

    def _preprocess_csv_data(self, csv_file: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess CSV data by computing norms and selecting relevant columns.

        Parameters
        ----------
        csv_file : pd.DataFrame
            Raw CSV data

        Returns
        -------
        pd.DataFrame
            Preprocessed CSV data
        """

        csv_file = csv_file.drop(self.COLUMNS_TO_DROP, errors="ignore")

        csv_file["Acc_norm"] = np.linalg.norm(
            csv_file[self.ACCELEROMETER_CHANNELS], axis=1
        )
        csv_file["gyro_norm"] = np.linalg.norm(
            csv_file[self.GYROSCOPE_CHANNELS], axis=1
        )

        csv_file = csv_file.drop(
            columns=self.ACCELEROMETER_CHANNELS + self.GYROSCOPE_CHANNELS,
            errors="ignore",
        )

        final_columns = self.EEG_CHANNELS + ["Acc_norm", "gyro_norm", "Validation"]
        return csv_file[final_columns]

    def _load_and_cache_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load CSV file with caching to avoid repeated file I/O.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file

        Returns
        -------
        pd.DataFrame
            Preprocessed CSV data
        """
        file_path_str = str(file_path)

        if file_path_str not in self.csv_cache:
            try:
                csv_file = pd.read_csv(file_path)
                csv_file = self._preprocess_csv_data(csv_file)
                self.csv_cache[file_path_str] = csv_file
                logger.debug(f"Loaded and cached: {file_path}")
            except FileNotFoundError:
                logger.error(f"EEG data file not found: {file_path}")
                raise

        return self.csv_cache[file_path_str]

    def _create_mne_raw(
        self, trial_data: pd.DataFrame, trial_info: Dict[str, Any]
    ) -> mne.io.RawArray:
        """
        Create MNE Raw object from trial data.

        Parameters
        ----------
        trial_data : pd.DataFrame
            Trial-specific EEG data
        trial_info : Dict[str, Any]
            Trial metadata

        Returns
        -------
        mne.io.RawArray
            MNE Raw object
        """
        ch_names = list(trial_data.columns)
        ch_types = ["eeg"] * len(self.EEG_CHANNELS) + ["misc"] * 2 + ["stim"]

        info = mne.create_info(
            ch_names=ch_names, sfreq=self.SAMPLING_FREQ, ch_types=ch_types
        )

        raw = mne.io.RawArray(trial_data.to_numpy().T, info)

        raw.info["subject_info"] = {
            "id": int(trial_info["id"]),
            "his_id": str(
                (
                    trial_info["val_mean"],
                    trial_info["acc_mean"],
                    trial_info["gyro_mean"],
                )
            ),
            "sex": 0,
            "birthday": None,
        }

        # Add label as description (if available)
        if trial_info.get("label") is not None:
            raw.info["description"] = trial_info["label"]

        return raw

    def _add_bad_annotations(self, raw: mne.io.RawArray) -> None:
        """
        Add annotations for bad segments based on validation flags.

        Parameters
        ----------
        raw : mne.io.RawArray
            MNE Raw object to annotate
        """
        validation_data = raw.copy().pick_channels(["Validation"]).get_data()[0]

        # Find bad segments (where validation == 0)
        bad_mask = validation_data == 0
        changes = np.diff(bad_mask.astype(int))

        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if bad_mask[0]:
            starts = np.insert(starts, 0, 0)
        if bad_mask[-1]:
            ends = np.append(ends, len(bad_mask))

        if len(starts) > 0:
            onsets = starts / self.SAMPLING_FREQ
            durations = (ends - starts) / self.SAMPLING_FREQ
            annotations = mne.Annotations(
                onset=onsets, duration=durations, description=["BAD"] * len(onsets)
            )
            raw.set_annotations(annotations)

    def _should_skip_trial(
        self, val_mean: float, gyro_mean: float, data_type: str
    ) -> bool:
        """
        Determine if a trial should be skipped based on quality metrics.

        Parameters
        ----------
        val_mean : float
            Mean validation score
        gyro_mean : float
            Mean gyroscope norm
        data_type : str
            Type of data (train, validation, test)

        Returns
        -------
        bool
            True if trial should be skipped
        """
        if data_type in ["train", "validation"]:
            return (
                val_mean <= self.MIN_VALIDATION_THRESHOLD
                or gyro_mean > self.MAX_GYRO_THRESHOLD
            )
        return False

    def save_data_as_fif(self, df: pd.DataFrame, data_type: str) -> None:
        """
        Process EEG trials and save them as .fif files.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing trial metadata
        data_type : str
            One of "train", "validation", or "test"
        """
        # Create output directories
        path_mi = self.output_directory / data_type / "MI"
        path_ssvep = self.output_directory / data_type / "SSVEP"
        path_mi.mkdir(parents=True, exist_ok=True)
        path_ssvep.mkdir(parents=True, exist_ok=True)

        skipped_trials = 0
        processed_trials = 0

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {data_type}"
        ):
            try:
                # Extract trial information
                trial_info = {
                    "id": row.id,
                    "subject_id": row.subject_id,
                    "task": row.task,
                    "trial_session": row.trial_session,
                    "trial": row.trial,
                    "label": row.label if data_type != "test" else None,
                }

                # Construct file path
                file_path = (
                    self.competitions_data_directory
                    / trial_info["task"]
                    / data_type
                    / trial_info["subject_id"]
                    / str(trial_info["trial_session"])
                    / "EEGdata.csv"
                )

                # Load and process CSV data
                csv_file = self._load_and_cache_csv(file_path)

                # Extract trial data
                raw_data = extract_trial(
                    trial_info["trial"] - 1, csv_file, task=trial_info["task"]
                )

                # Calculate quality metrics
                val_mean = float(raw_data["Validation"].mean())
                acc_mean = float(raw_data["Acc_norm"].mean())
                gyro_mean = float(raw_data["gyro_norm"].mean())

                # Skip low-quality trials for train/validation
                if self._should_skip_trial(val_mean, gyro_mean, data_type):
                    skipped_trials += 1
                    continue

                # Add quality metrics to trial info
                trial_info.update(
                    {"val_mean": val_mean, "acc_mean": acc_mean, "gyro_mean": gyro_mean}
                )

                # Create MNE Raw object
                raw = self._create_mne_raw(raw_data, trial_info)

                # Add bad segment annotations
                self._add_bad_annotations(raw)

                # Save as .fif file
                output_path = path_mi if trial_info["task"] == "MI" else path_ssvep
                filename = (
                    f"{trial_info['task']}_{trial_info['subject_id']}_"
                    f"{trial_info['trial_session']}_{trial_info['trial']}.fif"
                )
                save_path = output_path / filename

                raw.save(save_path, overwrite=True)
                processed_trials += 1

            except Exception as e:
                logger.error(f"Error processing trial {row.id}: {e}")
                continue

        logger.info(
            f"{data_type}: Processed {processed_trials} trials, skipped {skipped_trials} trials"
        )

    def process_all_data(self) -> None:
        """Process all train, validation, and test data."""
        try:
            # Load CSV labels
            train_labels, validation_labels, test_labels = self._load_csv_labels()

            # Sort by task and trial_session for better caching efficiency
            sort_columns = ["task", "trial_session"]

            # Process each dataset
            self.save_data_as_fif(train_labels.sort_values(by=sort_columns), "train")
            self.save_data_as_fif(
                validation_labels.sort_values(by=sort_columns), "validation"
            )
            self.save_data_as_fif(test_labels.sort_values(by=sort_columns), "test")

            logger.info("All data processing completed successfully!")

        finally:
            # Reset MNE log level
            mne.set_log_level("INFO")


def main():
    """Main function to run the EEG processor."""
    parser = argparse.ArgumentParser(
        description="Process EEG competition data to .fif files"
    )

    script_path = Path(__file__).parent
    default_data_path = script_path / "data"

    parser.add_argument(
        "--competitions_data_directory",
        type=str,
        default=str(default_data_path),
        help="Path to the directory containing the competition data",
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        default="data_fif",
        help="Path to the output directory for .fif files",
    )

    args = parser.parse_args()

    # Create and run processor
    processor = EEGProcessor(
        competitions_data_directory=args.competitions_data_directory,
        output_directory=args.output_directory,
    )

    processor.process_all_data()


if __name__ == "__main__":
    main()
