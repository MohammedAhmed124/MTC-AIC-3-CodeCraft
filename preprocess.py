from typing import List, Union, Tuple
from scipy import signal
import numpy as np
import logging
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalPreprocessor:
    order: int = 4
    fs: int = 250
    notch_freq: float = 50.0
    notch_q: float = 30.0
    bandpass_low: float = 6.0
    bandpass_high: float = 30.0
    window_size: int = 600
    window_stride: int = 600

    def __init__(self):
        self.nyquist = 0.5 * self.fs
        logger.info(
            f"Initialized preprocessor with fs={self.fs}Hz, nyquist={self.nyquist}Hz"
        )

    def _validate_frequency(self, freq: float, freq_name: str):
        if freq <= 0 or freq >= self.nyquist:
            raise ValueError(
                f"{freq_name} frequency {freq}Hz must be between 0 and {self.nyquist}Hz"
            )

        return freq / self.nyquist

    def create_bandpass_filter(
        self, low_freq: float = bandpass_low, high_freq: float = bandpass_high
    ) -> Tuple[np.ndarray, np.ndarray]:
        low_freq_norm = self._validate_frequency(low_freq, "Low freq.")
        high_freq_norm = self._validate_frequency(high_freq, "High freq")

        return signal.butter(
            self.order, [low_freq_norm, high_freq_norm], btype="bandpass"
        )

    def create_notch_filter(
        self, notch_freq: float = notch_freq, quality: float = notch_q
    ) -> Tuple[np.ndarray, np.ndarray]:
        return signal.iirnotch(notch_freq, quality, self.fs)

    def window(
        self, data, labels, subject_ids, size=window_size, stride=window_stride
    ) -> np.ndarray:
        _, n_channels, input_size = data.shape

        data_torch = torch.from_numpy(data)

        data_torch = data_torch.unfold(dimension=2, size=size, step=stride).permute(
            0, 2, 1, 3
        )

        data_torch = data_torch.reshape(-1, n_channels, size).cpu().numpy()

        num_windows = (input_size - size) // stride + 1

        labels = labels.repeat(num_windows)
        
        subject_ids = subject_ids.repeat(num_windows)
        
        return data_torch, labels, subject_ids

    def apply_filter(
        self, data: Union[np.ndarray, List[np.ndarray]], filter_type: str = "bandpass"
    ) -> Union[np.ndarray, List[np.ndarray]]:
        try:
            if filter_type.lower() == "bandpass":
                b, a = self.create_bandpass_filter()
                logger.info(f"Applying bandpass filter")
            elif filter_type.lower() == "notch":
                b, a = self.create_notch_filter()
                logger.info(f"Applying notch filter at {self.notch_freq}Hz")
            else:
                raise ValueError(
                    f"Unsupported filter type: {filter_type}. Use 'bandpass' or 'notch'"
                )

            if len(data) == 1:
                return np.array(signal.filtfilt(b, a, data))
            else:
                return np.stack([np.array(signal.filtfilt(b, a, signal_data)) for signal_data in data])

        except Exception as e:
            logger.error(f"Filter application failed: {str(e)}")
            raise


signal_preprocessor = SignalPreprocessor()

new_data = signal_preprocessor.apply_filter(datas, "notch")

new_data = signal_preprocessor.apply_filter(new_data, "bandpass")

new_data, labels, subject_ids = signal_preprocessor.window(new_data, labels, ids, 600, 600)
