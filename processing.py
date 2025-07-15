

import numpy as np
from typing import Union, List , Tuple
import logging
import mne
import torch 

logger = logging.getLogger(__name__)

class SignalPreprocessor:
    def __init__(
            self,
            fs: int = 250,
            notch_freq: Union[float, List[float]] = 50.0,
            notch_width: float = 1.0,
            bandpass_low: float = 6.0,
            bandpass_high: float = 30.0,
            n_cols_to_filter: int = 4,
            window_size : int = 600,
            window_stride : int = 600,
            idx_to_ignore_normalization : int =-1,
            ):
        self.fs = fs
        self.notch_freq = notch_freq
        self.notch_width = notch_width
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.n_cols_to_filter = n_cols_to_filter
        self.window_size = window_size
        self.window_stride = window_stride
        self.idx_to_ignore_normalization = idx_to_ignore_normalization

        logger.info(
            f"Initialized preprocessor with fs={self.fs}Hz, notch={self.notch_freq}Hz, "
            f"bandpass=({self.bandpass_low}–{self.bandpass_high}Hz), "
            f"FIlters will be applied to the first {n_cols_to_filter} channels only, "
            f"data will be windowed with window size = {window_size} and stride = {window_stride}"
        )

    def _window_data(
            self,
            data,
            labels,
            subject_ids,
            window_size,
            window_stride,
    )-> np.ndarray: 
        _ , n_channels , input_size = data.shape
        data_torch =torch.from_numpy(data)
        data_torch = data_torch.unfold(dimension=2, size=window_size, step=window_stride).permute(0, 2, 1, 3)
        data_torch = data_torch.reshape(-1 , n_channels , window_size).cpu().numpy()

        num_windows = (input_size - window_size) // window_stride + 1

        labels = labels.repeat(num_windows)
        subject_ids = subject_ids.repeat(num_windows)
        return data_torch , labels , subject_ids
    
    def _zscore_except_channels(
            self,
            data,
            ignored_ch=None,
            axis = 2
            ) -> np.ndarray:
        batch_size , n_channels , n_time_steps = data.shape
        if axis is None:
            raise ValueError("Please provide an axis to the normalization")
        
        if isinstance(ignored_ch, int):
            ignored_ch = [ignored_ch]

        mask = np.ones(n_channels, dtype=bool)
        if ignored_ch is not None and len(ignored_ch) > 0:
            mask[ignored_ch] = False




        data = data.copy() 
        selected = data[:, mask, :] 

        mean = selected.mean(axis=axis, keepdims=True)
        std = selected.std(axis=axis, keepdims=True)
        eps = 1e-10

        data[:, mask, :] = (selected - mean) / (std+eps)

        return data
        

    def _apply_filter(
            self,
            data: np.ndarray
        ) -> np.ndarray:
        try:
            is_list = isinstance(data, list)
            if is_list:
                data_array = np.stack(data)  # shape: (trials, channels, time)
            else:
                data_array = np.copy(data)

            if data_array.ndim != 3:
                raise ValueError(f"Expected 3D array (trials, channels, time), got shape {data_array.shape}")

            n_trials, n_channels, n_times = data_array.shape

            logger.info(f"Applying notch filter at {self.notch_freq} Hz with width {self.notch_width}")
            data_array[:,: self.n_cols_to_filter,:] = mne.filter.notch_filter(
                data_array[:,: self.n_cols_to_filter,:],
                Fs=self.fs,
                freqs=self.notch_freq,
                notch_widths=self.notch_width,
                filter_length="auto",
                verbose=False
            )

            logger.info(f"Applying bandpass filter: {self.bandpass_low}–{self.bandpass_high} Hz")
            data_array[:,: self.n_cols_to_filter,:] = mne.filter.filter_data(
                data_array[:,: self.n_cols_to_filter,:],
                sfreq=self.fs,
                l_freq=self.bandpass_low,
                h_freq=self.bandpass_high,
                filter_length="auto",
                verbose=False
            )


            return data_array

        except Exception as e:
            logger.error(f"Filtering failed: {str(e)}")
            raise
    def apply_preprocessing(
            self,
            data,
            labels,
            subject_ids
            ) -> Tuple[np.ndarray]:
        
        assert data.shape.__len__()==3 , "provided dimention of the data should be 3 (n_trials , n_channels , n_times)"
        data = np.copy(data)
        filtered_data = self._apply_filter(data)
        windows , labels , subject_ids = self._window_data(
            filtered_data,
            labels=labels,
            subject_ids=subject_ids,
            window_size=self.window_size,
            window_stride=self.window_stride
            )
        normalized_windows = self._zscore_except_channels(
            windows,
            ignored_ch=self.idx_to_ignore_normalization,
            axis = 2,
            )
        return normalized_windows , labels , subject_ids


preprocesser = SignalPreprocessor()
preprocessed_data , preprocessed_labels , preprocessed_sub_ids = preprocesser.apply_preprocessing(datas, labels , sub)
preprocessed_data
