

import numpy as np
from typing import Union, List , Tuple , Optional
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
            crop_range : Optional[Tuple[float , float]] = None
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
        self.crop_range = crop_range

        if not isinstance(fs, int) or fs <= 0:
            raise ValueError("Sampling frequency 'fs' must be a positive integer.")

        if not isinstance(notch_width, (float, int)) or notch_width <= 0:
            raise ValueError("Notch width must be a positive number.")

        if not isinstance(notch_freq, (float, list)):
            raise ValueError("notch_freq must be a float or a list of floats.")
        if isinstance(notch_freq, list):
            if not all(isinstance(f, (float, int)) and f > 0 for f in notch_freq):
                raise ValueError("All notch frequencies must be positive numbers.")
        elif notch_freq <= 0:
            raise ValueError("notch_freq must be positive.")

        if not isinstance(bandpass_low, (float, int)) or bandpass_low <= 0:
            raise ValueError("bandpass_low must be a positive number.")
        if not isinstance(bandpass_high, (float, int)) or bandpass_high <= 0:
            raise ValueError("bandpass_high must be a positive number.")
        if bandpass_low >= bandpass_high:
            raise ValueError("bandpass_low must be less than bandpass_high.")

        if not isinstance(n_cols_to_filter, int) or n_cols_to_filter <= 0:
            raise ValueError("n_cols_to_filter must be a positive integer.")

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        if not isinstance(window_stride, int) or window_stride <= 0:
            raise ValueError("window_stride must be a positive integer.")

        if not isinstance(idx_to_ignore_normalization, int):
            raise ValueError("idx_to_ignore_normalization must be an integer.")

        if crop_range is not None:
            if (
                not isinstance(crop_range, tuple)
                or len(crop_range) != 2
                or not all(isinstance(x, (float, int)) for x in crop_range)
                or crop_range[0] < 0
                or crop_range[1] <= crop_range[0]
            ):
                raise ValueError(
                    "crop_range must be a tuple of two numbers (start_sec, end_sec), with 0 <= start < end."
                )

        logger.info(
            f"Initialized SignalPreprocessor with configuration:\n"
            f"  • Sampling frequency (fs): {self.fs} Hz\n"
            f"  • Notch filter: freqs={self.notch_freq}, width={self.notch_width}\n"
            f"  • Bandpass filter: low={self.bandpass_low} Hz, high={self.bandpass_high} Hz\n"
            f"  • Number of filtered channels: {self.n_cols_to_filter}\n"
            f"  • Windowing: size={self.window_size}, stride={self.window_stride}\n"
            f"  • Channel index to ignore in normalization: {self.idx_to_ignore_normalization}\n"
            f"  • Crop range (in seconds): {self.crop_range if self.crop_range else 'None'}"
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

        assert data.shape[0] == labels.shape[0] == subject_ids.shape[0] , "mismatch in input arrays lengths"
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

    def _crop_signals(
            self,
            data,
            crop_range=None
            ) -> np.ndarray:
        assert bool(crop_range) and all(bool(i) for i in crop_range ) , "crop_range has to be a tuple with two valid floats (start_second , end_second)"
        start_sec , end_sec = crop_range
        start_frame , end_frame = int(start_sec*self.fs) , int(end_sec*self.fs)
        _ , _ , n_times = data.shape
        assert start_frame>=0 and end_frame <= n_times , f"crop range_start has to be >=0 seconds and range_end has to be <= {n_times//self.fs} seconds"

        return data[:,:,start_frame:end_frame]
    def apply_preprocessing(
            self,
            data,
            labels,
            subject_ids
            ) -> Tuple[np.ndarray]:
        
        assert data.shape.__len__()==3 , "provided dimention of the data should be 3 (n_trials , n_channels , n_times)"
        data = np.copy(data)
        filtered_data = self._apply_filter(data)

        if self.crop_range is not None:
            if self.crop_range[0] is None or self.crop_range[1] is None:
                raise ValueError("if crop_range is passed ..It has to consist of 2 valid floats")
            filtered_data = self._crop_signals(
                data=filtered_data,
                crop_range=self.crop_range
            )

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

####----Example Usage-----####
# preprocessor = SignalPreprocessor(
#     fs=250,                               
#     notch_freq=[50.0, 100.0],             
#     notch_width=1.0,                      
#     bandpass_low=6.0,                     
#     bandpass_high=30.0,                  
#     n_cols_to_filter=4,                   
#     window_size=600,                      
#     window_stride=600,                    
#     idx_to_ignore_normalization=-1,        
#     crop_range=(2.5, 7.5)                 
# )
# preprocessed_data , preprocessed_labels , preprocessed_sub_ids = preprocessor.apply_preprocessing(datas, labels , sub)
# preprocessed_data
