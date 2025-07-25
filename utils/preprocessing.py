
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
            bandpass_low: float = 6.0,
            bandpass_high: float = 30.0,
            n_cols_to_filter: int = 4,
            window_size : int = 600,
            window_stride : int = 600,
            idx_to_ignore_normalization : int =-1,
            crop_range : Optional[Tuple[float , float]] = None,
            ):
        """
    A utility class for EEG preprocessing — handles filtering, normalization, cropping, and windowing of EEG data
    before feeding it to a model. Meant to clean and prepare signals from raw trials into well-shaped inputs for training.

    Parameters
    ----------
    fs : int
        Sampling frequency of the EEG signals, in Hz. Defaults to 250 Hz.
    
    bandpass_low : float
        The low cutoff frequency for the bandpass filter. EEG activity below this (like drift) will be removed.
    
    bandpass_high : float
        The high cutoff frequency for the bandpass filter. Anything above this (like muscle artifacts) gets filtered out.
    
    n_cols_to_filter : int
        Number of the first channels (e.g., EEG electrodes) to apply bandpass filtering to. Often, only EEG channels are filtered, not auxiliary sensors.
    
    window_size : int
        Number of time points per window. This slices each trial into overlapping or non-overlapping segments.
    
    window_stride : int
        Number of time points to step forward when creating the next window. If equal to window_size, windows don’t overlap.
    
    idx_to_ignore_normalization : int
        Index of a channel (like a quality/validation signal) to skip during z-scoring. Often used to preserve labels or non-EEG signals.
    
    crop_range : tuple(float, float), optional
        Time in seconds to crop the signal (start_time, end_time). Used to ignore irrelevant portions of the signal like pre-trial pauses.
    """
        self.fs = fs
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.n_cols_to_filter = n_cols_to_filter
        self.window_size = window_size
        self.window_stride = window_stride
        self.idx_to_ignore_normalization = idx_to_ignore_normalization
        self.crop_range = crop_range

        if not isinstance(fs, int) or fs <= 0:
            raise ValueError("Sampling frequency 'fs' must be a positive integer.")


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
        
        """
    Efficiently slices each EEG trial into overlapping windows using a vectorized PyTorch operation.

    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_trials, n_channels, input_size). Each trial is a full recording window.

    labels : np.ndarray
        Trial-wise class labels of shape (n_trials,). Each label is repeated for all windows from that trial.

    subject_ids : np.ndarray
        Trial-wise subject identifiers of shape (n_trials,). Also repeated for each window.

    window_size : int
        Number of time points in each window.

    window_stride : int
        Number of time points to move forward between windows.

    Returns
    -------
    windowed_data : np.ndarray
        A stacked array of shape (n_windows_total, n_channels, window_size), where all windows from all trials are packed together.

    windowed_labels : np.ndarray
        A 1D array of shape (n_windows_total,) with repeated trial labels for each window.

    windowed_subject_ids : np.ndarray
        A 1D array of shape (n_windows_total,) with repeated subject IDs for each window.

    Notes
    -----
    This method is fully vectorized using `torch.unfold`, which avoids slow Python loops and makes the windowing process fast and scalable even on large EEG datasets.
    """
        _ , n_channels , input_size = data.shape

        assert data.shape[0] == labels.shape[0] == subject_ids.shape[0] , "mismatch in input arrays lengths"
        data_torch =torch.from_numpy(data)
        data_torch = data_torch.unfold(dimension=2, size=window_size, step=window_stride).permute(0, 2, 1, 3)
        data_torch = data_torch.reshape(-1 , n_channels , window_size).cpu().numpy()

        self.num_windows_per_trial = (input_size - window_size) // window_stride + 1

        labels = labels.repeat(self.num_windows_per_trial)
        subject_ids = subject_ids.repeat(self.num_windows_per_trial)
        return data_torch , labels , subject_ids
    
    def _zscore_except_channels(
            self,
            data,
            ignored_ch=None,
            axis = 2
            ) -> np.ndarray:
        
        """
    Applies z-score normalization to the EEG signal along the specified axis,
    excluding some channels (like the validation channel) to avoid distorting
    binary or categorical information.

    Normalization is applied only to the channels not listed in `ignored_ch`.
    This is especially useful when one of the channels holds metadata (e.g.,
    validation masks in our case) to avoid destorting its binary nature.

    Mathematically, it performs:
        X_norm = (X - mean) / (std + eps)
    where eps is a small value to avoid division by zero.

    Parameters:
        data (np.ndarray): Input array of shape (batch, channels, time).
        ignored_ch (list or int, optional): Channels to exclude from normalization.
        axis (int): The axis over which to compute mean and std.

    Returns:
        np.ndarray: Normalized data with the same shape.
    """
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

        """
    Bandpass filters each trial using MNE, working across batched EEG input.

    This function expects a 3D array shaped (trials × channels × time) and applies
    a bandpass filter using MNE’s `filter_data`. It only filters the first 
    `self.n_cols_to_filter` channels — so make sure your input channel order 
    puts the EEG ones first! (i.e., before any extras like validation masks or accelerometers).

    It also handles both lists of trials and pre-stacked arrays smoothly.

    Logs what it's doing, and throws a clear error if the input shape isn’t what it expects.

    Returns:
        np.ndarray: Filtered EEG data with the same shape as input.
    """
        try:
            is_list = isinstance(data, list)
            if is_list:
                data_array = np.stack(data)  # shape: (trials, channels, time)
            else:
                data_array = np.copy(data)

            if data_array.ndim != 3:
                raise ValueError(f"Expected 3D array (trials, channels, time), got shape {data_array.shape}")

            n_trials, n_channels, n_times = data_array.shape

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
        
        """
        Crops each trial in a batch of EEG signals to a specific time window.

        This function takes a 3D array shaped (trials × channels × time) and trims 
        each trial to a window defined by `crop_range`, which should be a tuple like 
        (start_sec, end_sec). It converts those seconds into frames using the sampling 
        rate (`self.fs`) and slices the data along the time axis.

        Make sure the crop range is within the actual signal length — otherwise it’ll 
        throw an assertion error. Also returns the data with the same shape format, 
        just shorter in time.

        Args:
            data (np.ndarray): The batched EEG input to crop.
            crop_range (tuple): A pair of floats (start_sec, end_sec) that define the 
                                segment to keep in each trial.

        Returns:
            np.ndarray: Cropped EEG data of shape (trials × channels × new_time).
        """
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
            subject_ids,
            ) -> Tuple[np.ndarray]:
        

        """
        Main preprocessing pipeline for EEG data.

        This is the central function that orchestrates all preprocessing steps. It applies
        a sequence of transformations to raw EEG trials in a fixed order to prepare the 
        data for training or inference. All other helper functions (_apply_filter, 
        _crop_signals, _window_data, _zscore_except_channels) are called from here.

        The pipeline follows this sequence:
            1. Apply bandpass filtering (if enabled).
            2. Crop the signal in time using `crop_range` (if provided).
            3. Segment each trial into overlapping windows.
            4. Apply z-score normalization to each window, except on specified channels.
            5. Compute window-level weights based on the average value of the last channel 
            (assumed to be a 'validation' or signal quality indicator).

        NOTE:
            - Input `data` must have shape (n_trials, n_channels, n_times).
            - Make sure your input channels are ordered correctly, especially if using 
            bandpass filtering or channel-specific normalization.
            - This function returns preprocessed data, updated labels/subject IDs per window, 
            and computed weights for each window.

        Args:
            data (np.ndarray): Raw EEG data of shape (n_trials, n_channels, n_times).
            labels (np.ndarray): Trial-level labels.
            subject_ids (np.ndarray): Subject ID per trial.

        Returns:
            Tuple[np.ndarray]: A tuple of:
                - normalized_windows: preprocessed EEG windows,
                - labels: repeated labels per window,
                - subject_ids: repeated subject IDs per window,
                - weights: window-level weights based on validation channel.
            """
        
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
        weights = normalized_windows[:,-1,:].mean(axis = -1) #mean of validation column (assumed to be placed last)
        return normalized_windows , labels , subject_ids , weights
