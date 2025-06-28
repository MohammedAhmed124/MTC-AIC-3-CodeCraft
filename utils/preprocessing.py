import logging
from joblib import delayed, Parallel
from copy import deepcopy
import mne 
from tqdm import tqdm
import numpy as np 



def window(mne_file, size = 250, stride = 125):

        """
    Slice a continuous MNE Raw object into overlapping time windows.

    This function takes an MNE Raw object (containing EEG)
    and applies a sliding window over the time axis to extract multiple short, fixed-length
    time segments (windows) from the original recording. Each window is then wrapped again
    as a new MNE Raw object with the same channel information.

    Parameters
    ----------
    mne_file : mne.io.Raw
        The input continuous MNE Raw object (e.g., EEG recording) to be windowed.
    
    size : int, optional (default=250)
        The number of time points (samples) in each window.
        For example, with a sampling rate of 250 Hz, a window size of 250 corresponds to 1 second.

    stride : int, optional (default=125)
        The number of time points to shift the window by for each step (controls overlap).
        For example, with a stride of 125 and size of 250, the windows will have 50% overlap.

    Returns
    -------
    result_windows : List[mne.io.Raw]
        A list of new MNE Raw objects, each containing a snapshot (window) of the original signal.
        These windows can be used as inputs for machine learning or signal processing pipelines.

    Notes
    -----
    - If a window exceeds the original signal length (i + size > n_times), it is skipped.
    - All windows preserve the original channel names and types.
    

    Reason of use:
    -----
    - Primarily to increase the number of training examples.

    Example
    -------
    >>> raw = mne.io.read_raw_fif("subject01_raw.fif", preload=True)
    >>> windows = window(raw, size=500, stride=250)
    >>> print(len(windows))  # Number of time windows
    >>> print(windows[0].get_data().shape)  # (n_channels, window_size)
    """
        
        data = mne_file.get_data()
        ch_names = mne_file.ch_names
        ch_types = mne_file.get_channel_types()
        n_channels , n_times = data.shape
        result_windows = []
        info = mne.create_info(ch_names=ch_names , ch_types=ch_types,sfreq=mne_file.info["sfreq"])
        for i in range(0 ,n_times,stride):
            if i+size>=n_times:
                break
            window = data[:,i:i+size]
            window_mne = mne.io.RawArray(window,info=info)
            result_windows.append(window_mne)
        return result_windows



def z_score_normalize(arr,channel_idx_to_ignore=None):
    """
        Apply z-score normalization to EEG data along the time axis (last axis),
    optionally skipping specific channel indices (e.g., binary channels like 'Validation').

    This function performs per-channel normalization across time for multichannel EEG data.
    It standardizes each signal to zero mean and unit variance using the z-score formula:

        z = (x - μ) / (σ + ε)

    where:
    - x is the signal (for a specific channel),
    - μ is the mean over time,
    - σ is the standard deviation over time,
    - ε is a small constant to avoid division by zero.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array of shape `(n_channels, n_times)` representing EEG signal data.
    
    channel_idx_to_ignore : int or list of int, optional (default=None)
        Index (or indices) of channel(s) to exclude from normalization.
        Commonly used to exclude non-continuous or categorical channels like 'Validation',
        which is provided as binary (0/1) and should not be scaled as it distorts its meaning.

    Returns
    -------
    normalized_arr : np.ndarray
        The z-score normalized EEG data. If `channel_idx_to_ignore` is provided,
        those channels retain their original values.

    Notes
    -----
    - Normalization is done across the time axis (last axis), i.e., per channel.
    - A very small epsilon (1e-10) is added to the denominator to ensure numerical stability.
    - Skipping normalization for binary channels (e.g., event markers) prevents the loss of their semantic value.

    Example
    -------
    >>> eeg_data.shape  # (8 channels, 3000 time points)
    (8, 3000)
    >>> z_normalized = z_score_normalize(eeg_data, channel_idx_to_ignore=7)
    >>> z_normalized.shape
    (8, 3000)
    >>> np.allclose(z_normalized[0].mean(), 0, atol=1e-3)  # normalized
    True
    >>> np.array_equal(z_normalized[7], eeg_data[7])  # not normalized because we chose it as an index to ignore
    True

    
    """


    up = arr-arr.mean(axis = -1 , keepdims = True)
    down = arr.std(axis = -1 , keepdims=True)+1e-10
    normalized_arr = up/down
    if not channel_idx_to_ignore:
        return normalized_arr
    else:
        normalized_arr[channel_idx_to_ignore,:] = arr[channel_idx_to_ignore,:]
        return normalized_arr



def preprocess_one_file(
        mne_file,
        label,
        subject_label,
        cols_to_pick=None,
        l_freq=6,
        h_freq=30,
        notch_freqs=[50,100],
        notch_width = 1.0,
        window_size=600,
        window_stride=25
        ):
    
    """
    Preprocess a single EEG recording (`mne_file`) by applying a sequence of filtering,
    channel selection, windowing, and normalization operations.

    This function is designed to be used within a parallel processing setup where
    multiple files are preprocessed simultaneously. It performs the following steps:

    1. **Suppress Logging**: Reduces verbosity from `joblib` and `mne` for cleaner outputs.
    2. **Deep Copy**: Copies the input MNE Raw object to avoid modifying it in place.
    3. **Notch Filtering**: Removes power-line artifacts (e.g., 50Hz and harmonics).
    4. **Bandpass Filtering**: Isolates the frequency range of interest (default: 6–30 Hz for MI).
    5. **Channel Selection**: Picks only relevant EEG/auxiliary channels (e.g., 'C3', 'gyro_norm', etc.).
    6. **Windowing**: Slices the continuous signal into overlapping fixed-length segments (sliding windows).
    7. **Normalization**: Applies z-score normalization per window, with optional exclusion
       of binary or non-continuous channels (e.g., 'Validation').

    Parameters
    ----------
    mne_file : mne.io.Raw
        Raw EEG signal loaded using the MNE library.
    
    label : Any
        The class label corresponding to the entire file (e.g., motor imagery class or SSVEP class).
    
    subject_label : Any
        The subject ID or identifier for the person whose EEG was recorded. (for later adversarial training)
    
    cols_to_pick : list of str, optional
        Channel names to keep from the EEG signal. Must include the 'Validation' channel
        if it's used for weighting later.
    
    l_freq : float, default=6
        Low cutoff frequency for bandpass filtering.
    
    h_freq : float, default=30
        High cutoff frequency for bandpass filtering.
    
    notch_freqs : list of float, default=[50, 100]
        Frequencies at which to apply notch filters to remove line noise.
    
    notch_width : float, default=1.0
        The width of each notch filter.
    
    window_size : int, default=600
        Number of time samples per window.
    
    window_stride : int, default=25
        Step size (in samples) to move the window between successive windows.

    Returns
    -------
    windows : list of np.ndarray
        List of normalized window arrays, each with shape (n_channels, window_size).
    
    label_list : list
        A list of labels (repeated `len(windows)` times), one for each window.
    
    subject_label_list : list
        A list of subject identifiers (repeated `len(windows)` times), one for each window.

    Notes
    -----
    - The function uses `z_score_normalize` to normalize each window individually.
    - The Validation channel, if present, is excluded from normalization to avoid distorting its binary nature.
     (e.g., values like 0 or 1 which represent event markers).
    - The output is designed to be aggregated later using a wrapper function such as
      `preprocess_data()` which runs this function in parallel for many files.

    """
        

    logging.getLogger('joblib').setLevel(logging.ERROR)
    mne.set_log_level('ERROR')
    mne_file = deepcopy(mne_file)



    mne_file.notch_filter(freqs=notch_freqs, verbose=False, notch_widths=notch_width,
                          filter_length="auto",
                          picks='eeg',)

    mne_file.filter(l_freq=l_freq, h_freq=h_freq)
    
    try:
        validation_idx = cols_to_pick.index("Validation")
    except:
        validation_idx = None


    mne_file.pick_channels(cols_to_pick)
    windows = window(mne_file,  size = window_size, stride = window_stride)

    windows = [
        z_score_normalize(
            x.get_data(),
            channel_idx_to_ignore=validation_idx
            )   for x in windows
        ]
    return windows , [label]*len(windows),[subject_label]*len(windows)



def preprocess_data(
        train_data_mne,
        labels,
        subject_labels,
        preprocess_func=None,
        params=None,
        n_jobs=4
        ):
    

    """
    Preprocess a batch of EEG recordings in parallel and organize the resulting windowed data.

    This function is designed to:
    1. Run preprocessing in parallel for efficiency using `joblib.Parallel`.
    2. Aggregate and organize the outputs into clean NumPy arrays.
    3. Extract per-window weights based on a binary "Validation" channel (takes the mean for each window), used for sample weighting and soft voting later.

    Parameters
    ----------
    train_data_mne : list of mne.io.Raw
        List of EEG recordings (MNE Raw objects).

    labels : list
        Class label for each file in `train_data_mne`.

    subject_labels : list
        A subject ID or identifier for each file (parallel to `train_data_mne` and `labels`).

    preprocess_func : callable
        A function that processes a single MNE file. Expected to return a tuple of:
        - windowed signal list (list of np.ndarrays),
        - corresponding labels list,
        - corresponding subject IDs list.
        Example: `preprocess_one_file()`.

    params : dict, optional
        A dictionary of parameters to pass into `preprocess_func`. This includes:
        - `"cols_to_pick"` (list of str): channel names to keep (must include `"Validation"`).
        - `"window_size"`, `"window_stride"`, `"l_freq"`, `"h_freq"`, etc.

    n_jobs : int, default=10
        Number of parallel processes to spawn (should not exceed your CPU core count).

    Returns
    -------
    data : np.ndarray
        3D array of shape `(N_windows, N_channels, window_length)` containing the normalized signal windows.

    validation_weights : np.ndarray
        1D array of length `N_windows`, containing the average value of the 'Validation' channel for each window. (How much of the signal is not corrupted)
        This will be used later for weighting losses and confidence-aware learning.

    labels : np.ndarray
        Array of window-level labels (same label repeated across all windows of a file).

    subject_labels : np.ndarray
        Array of window-level subject IDs (same subject ID repeated across all windows of a file).

    WINDOW_LEN : int
        Number of windows per file (assumes all files produce the same number of windows, taken from the last file).

    Raises
    ------
    ValueError
        If the "Validation" channel is not found in `cols_to_pick`, since this channel is required
        for calculating `validation_weights`.

    Notes
    -----
    - The function uses `tqdm` to show progress through the dataset.
    - It assumes each file is fully preprocessed independently and it merges the resulting lists.
    - The final structure is ideal for feeding into the machine learning pipelines (e.g., PyTorch Datasets).
    - `validation_weights` are especially useful for semi-supervised or domain-adaptive training,
      where some windows carry more relevance than others.

    Example
    -------
    >>> params = {"cols_to_pick": ["C3", "C4", "Validation"], "window_size": 600}
    >>> data, weights, y, s, win_len = preprocess_data(
    ...     mne_data_list, labels, subject_ids,
    ...     preprocess_func=preprocess_one_file, params=params, n_jobs=4
    ... )
    >>> data.shape
    (total_windows, n_channels, 600)
    """
        

    params={} if not params else params
    results = Parallel(n_jobs=n_jobs, verbose=0)( 
        delayed(preprocess_func)(mne_file,label,subject_label,**params)
        for mne_file,label,subject_label in tqdm(zip(train_data_mne,labels,subject_labels))
    )

    try:
        validation_idx = params['cols_to_pick'].index("Validation")
    except:
        raise ValueError("Validation is not available")




    data = []
    label_list = []
    subject_label_list = []
    validation_weights = []
    for window, label ,subject_label in results:
        window_split = window.copy()
        val_weights = [
            x[validation_idx,:].copy().mean() for x in window
            ]


        data.extend(window_split)    
        validation_weights.extend(val_weights)            
        label_list.extend(label)
        subject_label_list.extend(subject_label)


        WINDOW_LEN = len(window)

    data = np.asarray(data)
    labels = np.asarray(label_list)
    subject_labels = np.asarray(subject_label_list)
    validation_weights = np.asarray(validation_weights)

    return data,validation_weights, labels,subject_labels , WINDOW_LEN