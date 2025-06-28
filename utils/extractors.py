import numpy as np
import mne

def extract_subject_labels(mne_files):

    """
    Extracts subject IDs from MNE file names and converts them to zero-based indices.

    This function assumes that each file name follows a specific format like:
        "MI_S36_1_9_.fif"
    where:
        - "S36" indicates the subject ID (in this case, subject 36).
    
    The function extracts the number following the 'S', converts it to an integer,
    and then subtracts 1 to produce a zero-based subject label (e.g., S36 → 35),
    which is often required by classifiers or one-hot encoders.

    Parameters
    ----------
    mne_files : list of mne.io.Raw or similar objects
        Each object must have a `filenames` attribute with the full file path as the first element.

    Returns
    -------
    np.ndarray
        An array of shape (n_files,) containing zero-based integer subject labels.

    Example
    -------
    >>> mne_files = [RawObjectWithFilename(".../MI_S36_1_9_.fif"), ...]
    >>> labels = extract_subject_labels(mne_files)
    >>> print(labels)
    array([35, ...])  # where 35 = 36 - 1
    """
        
    subject_labels = []
    for file in mne_files:
        path = str(file.filenames[0])
        subject_id = path.split("/")[-1].split(".")[0].split("_")[1]
        subject_id_int = (int(subject_id.removeprefix("S")))
        subject_labels.append(subject_id_int)
    return np.asarray(subject_labels,dtype=int)-1




def extract_data(fif_paths , return_id = False):

    """
    Loads raw EEG data from a list of `.fif` file paths and extracts labels and optionally subject IDs.

    This function is designed for EEG datasets where each `.fif` file contains:
        - The EEG signal data.
        - A description label stored in `raw.info['description']` (e.g., class or task label).
        - Subject information in `raw.info['subject_info']['id']` (e.g., subject ID).

    Parameters
    ----------
    fif_paths : list of str
        A list of file paths to `.fif` files (typically raw EEG files).
    
    return_id : bool, optional (default=False)
        Whether to also return the subject IDs extracted from the MNE metadata.

    Returns
    -------
    tuple
        If return_id is False:
            (mne_files, labels)
        If return_id is True:
            (mne_files, labels, subject_ids)

        - mne_files : list of mne.io.Raw
            The raw EEG objects loaded from each file.
        
        - labels : np.ndarray of str
            The description field from each file, used as a class or condition label.
        
        - subject_ids : np.ndarray of int (only if return_id=True)
            Subject identifiers from the metadata.

    Notes
    -----
    - This function temporarily sets the MNE logging level to "ERROR" to suppress verbose output 
      during loading, and restores it to "INFO" afterward.
    - It assumes that the `.fif` files have a valid 'description' field and subject info.
      If these fields are missing, the function will raise a KeyError.

    Example
    -------
    >>> fif_paths = ["./data/MI_S36_1_9_.fif", "./data/MI_S37_1_9_.fif"]
    >>> mne_files, labels = extract_data(fif_paths)
    >>> print(labels)
    array(['left', 'right'])

    >>> mne_files, labels, ids = extract_data(fif_paths, return_id=True)
    >>> print(ids)
    array([1234567, 1234568])
    """

    mne.set_log_level("ERROR")
    mne_files = []
    labels = []
    ids = []
    for path in fif_paths:
        raw = mne.io.read_raw_fif(path, preload=True)
        mne_files.append(raw)
        labels.append(raw.info["description"])
        ids.append(raw.info['subject_info']['id'])
    mne.set_log_level("INFO")
    if return_id:
        return mne_files, np.asarray(labels) , np.asarray(ids)
    else:
        return mne_files , np.asarray(labels)
    



def extract_trial(n , df , task = None):
    """
    Extracts the `n`-th EEG trial from a continuous EEG recording DataFrame.

    This function assumes that the EEG data is stored as a flat (continuous) DataFrame
    containing multiple trials *back-to-back* — each of fixed length depending on the task type:
    
        - For SSVEP tasks: each trial is 7 seconds long.
        - For MI (Motor Imagery) tasks: each trial is 9 seconds long.

    The function extracts the correct range of samples corresponding to trial `n`, based on the task.

    Parameters
    ----------
    n : int
        The index of the trial to extract (0-based).
    
    df : pandas.DataFrame
        The continuous EEG data. Expected to be sampled at 250 Hz.
        The total number of rows should be: `n_trials * trial_length_in_seconds * 250`.
    
    task : str
        The task type, either:
            - `"ssvep"`: Steady-State Visual Evoked Potential
            - `"mi"`: Motor Imagery
        This determines the trial length in seconds.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only the `n`-th trial's data.
        The index is reset (i.e., starts from 0).

    Raises
    ------
    AssertionError
        If `task` is not "ssvep" or "mi" (case-insensitive).

    Example
    -------
    >>> trial_5 = extract_trial(5, eeg_df, task="MI")
    >>> print(trial_5.shape)
    (2250, n_channels)  # 9 seconds * 250 Hz = 2250 samples
    """
    assert task.lower() in ["mi" , "ssvep"]
    n_samples = 250 * (7 if task.lower()=="ssvep" else 9)
    start_index = n*n_samples
    end_index = (n+1)*n_samples
    return df[start_index:end_index].copy().reset_index(drop=True)

