import torch

class EEGDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for EEG data.

    This dataset is designed to handle EEG data for deep learning models.
    Each sample includes:
        - An EEG trial (channels × time samples)
        - A label
        - A sample weight (e.g., for class imbalance)
        - A subject identifier
        - Optional data augmentation

    Parameters
    ----------
    data_tensor : torch.Tensor
        Tensor of shape (n_trials, n_channels, n_samples). Each entry is an EEG trial.
    
    weights : torch.Tensor
        Tensor of shape (n_trials,). Contains sample weights — useful for weighted loss functions.
    
    label_tensor : torch.Tensor
        Tensor of shape (n_trials,). Contains class labels for each trial.
    
    subject_labels : torch.Tensor
        Tensor of shape (n_trials,). Identifiers for the subject corresponding to each trial.
    
    augment : bool, optional (default=False)
        Whether to apply data augmentation during training.
    
    augmentation_func : callable, optional
        A function that takes a single EEG trial (or a batch) and returns an augmented version of it.
        Must be provided if `augment=True`.

    Raises
    ------
    AssertionError
        If the number of trials (samples) across the input tensors don't match.

    TypeError
        If `augment=True` but no valid callable is passed for `augmentation_func`.

    Example
    -------
    >>> dataset = EEGDataset(data_tensor, weights, label_tensor, subject_labels,
    ...                      augment=True, augmentation_func=my_aug_fn)
    >>> trial, weight, label, subject_id = dataset[0]
    """

    def __init__(self, data_tensor, weigths, label_tensor, subject_labels, augment=False, augmentation_func=None,seed=None):
        # Input shape check
        assert data_tensor.shape[0] == label_tensor.shape[0] == weigths.shape[0] == subject_labels.shape[0], \
            "Mismatch in number of samples and labels"

        self.data = data_tensor
        self.labels = label_tensor
        self.weigths = weigths
        self.subject_labels = subject_labels
        self.augment = augment
        self.augmentation_func = augmentation_func
        self.seed = seed
        self.batch_counter=0
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.data.shape[0]
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def _increment_batch_counter(self):
        self.batch_counter+=1
    def _reset_batch_counter(self):
        self.batch_counter=0

    def __getitem__(self, idx):
        """
        Retrieves a single data sample (or a batch) by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple (data, weight, label, subject_label), with optional augmentation applied.
        """
        data = self.data[idx]
        weight = self.weigths[idx]
        label = self.labels[idx]
        subject_label = self.subject_labels[idx]

        # Apply augmentation if enabled
        if self.augment:
            if not callable(self.augmentation_func):
                raise TypeError("Parameter 'augmentation_func' must be a callable when 'augment' is True.")
            data = self.augmentation_func(data,seed=self.seed , idx = self.batch_counter)
            self._increment_batch_counter()

        return data, weight, label, subject_label