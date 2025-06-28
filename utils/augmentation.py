import torch

class AddGaussianNoise:
    """
    A callable class that adds Gaussian (normal) noise to a tensor.

    Parameters
    ----------
    mean : float, optional (default=0.0)
        The mean of the Gaussian noise to be added.
    std : float, optional (default=0.05)
        The standard deviation (spread or "intensity") of the Gaussian noise.

    """

    def __init__(self, mean=0.0, std=0.05):
        """
        Initialize the AddGaussianNoise object with desired mean and std.
        
        Parameters
        ----------
        mean : float
            The mean of the noise distribution.
        std : float
            The standard deviation (controls the spread) of the noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        Apply Gaussian noise to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to which noise will be added.

        Returns
        -------
        torch.Tensor
            A tensor of the same shape as `x`, with Gaussian noise added.
        """
        # Generate noise from normal distribution and add to the input
        return x + torch.randn_like(x) * self.std + self.mean



def augment_data(x):
    """
    Conditionally apply data augmentation to a tensor by adding Gaussian noise.

    This function randomly decides (with 40% probability) whether to apply 
    Gaussian noise to the input tensor, using a standard deviation of 0.1.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor representing a single data sample or batch.

    Returns
    -------
    torch.Tensor
        The (potentially augmented) tensor. If augmentation is not applied, 
        the original input is returned unchanged.

    """
    # Generate a random number from uniform[0,1] and apply noise with 40% chance
    if torch.rand((1,)).item() < 0.4:
        x = AddGaussianNoise(std=0.1)(x)
    
    return x