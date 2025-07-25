# import torch

# class AddGaussianNoise:
#     """
#     A callable class that adds Gaussian (normal) noise to a tensor.

#     Parameters
#     ----------
#     mean : float, optional (default=0.0)
#         The mean of the Gaussian noise to be added.
#     std : float, optional (default=0.05)
#         The standard deviation (spread or "intensity") of the Gaussian noise.

#     """

#     def __init__(self, mean=0.0, std=0.05):
#         """
#         Initialize the AddGaussianNoise object with desired mean and std.
        
#         Parameters
#         ----------
#         mean : float
#             The mean of the noise distribution.
#         std : float
#             The standard deviation (controls the spread) of the noise.
#         """
#         self.mean = mean
#         self.std = std

#     def __call__(self, x):
#         """
#         Apply Gaussian noise to the input tensor.

#         Parameters
#         ----------
#         x : torch.Tensor
#             The input tensor to which noise will be added.

#         Returns
#         -------
#         torch.Tensor
#             A tensor of the same shape as `x`, with Gaussian noise added.
#         """
#         # Generate noise from normal distribution and add to the input
#         return x + torch.randn_like(x) * self.std + self.mean



# def augment_data(x):
#     """
#     Conditionally apply data augmentation to a tensor by adding Gaussian noise.

#     This function randomly decides (with 40% probability) whether to apply 
#     Gaussian noise to the input tensor, using a standard deviation of 0.1.

#     Parameters
#     ----------
#     x : torch.Tensor
#         The input tensor representing a single data sample or batch.

#     Returns
#     -------
#     torch.Tensor
#         The (potentially augmented) tensor. If augmentation is not applied, 
#         the original input is returned unchanged.

#     """
#     # Generate a random number from uniform[0,1] and apply noise with 40% chance
#     if torch.rand((1,)).item() < 0.4:
#         x = AddGaussianNoise(std=0.1)(x)
    
#     return x




import random
import scipy.signal
import torch
import numpy as np

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std + self.mean


class TimeShift:
    def __init__(self, max_shift=100):  # in samples
        self.max_shift = max_shift

    def __call__(self, x):
        shift = random.randint(-self.max_shift, self.max_shift)
        return torch.roll(x, shifts=shift, dims=-1)


class AmplitudeScaling:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x):
        scale = torch.empty((x.shape[0], 1)).uniform_(*self.scale_range).to(x.device)
        return x * scale


class TimeWarp:
    def __init__(self, max_warp=0.2):  # 0.2 = 20% of time axis
        self.max_warp = max_warp

    def __call__(self, x):
        orig = np.linspace(0, 1, x.shape[-1])
        warp = orig + np.random.uniform(-self.max_warp, self.max_warp, size=orig.shape)
        warp = np.clip(warp, 0, 1)
        warped = np.zeros_like(x)
        for c in range(x.shape[0]):
            warped[c] = np.interp(orig, warp, x[c].cpu().numpy())
        return torch.from_numpy(warped).to(x.device).type_as(x)




def augment_data(x):
    """
    Apply a suite of augmentations with some probability to MI EEG data.
    
    Parameters
    ----------
    x : torch.Tensor (C x T)
        EEG sample with channels C and time points T.
    
    Returns
    -------
    torch.Tensor
        Augmented EEG sample.
    """


    if torch.rand(1).item() < 0.4:
        x = AddGaussianNoise(std=0.1)(x)

    # if torch.rand(1).item() < 0.3:
    #     x = TimeShift(max_shift=80)(x)

    # if torch.rand(1).item() < 0.3:
    #     x = AmplitudeScaling(scale_range=(0.8, 1.2))(x)

    # if torch.rand(1).item() < 0.3:
    #     x = TimeWarp(max_warp=0.15)(x)

    # if torch.rand(1).item() < 0.2:
    #     x = RandomChannelDropout(drop_prob=0.3)(x)

    return x
