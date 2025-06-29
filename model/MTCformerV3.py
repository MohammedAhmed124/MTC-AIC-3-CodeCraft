import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import math
import argparse
import sys


class ConvolutionalAttentionBlock(nn.Module):
    """
    ConvolutionalAttentionBlock implements a convolution-inspired alternative to self-attention, 
    inspired by the SSVEPFormer architecture.

     Overview:
    This block replaces classical token-wise attention with temporal convolution, followed by 
    feedforward processing, using residual connections and LayerNorm.

     Input shape:
        x: Tensor of shape [B, proj_dim, n_times], where:
            - B: Batch size
            - proj_dim: Number of input/output channels (features)
            - n_times: Temporal length of the signal (time dimension)

     Architecture:
    1. LayerNorm over time axis: 
       - Normalizes each time point across all channels.
    2. Temporal convolution over time:
       - Uses a grouped conv (`groups=1`) to mix local temporal features.
       - This acts like a local attention over time (temporal receptive field).
    3. GELU + Dropout + Residual Connection
        – Uses the Gaussian Error Linear Unit (GELU) activation.
        – Dropout is applied for regularization.
        – A residual connection (skip connection) helps stabilize training and enables deeper architectures by mitigating vanishing gradients.

    4. LayerNorm again + Feedforward Linear Layer over time axis:
       - Projects time features independently per channel.
    5. Final GELU + Dropout + Residual

     Purpose:
        -Capture short-range temporal dependencies in EEG signals using convolutions instead of traditional attention mechanisms.

        -Use temporal convolutional filters to efficiently learn local patterns over time.

        -Employ residual connections to stabilize training, preserve signal identity, and ensure effective gradient flow.

        -Avoid the high computational cost associated with traditional self-attention layers.

        -Most importantly, this approach has shown to achieve strong performance in practice.

     Linked to Convolutional Attention:
    - Instead of learning attention weights explicitly (like in Transformers), 
      the convolution here acts as a fixed-size temporal attention window.
    - It mimics attention behavior by locally aggregating information over time using filters.

    Inspired from: SSVEPFormer.
    """
    def __init__(
            self,
            proj_dim,
            n_times,
            kernel_size,
            dropout
            ):
        super().__init__()
        self.norm_1 = nn.LayerNorm(n_times) 
        self.conv=nn.Conv1d(proj_dim, proj_dim, kernel_size=kernel_size, padding="same", groups=1)

        self.activation_1 = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)

        self.norm_2 = nn.LayerNorm(n_times)
        self.feedforward = nn.Linear(n_times,n_times)
        self.activation_2 = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x):  # x: [B, proj_dim, n_times]
        residual = x
        
        x = self.norm_1(x)

        x = self.conv(x)

        x = self.activation_1(x)
        x = self.dropout_1(x)
        x = x + residual                       # Residual connection

        residual = x
        x = self.norm_2(x)
        x = self.feedforward(x)               
        x = self.activation_2(x)
        x = self.dropout_2(x)
        x = x + residual                       # Residual connection

        return x


class ConvolutionalAttention(nn.Module):
    """
    A stacked sequence of convolutional attention blocks.

    Args:
        depth (int): Number of repeated convolutional attention blocks.
        proj_dim (int): Number of projection dimensions (input/output channels).
        n_times (int): Temporal length of the input sequence.
        kernel_size (int): Kernel size used in the temporal convolution.
        dropout (float): Dropout rate used inside each block.

    Forward Input:
        x (Tensor): Input tensor of shape [batch_size, proj_dim, n_times]

    Output:
        Tensor: Output tensor of the same shape after passing through stacked attention blocks.

    Purpose:
        - Each block learns local temporal features with convolutional filters.
        - Stacked structure allows the model to capture hierarchical temporal dependencies.
    """
    def __init__(
            self,
            depth,
            proj_dim,
            n_times,
            kernel_size,
            dropout
            ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvolutionalAttentionBlock(
                proj_dim,
                n_times,
                kernel_size,
                dropout
            )

            for _ in range(depth)
        ])

    def forward(self, x): 
        for block in self.layers:
            x = block(x)
        return x




class GradientReversalFunction(torch.autograd.Function):

    """
    A custom autograd function for domain adaptation that reverses the gradient during backpropagation.

    Usage:
        Primarily used before the domain classifier head in domain adaptation setups (e.g., DANN)
        to encourage the feature extractor to learn domain-invariant representations.

    Forward Pass:
        - Passes the input `x` unchanged to the domain classifier.

    Backward Pass:
        - Reverses the gradient by multiplying it with `-lambda_`.
        - When `lambda_ = 0`, it effectively blocks gradient flow to the preceding layers (no domain signal).
        - When `lambda_ > 0`, it fools the domain classifier by making the feature extractor maximize its loss.

    Args:
        x (Tensor): Input tensor to the domain classifier.
        lambda_ (float): Scaling factor for the reversed gradient.

    Returns:
        Tensor: Same as input `x` during forward, with gradient reversed during backward.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None
    

def grad_reverse(x, lambda_):
    """
    Applies the gradient reversal operation.

    Args:
        x (Tensor): Input features.
        lambda_ (float): Gradient reversal coefficient (controls strength of domain confusion).

    Returns:
        Tensor: Input passed forward as-is, but gradients are reversed during backprop.
    """
    return GradientReversalFunction.apply(x, lambda_)


class DomainClassifier(nn.Module):
    """
    Domain Classifier Head used in adversarial domain adaptation (e.g., DANN).

    Purpose:
    --------
    - Predicts the domain label (e.g., subject ID) from extracted features.
    - Helps ensure the feature extractor learns domain-invariant features by reversing the gradient
      through the `grad_reverse` function during backpropagation.

    Architecture:
    -------------
    - Identical in structure to the task classification head:
        - Flatten the input feature map
        - Apply dropout for regularization
        - Feed through two fully connected layers with GELU activation and LayerNorm
        - Final linear layer outputs logits for domain classification

    Forward Pass:
    -------------
    1. The input `features` are first passed through `grad_reverse(features, lambda_)`:
        - When `lambda_ > 0`, it reverses gradients to confuse the domain classifier.
        - This encourages the feature extractor to produce domain-invariant embeddings, Which effectively encourages earlier layers to not memorize domain (subject) specific patterns.
        - When `lambda_ == 0`, gradients are blocked (no domain adaptation effect).
    2. The transformed features are passed through the classification network to produce domain logits.

    Args:
    -----
    class_num_domain (int): Number of domain classes (e.g., number of subjects).
    n_times (int): Temporal dimension of the feature map.
    dropout (float): Dropout probability for regularization.
    proj_dim (int): Channel dimension of the feature map.

    Returns:
    --------
    Tensor: Raw domain classification logits (before softmax).
    """
    def __init__(
            self,
            class_num_domain,
            n_times,
            dropout,
            proj_dim
            ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_times * proj_dim, class_num_domain * 6),
            nn.LayerNorm(class_num_domain * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num_domain * 6, class_num_domain)
        )

    def forward(self, features, lambda_):
        x = grad_reverse(features, lambda_)
        return self.net(x)



class ChannelWiseLayerNorm(nn.LayerNorm):

    """
    Applies Layer Normalization across the channel axis (instead of the last dimension).

    Purpose:
    --------
    - Standard `nn.LayerNorm` normalizes over the last dimension, but EEG and similar
      time-series data often have the shape [Batch, Channels, Time].
    - This wrapper normalizes per time step, across channels (i.e., axis=1),
      which aligns better with how channel-wise normalization is intended in this task.

    Input Shape:
    ------------
    - x: Tensor of shape [B, C, T]
        B = batch size  
        C = number of channels (to be normalized)  
        T = number of time steps

    Output:
    -------
    - Tensor of the same shape [B, C, T] with normalized values across the channel axis.

    Notes:
    ------
    - Internally transposes to [B, T, C] for normalization, then transposes back to [B, C, T].
    - Inherits from `nn.LayerNorm`, so accepts the same arguments like `eps`, and `elementwise_affine`.
    """

    def __init__(
            self,
            normalized_shape,
            eps=1e-5,
            elementwise_affine=True
            ):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = x.transpose(1, 2)                  
        x = super().forward(x)              
        x = x.transpose(1, 2)  
        return x

class TemporalModulator(nn.Module):

    """
    TemporalModulator: A Novel Modulation Block for EEG + Sensor Fusion

    Purpose:
    --------
    - Learns to modulate EEG signals based on auxiliary sensor inputs (e.g., gyro, accelerometer).
    - Inspired by attention-like gating mechanisms, it allows the model to emphasize or suppress
      parts of the EEG signal dynamically, based on context from non-EEG channels.

    Key Assumptions:
    ----------------
    - Input tensor has shape [B, C_total, T] where:
        C_total = eeg_ch_nums + sensor_ch_nums
    - **Sensor channels are ordered last**, so splitting works correctly.

    How It Works (Step-by-Step):
    ----------------------------
    1. **Split** the input into EEG channels and Sensor channels along the channel axis:
       - `eeg_in`: shape [B, eeg_ch_nums, T]
       - `sensor_in`: shape [B, sensor_ch_nums, T]

    2. **Sensor Path**:
       - Pass `sensor_in` through a 1D convolution to extract temporal features.
       - Normalize with `ChannelWiseLayerNorm`.
       - Apply dropout for regularization.
       - Pass through `Sigmoid` to create modulation **weights** ∈ [0,1].

    3. **Modulation & Residual Fusion**:
       - Perform element-wise multiplication of `eeg_in * weights` to apply modulation.
       - Add a residual connection: `out = eeg_in * weights + eeg_in`
         → Ensures stability and allows the model to fallback to raw EEG if modulation is unnecessary.

    Output:
    -------
    - Returns a tensor of shape [B, eeg_ch_nums, T] — the modulated EEG signal.

    Notes:
    ------
    - This is not a classic attention block, but rather a learned **feature-wise gate** using auxiliary data.
    - Works best when non-EEG signals are informative about noise or motion artifacts in EEG. Which is the case for this competition's dataset.
    """
    def __init__(
            self,
            eeg_ch_nums,
            sensor_ch_nums,
            kernel_size,
            dropout
            ):
        super().__init__()
        self.eeg_ch_nums = eeg_ch_nums
        self.sensor_ch_nums = sensor_ch_nums

        proj_dim = eeg_ch_nums
        self.sensor_conv = nn.Conv1d(sensor_ch_nums, proj_dim ,kernel_size=kernel_size,padding= "same", groups=1)
        self.sensor_norm = ChannelWiseLayerNorm(proj_dim)  
        self.sensor_dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        eeg_in, sensor_in = torch.split(x, [self.eeg_ch_nums, self.sensor_ch_nums], dim=1)

        x = self.sensor_conv(sensor_in)       
        x = self.sensor_norm(x)               
        weights = self.sigmoid(self.sensor_dropout(x))

        out = eeg_in * weights + eeg_in
        return out


class PointWiseConvolutionalBlock(nn.Module):
    """

    Purpose:
    --------
    A modular 1×1 convolution block designed feature transformation
    across channels, used as an early projection step in temporal models
    like EEG transformers or convolutional attention blocks.

    Structure:
    ----------
    - Conv1D (kernel size = 1): Projects `output_chs` to `proj_dim` across channels.
    - Channel-wise LayerNorm: Normalizes each channel independently across time.
    - GELU Activation: Applies a smooth non-linearity.
    - Dropout: Regularizes activations during training.

    Forward Input/Output:
    ---------------------
    - Input:  Tensor of shape [B, output_chs, T]
    - Output: Tensor of shape [B, proj_dim, T]

    Notes:
    ------
    - Because we use `kernel_size=1`, It makes this a **pointwise convolution**, meaning it mixes
      features across channels but preserves the temporal dimension.
    - Not depthwise — Uses Conv1d with kernel_size=1 to apply learned linear combinations across the input channels, independently at each time step.

    bind_to():
    ----------
    - A utility method to **bind the submodules** into a parent model under a given prefix.
    - Also binds the forward function under `parent.PointWiseConvolution` for easy access.
    - So when we want to call forward we call it via 'parent.PointWiseConvolution(x)'

    """
    def __init__(self,
                output_chs,
                proj_dim,
                dropout
                ):
        super().__init__() 

        self.conv = nn.Conv1d(output_chs, proj_dim, 1, padding= "same", groups=1)
        self.norm = ChannelWiseLayerNorm(proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self , x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

    def bind_to(self, parent, prefix: str = "patch"):
        setattr(parent, f"{prefix}_conv", self.conv)
        setattr(parent, f"{prefix}_norm", self.norm)
        setattr(parent, f"{prefix}_act", self.act)
        setattr(parent, f"{prefix}_dropout", self.dropout)

        parent.PointWiseConvolution = self.forward

class MTCFormer(nn.Module):
    def __init__(self,
                 depth,
                 kernel_size,
                 n_times,
                 chs_num,
                 eeg_ch_nums,
                 class_num,
                 class_num_domain,
                 modulator_kernel_size=5,
                 domain_dropout=0.5,
                 modulator_dropout=0.5,
                 mid_dropout=0.6,
                 output_dropout=0.4,
                 weight_init_std = 0.01,
                 weight_init_mean = 0.00,
                 ):
        

        """
        MTCFormer.

        This model is designed for EEG-based classification tasks in settings with domain shift (e.g., different subjects or sessions),
        and features a dual-head structure: one head for task prediction, and another for domain discrimination.

        Architecture Features:
        ----------------------
        - **Temporal Attention Modulation** (`TemporalModulator`): 
            Uses auxiliary sensor channels (e.g., accelerometer, gyroscope) to adaptively modulate EEG inputs.
        - **Pointwise Convolutional Block**:
            Projects EEG channels to a higher-dimensional representation using a 1×1 convolution (Conv1D with kernel_size=1),
            followed by normalization, GELU activation, and dropout.
        - **Convolutional Attention Block**:
            A depth-controlled stack (`depth`) of residual blocks combining LayerNorm, Conv1D, FeedForward (Linear) layers.
            This component captures **local temporal dependencies** efficiently.
        - **Task Classification Head**:
            A multilayer perceptron that predicts task labels (e.g., motor imagery class).
        - **Domain Classification Head**:
            Uses a **gradient reversal layer** to train on domain labels adversarially, helping the model generalize across domains.
            Controlled by `domain_lambda`. When `domain_lambda = 0`, **domain adaptation is disabled** (no gradients flow back).

        Forward Input:
        --------------
        - `x`: Tensor of shape `[batch_size, num_channels, time_points]`
            Input EEG + auxiliary sensor data. EEG channels must come first in the channel dimension.
        - `domain_lambda`: float
            Coefficient for controlling domain adaptation strength. If set to 0, no gradient reversal is applied.

        Forward Pass Steps:
        -------------------
        1. Input is split internally into EEG channels and sensor channels.
        2. Sensor channels are used to generate a temporal mask that modulates the EEG channels (`TemporalModulator`).
        3. EEG is projected to a higher dimension using a **pointwise convolution**.
        4. The projected signal is passed through a series of **convolutional attention blocks** (`depth` controls the number).
        5. The resulting shared features are fed into:
            - A **task classification head** for predicting class labels.
            - A **domain classification head**, where gradients are optionally reversed using `domain_lambda`.

        Outputs:
        --------
        - `label_output`: Tensor of shape `[batch_size, class_num]`
            Predicted task labels.
        - `domain_output`: Tensor of shape `[batch_size, class_num_domain]`
            Predicted domain (e.g., subject ID), only used during training.

        Dropout Parameters:
        -------------------
        - `modulator_dropout`: Dropout inside the attention modulation layer.
        - `mid_dropout`: Dropout used after pointwise conv and inside attention blocks.
        - `output_dropout`: Dropout before the final classification head.
        - `domain_dropout`: Dropout inside the domain classifier.

        Initialization:
        ---------------
        - All Conv1D and Linear layers are initialized with `Normal(mean=weight_init_mean, std=weight_init_std)`.

        Parameters:
        -----------
        - `depth` (int): Number of convolutional attention layers.
        - `kernel_size` (int): Kernel size for temporal conv in attention blocks.
        - `n_times` (int): Number of time points per sample.
        - `chs_num` (int): Total number of input channels (EEG + sensor).
        - `eeg_ch_nums` (int): Number of EEG channels (comes first in input).
        - `class_num` (int): Number of task classes.
        - `class_num_domain` (int): Number of domain classes (e.g., subjects).
        - `modulator_kernel_size` (int): Kernel size in the temporal modulator block.
        - `domain_dropout` (float): Dropout used in the domain head.
        - `modulator_dropout` (float): Dropout in the temporal modulator.
        - `mid_dropout` (float): Dropout in the pointwise conv and attention layers.
        - `output_dropout` (float): Dropout before final task classification.
        - `weight_init_std` (float): Std for Conv/Linear weight initialization.
        - `weight_init_mean` (float): Mean for Conv/Linear weight initialization.
        """
            
        super().__init__()
        sensor_ch_nums = chs_num - eeg_ch_nums
        self.temporal_attention = TemporalModulator(
            eeg_ch_nums,
            sensor_ch_nums,
            kernel_size=modulator_kernel_size,
            dropout=modulator_dropout
            )

        output_chs = eeg_ch_nums
        proj_dim = output_chs * 2


        PointWise_Conv = PointWiseConvolutionalBlock(
            output_chs,
            proj_dim,
            mid_dropout
            )
        PointWise_Conv.bind_to(self)

        self.transformer = ConvolutionalAttention(
            depth,
            proj_dim,
            n_times,
            kernel_size,
            mid_dropout
            )

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(output_dropout),
            nn.Linear(n_times * proj_dim, class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

        self.domain_classifier = DomainClassifier(
            class_num_domain=class_num_domain,
            n_times=n_times,
            dropout=domain_dropout,
            proj_dim=proj_dim
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=weight_init_mean, std=weight_init_std)


    def forward(self, x,domain_lambda):
        x = self.temporal_attention(x)       
        x = self.PointWiseConvolution(x)
        x_shared = self.transformer(x)  


        domain_output = self.domain_classifier(x_shared,domain_lambda)
        label_output =  self.mlp_head(x_shared)
        return label_output , domain_output

