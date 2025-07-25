import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import math
import argparse
import sys
from entmax import sparsemax, entmax15 , entmax_bisect

class SparsemaxLayer(nn.Module):
    """
    Sparsemax activation layer.

    Sparsemax is an alternative to softmax that encourages sparsity in the output
    (i.e., it outputs exact zeros for less relevant features).

    Equation:
        Given an input vector z ∈ ℝ^K, sparsemax is defined as the Euclidean projection 
        of z onto the probability simplex:

            sparsemax(z) = argmin_p ||p - z||^2
                            s.t.  p ∈ Δ^K

        where Δ^K = {p ∈ ℝ^K | p ≥ 0, Σp_i = 1}

    Compared to softmax:
        - Both produce outputs that sum to 1.
        - Softmax outputs are always dense (all values > 0).
        - Sparsemax can produce sparse outputs (some values = 0).
          and feature selection.

    Use in Deep Learning:
        - Introduced in "From Softmax to Sparsemax: A Sparse Model of Attention" (2016).
        - Became popular through the 2022 TabNet architecture, where it helped in learning
          sparse, interpretable masks over features at each decision step.

    Args:
        dim (int): The dimension along which to apply sparsemax (default: -1).
        """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return sparsemax(x, dim=self.dim)


class GatedConvolutionalBlock(nn.Module):

    """
    Gated Convolutional Block with a learnable sparse mask.

    This block processes time-series inputs (shape: B × C × T) using a 1D convolution, 
    and modulates the output using a learned attention mask generated in parallel.

    The mask is created from a separate convolutional path and passed through a 
    Sparsemax activation, which encourages sparsity (some mask weights are exactly 0).
    This mask is multiplied with a prior (if given), then used to gate the residual 
    connection from the input — meaning only certain parts of the input are allowed 
    to pass through and influence the final output. It also supports prior scaling logic to encourage diversity (Tabnet 2022).

    Main Path:
        - LayerNorm → Conv1D → GELU → Dropout → + gated residual

    Mask Path:
        - LayerNorm → Conv1D → Linear(pool across time) → Sparsemax → Dropout

    Args:
        proj_dim (int): Number of channels in the input and output.
        n_times (int): Number of time steps (input sequence length).
        kernel_size (int): Size of the convolution kernel over time.
        dropout (float): Dropout probability for both main and mask paths.
        k (int): Not used here, placeholder for future top-k masking logic.

    Inputs:
        x (Tensor): Input tensor of shape (B, C, T)
        prior (Tensor, optional): Prior mask of shape (B, C, 1). If None, defaults to 1.

    Returns:
        x (Tensor): Output tensor of shape (B, C, T), after gated convolution.
        mask_weights (Tensor): Learned sparse mask of shape (B, C, 1).
        prior (Tensor): The same prior used (either passed or default ones).
    """

    def __init__(
        self,
        proj_dim,
        n_times,
        kernel_size,
        dropout,
        ):
        super().__init__()
        self.norm_1 = nn.LayerNorm(n_times) 
        self.conv=nn.Conv1d(proj_dim, proj_dim, kernel_size=kernel_size, padding="same", groups=1)

        self.activation_1 = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)

        self.norm_2 = nn.LayerNorm(n_times)


        self.mask_layer_norm = nn.LayerNorm(n_times) 
        self.mask_conv = nn.Conv1d(proj_dim, proj_dim, kernel_size=kernel_size, padding="same", groups=1)
        self.pooler = nn.Linear(n_times , 1)
        self.mask_sigmoid = SparsemaxLayer(1)
        self.mask_dropout = nn.Dropout(dropout)

    def forward(self, x , prior=None):  

        """
        Forward pass through the Gated Convolutional Block.

        This method processes the input `x` in two parallel paths:
        
        1. **Mask Generation Path**:
            - Learns a gating mask over the time dimension using a convolution,
            pooling layer, and Sparsemax activation.
            - The output is a sparse, normalized attention mask (`mask_weights`) 
            shaped (B, C, 1), indicating how much each channel should be allowed to 
            pass through.
            - A `prior` tensor can be passed in to scale the learned mask. This is 
            useful when stacking multiple such blocks (e.g., in TabNet), where 
            each block wants to focus on *different* parts of the input.
            → In TabNet, this promotes **mask diversity** by reducing overlap 
                between feature selections across steps.

        2. **Main Processing Path**:
            - The input is normalized, convolved, passed through GELU and dropout.
            - A residual connection is added, but it's gated by `(mask_weights * prior)`,
            so only the selected parts of the input are allowed to contribute.

        Args:
            x (Tensor): Input tensor of shape (B, C, T), where:
                        B = batch size, C = channels, T = time steps.
            prior (Tensor, optional): A tensor of shape (B, C, 1) used to scale 
                                    the learned mask. If None, defaults to all ones.

        Returns:
            x (Tensor): The processed output tensor of shape (B, C, T).
            mask_weights (Tensor): The sparse, learned gating mask of shape (B, C, 1).
            prior (Tensor): The scaling mask that was used (either passed or default ones).
        """
        mask_path = self.mask_layer_norm(x)
        mask_path = self.mask_conv(mask_path)
        mask_weigths = self.mask_sigmoid(self.pooler(mask_path))
        mask_weigths = self.mask_dropout(mask_weigths)
        if prior is None:
            prior = torch.ones_like(mask_weigths)

        residual = x * (mask_weigths * prior)
        
        x = self.norm_1(x)

        x = self.conv(x)

        x = self.activation_1(x)
        x = self.dropout_1(x)
        x = x + residual   

        return x   ,  mask_weigths , prior     

class ConvolutionalAttentionBlock(nn.Module):

    """
    Multi-branch convolutional attention block with sparse gating and prior-guided diversity.

    This block processes the input through three sequential GatedConvolutionalBlocks.
    Each block learns a sparse attention mask (via Sparsemax) that decides which
    channels/timesteps to focus on.

    To encourage **diversity between branches**, each subsequent block receives a 
    dynamically updated `prior` mask that discourages re-selecting features already 
    focused on by earlier blocks. This technique is inspired by TabNet's sparse 
    feature selection mechanism.

    The attention process looks like this:

        Let:
            x  : input tensor of shape (B, C, T)
            M₁ : first learned mask
            M₂ : second learned mask
            M₃ : third learned mask
            P₀ : initial prior (default = all ones)

        The priors are updated step-by-step as:

            P₁ = P₀ * (1 - M₁)
            P₂ = P₁ * (1 - M₂)
            P₃ = P₂ * (1 - M₃)

        This encourages each mask M₁, M₂, M₃ to focus on different (non-overlapping) parts.

    After the three gated convolutions, the outputs are averaged and passed through
    a small feedforward layer with GELU and dropout, with a final residual connection.

    ---
    Args:
        proj_dim (int): Number of channels in the input and output (C).
        n_times (int): Number of time steps (T).
        kernel_size (int): Temporal convolution kernel size.
        dropout (float): Dropout rate used throughout the block.
        k (int): Hidden size for the feedforward layer (used as bottleneck).

    ---
    Inputs:
        x (Tensor): Input tensor of shape (B, C, T).
        prior (Tensor, optional): Optional prior mask of shape (B, C, 1). Defaults to ones.

    ---
    Returns:
        x (Tensor): Output tensor after attention and feedforward (B, C, T).
    """
    def __init__(
        self,
        proj_dim,
        n_times,
        kernel_size,
        dropout,
        k,
    ):
        super().__init__()
        
        
        self.gated_conv_1 = GatedConvolutionalBlock(
            proj_dim=proj_dim,
            n_times=n_times,
            kernel_size=kernel_size,
            dropout=dropout, 
        )
        self.gated_conv_2 = GatedConvolutionalBlock(
            proj_dim=proj_dim,
            n_times=n_times,
            kernel_size=kernel_size,
            dropout=dropout,
        )


        
        self.norm_2 = nn.LayerNorm(n_times)
        self.feedforward = nn.Sequential(
            nn.Linear(n_times, k, bias=False),
            nn.Linear(k, n_times)
        )
        self.activation_2 = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, prior=None):  
        x1 , x1_weights , prior_x1 = self.gated_conv_1(x, prior=prior)

        prior_next = prior_x1*(1-x1_weights.detach())

        x2 , x2_weights , prior_x2= self.gated_conv_2(x, prior=prior_next)

        # prior_next= prior_x2*(1-x2_weights.detach())

        x = (x1 + x2 ) / 2.0

        self.last_gates = torch.stack([
        x1_weights.detach(), 
        x2_weights.detach(), 
    ])

        residual = x
        x = self.norm_2(x)
        x = self.feedforward(x)
        x = self.activation_2(x)
        x = self.dropout_2(x)
        x = x + residual

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
            dropout,
            k
            ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvolutionalAttentionBlock(
                proj_dim,
                n_times,
                kernel_size,
                dropout,
                k
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
        red=10
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=n_times // red),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear((n_times // red) * proj_dim, class_num_domain * 6),
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

        self.temporal_weights = weights.detach().clone()
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
    def __init__(
            self,
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
    def __init__(
        self,
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
        k = 30,
        projection_dimention = 2,
        seed = None,
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
        self.seed = seed
        sensor_ch_nums = chs_num - eeg_ch_nums
        self.temporal_attention = TemporalModulator(
            eeg_ch_nums,
            sensor_ch_nums,
            kernel_size=modulator_kernel_size,
            dropout=modulator_dropout
            )

        output_chs = eeg_ch_nums
        proj_dim = output_chs * projection_dimention


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
            mid_dropout,
            k = k
            )
        red = 10
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=n_times // red),
            nn.Flatten(),
            nn.Dropout(output_dropout),
            nn.Linear((n_times // red) * proj_dim, class_num * 6),
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

        self.init_weights_with_seed(seed)


    def forward(
            self,
            x,
            domain_lambda=0.0
            ):
        

        x = self.temporal_attention(x)       
        x = self.PointWiseConvolution(x)
        x_shared = self.transformer(x)  


        domain_output = self.domain_classifier(x_shared,domain_lambda)
        label_output =  self.mlp_head(x_shared)
        return label_output , domain_output


    def init_weights_with_seed(self, seed=None,reset_everything=False):
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if 'temporal_attention' in name:
                    # Sigmoid -> Xavier init is best
                    nn.init.xavier_normal_(m.weight)
                else:
                    
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif hasattr(m, 'reset_parameters'):
                if reset_everything:
                    m.reset_parameters()

        if seed is not None:
            torch.set_rng_state(rng_state)
