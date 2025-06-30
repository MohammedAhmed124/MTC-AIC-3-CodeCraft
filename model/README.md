# MTCFormer:

![MTCFormer Architecture](Arch.jpeg)

## Overview
We are happy to introduce MTCFormer, our custom-designed architecture tailored for EEG classification tasks. Built with domain adaptation in mind.
`MTCFormerV3.py` implements the MTCFormer model, a deep learning architecture designed for EEG-based classification tasks, especially in scenarios with domain shift (e.g., different subjects or sessions). The model is built for both task prediction and domain adaptation, making it robust for real-world EEG competitions and research.

## Key Architecture Features

- **Temporal Modulation:** Learns to modulate EEG signals using auxiliary sensor channels (e.g., accelerometer, gyroscope) to suppress noise and enhance relevant patterns.
- **Pointwise Convolutional Projection:** Projects EEG features to a higher-dimensional space for richer representations.
- **Convolutional Attention Blocks:** Stacked blocks capture local temporal dependencies efficiently, inspired by SSVEPFormer.
- **Dual-Head Output:**
  - **Task Classification Head:** Predicts the main task label (e.g., motor imagery class).
  - **Domain Classification Head:** Uses a gradient reversal layer for adversarial domain adaptation, encouraging domain-invariant features.
- **Highly Configurable:** Depth, kernel sizes, dropout rates, and more are easily adjustable.


#  How to Use `MTCFormer`

This guide walks you through using the `MTCFormer` architecture for EEG-based classification tasks, with optional **domain adaptation** and **adversarial training**.

---

##  Step 1: Import Required Components

```python
from model.MTCformerV3 import MTCFormer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
```

---

##  Step 2: Initialize the Model

```python
model = MTCFormer(
    depth=2,                   # Number of convolutional attention blocks
    kernel_size=5,             # Kernel size for both conv-attention and pointwise conv
    n_times=600,               # Input time points
    chs_num=7,                 # Total channels = EEG + sensor (e.g. 4 EEG + 3 sensors)
    eeg_ch_nums=4,             # EEG channels only (must come first)
    class_num=2,               # Number of task classes (e.g., MI left/right)
    class_num_domain=30,       # Number of domain classes (e.g., subjects)
    modulator_dropout=0.3,     # Dropout inside the modulation layer
    mid_dropout=0.5,           # Dropout after pointwise conv and inside conv-attention
    output_dropout=0.5,        # Dropout before the classification head
    weight_init_mean=0,        # Mean for normal initialization
    weight_init_std=0.5        # Std for normal initialization
).to(device)
```

>  **Note**: The model's `forward` takes two inputs:
> - `x`: A tensor of shape `[batch_size, chs_num, n_times]`
> - `domain_lambda`: A float controlling how strongly the domain classifier contributes to training (0 disables it).

---

##  Step 3: Set Up Optimizer, Loss, and Scheduler

```python
optimizer = Adam(model.parameters(), lr=0.002)
criterion = CrossEntropyLoss(reduction="none")
scheduler = MultiStepLR(optimizer, milestones=[70], gamma=0.1)
```

---

##  Step 4: Train the Model

```python

from utils.training import train_model
save_path = "path/to/save/checkpoints"
train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    window_len=WINDOW_LEN,
    original_val_labels=orig_labels_val_torch,
    n_epochs=250,
    patience=100,
    scheduler=scheduler,
    domain_lambda=0.01,             # Enables domain adaptation (0 disables)
    lambda_scheduler_fn=None,       # Optionally modify domain_lambda over time
    adversarial_steps=1,            # Steps for PGD adversarial attack
    adversarial_epsilon=0.05,       # Max perturbation for adversarial noise
    adversarial_alpha=0.005,        # Step size for PGD attack
    adversarial_training=True,      # Enable PGD-style adversarial training
    save_best_only=True,            # Save only best-performing model
    save_path=save_path,
    n_classes=2,
    device=device
)

```


---

##  Adversarial Training Details

If `adversarial_training=True`, `train_model` adds **PGD-style attacks** on input during training. The full theory is available in the `system_description.pdf`

The adversarial attack is implemented in: `utils/gradient_attack.py`



- `adversarial_steps`: Number of PGD steps
- `adversarial_epsilon`: Max allowed perturbation
- `adversarial_alpha`: Step size per PGD iteration

---

##  Domain Adaptation Details

If `domain_lambda > 0`, `train_model` enables **domain adaptation** using a **gradient reversal layer** to help the model generalize across subjects or sessions. This mechanism is useful for combating domain shift.

Domain adaptation is handled inside the model’s forward pass:

```python
task_outputs, domain_outputs = model(inputs, domain_lambda)
```

The second argument, `domain_lambda`, is passed **directly through `train_model`**:

```python
train_model(..., domain_lambda=0.01, ...)
```

###  What `domain_lambda` Controls

- If `domain_lambda = 0`:  
  - Domain adaptation is **disabled**  
  - No gradient reversal is applied  
  - Domain head is **bypassed** (no backward flow)

- If `domain_lambda > 0`:  
  - **Gradients are reversed** for the domain classification loss  
  - Model learns domain-invariant features
  - Domain head contributes to total loss, weighted by `domain_lambda`

---

##  Code Structure References

- Model code: `model/MTCformerV3.py`
- Training logic: `utils/training.py`
- PGD adversarial attack: `utils/gradient_attack.py`

---

##  Model Highlights 

- **Temporal Attention Modulation**: Uses auxiliary sensors to modulate EEG signals.
- **Pointwise Convolution Block**: Projects EEG to higher-dimensional feature space.
- **Conv-Attention Blocks**: Capture local temporal patterns (`depth` controls number).
- **Task Head**: Predicts task class.
- **Domain Head**: Adversarial domain classifier using gradient reversal (`domain_lambda`).
- **Dropouts**:
  - `modulator_dropout`: inside the sensor modulator
  - `mid_dropout`: between main layers
  - `output_dropout`: before task classifier
  - `domain_dropout`: inside domain classifier (if used)

---

>  Remember: EEG channels must come **first** in the channel dimension of the input.

## File Structure

- `ConvolutionalAttentionBlock`: Local temporal attention via convolution.
- `ConvolutionalAttention`: Stacked attention blocks.
- `TemporalModulator`: Sensor-driven modulation of EEG.
- `PointWiseConvolutionalBlock`: 1x1 conv projection.
- `DomainClassifier`: Adversarial domain head.
- `MTCFormer`: Main model class.

## Reference

- Inspired by SSVEPFormer and DANN (Domain-Adversarial Neural Network).

---

**For more details, see the code comments in `MTCformerV3.py`.**  
**For more theoritical details check system_description.pdf**
