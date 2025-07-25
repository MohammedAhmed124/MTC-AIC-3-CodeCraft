## 🛠️ `utils/` Directory

The `utils/` folder contains all the modular building blocks that handle core preprocessing, data augmentation, training loops, and ensembling. These components are tightly integrated with the rest of the competition pipeline and designed for extensibility.

---

### `loader.py`

**Trial-Level Data Loader**

- Responsible for loading EEG trial data and converting it into a machine learning–ready format:  
  `(n_trials, n_channels, n_timesteps)`.
- Uses `polars` instead of pandas for efficient data loading.
- Automatically excludes metadata channels like `Battery`, `Counter`, and `Time`.
- Computes L2 norms for the accelerometer and gyroscope channels and retains only these norms.
- Channel selection depends on the task:
  - **MI tasks** retain: `C3`, `C4`, `CZ`, `PZ`
  - **SSVEP tasks** retain: `PO7`, `PO8`, `POZ`, `OZ`
  - Both tasks also include: `L2-norm(Acc)`, `L2-norm(Gyro)`, and the `Validation` signal
- Extracts subject IDs and class labels from filenames and CSV metadata.
- It does automatic `quality filtering` based on `L2-norm(Acc), L2-norm(Gyro) and validation means`.
---

### `preprocessing.py`

**Signal Filtering & Normalization**

- Applies filtering, normalization, cropping etc.
- Uses efficient vectorized processing to prepare data for downstream modeling.
- Handles windowing and quality filtering when needed.

---

### `augmentation.py`

**Data Augmentation for EEG**

- Implements EEG-specific augmentations:
  - Gaussian noise  
  - Circular time shifting  
  - Signal Scaling
  - Time warping  
  - Random dropout  
.

---

### `training.py`

**Training Utilities**

- Contains the main `train_model()` function used in `train/train_mi_model1.py`, and `train/train_ssvep_model.py`.
- Handles:
  - Model training loop  
  - Checkpoint saving  
  - Evaluation metrics  
  - Learning rate scheduling  
  - Early stopping  
  - Optional adversarial training  
- Also includes prediction logic for inference-time model evaluation.

---

### `gradient_attack.py`

**Adversarial Robustness**

- Implements PGD-style (Projected Gradient Descent) adversarial attacks.
- Used to test and improve model robustness against input perturbations.
- Can be optionally enabled during training to produce adversarial examples for training.

---



