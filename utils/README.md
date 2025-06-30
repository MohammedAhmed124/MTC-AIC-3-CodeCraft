# Utils Directory

The `utils` directory provides a suite of modular, reusable helper modules that streamline data processing, augmentation, and model training for EEG deep learning workflows. Each file is designed to be composable and extensible within the competition pipeline.

---

### augmentation.py

*Data Augmentation*

- Implements techniques such as adding Gaussian noise to improve model robustness and generalization.
- Useful for simulating real-world signal variability and preventing overfitting.

### CustomDataset.py

*Custom Dataset Loader*

- Defines the `EEGDataset` PyTorch class for efficient loading, batching, and optional augmentation of EEG data.
- Supports sample weights, subject labels, and flexible augmentation pipelines for both training and evaluation.

### extractors.py

*Feature & Label Extraction*

- Functions to extract subject labels and features from raw EEG files, including subject ID parsing from filenames.
- Ensures consistent subject mapping and feature engineering for downstream tasks.

### gradient_attack.py

*Adversarial Robustness*

- Utilities for adversarial attacks (e.g., Projected Gradient Descent) to test and improve model robustness.
- Enables adversarial training and evaluation, making models more resilient to input perturbations and noise.

### preprocessing.py

*Signal Preprocessing*

- Handles windowing, filtering, normalization, and artifact removal for EEG data.
- Includes parallelized routines for efficient batch processing, ensuring data is ready for model input.

### rank_ensemble.py

*Ensembling & Ranking*

- Implements rank-based ensemble methods for combining predictions from multiple models using rank averaging.
- Improves prediction accuracy and stability by leveraging the strengths of diverse models.

### training.py

*Training & Evaluation Utilities*

- Advanced training loops, evaluation routines, adversarial training, and prediction utilities for model development.
- Supports domain-adversarial training, early stopping, checkpointing, and flexible evaluation metrics.

---

These utilities modularize the pipeline, making it easier to preprocess data, augment datasets, train, evaluate, and ensemble models. They are designed for extensibility and can be adapted for other EEG or time-series machine learning projects.
