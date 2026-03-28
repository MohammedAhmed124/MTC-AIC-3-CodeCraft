# MTC-AIC-3 Winning pipeline 2025 🥇

This repository provides the complete, end‑to‑end winning pipeline for the MTC-AIC-3 EEG competition. It covers data loading, preprocessing, model training, and inference for both Motor Imagery (MI) and SSVEP tasks using the custom MTCFormer architecture.

---

## Project Structure
```
├── data/ # Competition data
│ ├── MI/ # MI trial files (.fif or raw)
│ ├── SSVEP/ # SSVEP trial files
│ ├── train.csv # Training labels and metadata
│ ├── validation.csv # Validation labels and metadata
│ ├── test.csv # Test metadata (no labels)
│ └── sample_submission.csv # Submission format example
├── model/ # MTCFormer implementations and docs
│ ├── MTCformerV2.py # Version 2 of the MTCFormer (used for ssvep)
│ ├── MTCformerV3.py # Version 3 of the MTCFormer (used for MI)
│ └── README.md # Architecture details and diagram
├── train/ # Training scripts
│ ├── checkpoints/ # Saved model checkpoints
│ ├── train_mi_model1.py # Train MI model variant 
│ └── train_ssvep_model.py # Train SSVEP model
├── inference/ # Inference pipeline
│ └── inference.py # Load models, run on data, save predictions
├── utils/ # Utility modules
│ ├── loader.py # Trial-level data loader (handles CSV→tensor)
│ ├── preprocessing.py # Filtering, normalization, bad-segment marking
│ ├── augmentation.py # EEG augmentations: noise, shifting, warping, dropout
│ ├── training.py # train_model() loop, schedulers, checkpointing
│ ├── gradient_attack.py # FGSM adversarial training support
│ └── rank_ensemble.py # Rank‑averaging ensembling utilities
├── checkpoints/ # Top‑level folder for final best checkpoints
├── submission/ # Scripts/notebooks for preparing submissions
├── requirements.txt # Python dependencies
└── README.md # This file
```


## Main Components

### Data Loading  
`utils/loader.py` replaces the old CSV‑to‑FIF converter. It:
- Reads trial files CSV  
- Matches filenames to `train.csv` / `validation.csv`  
- Does feature engineering.
- filters trials automatically based on quality.
- returns a batched numpy array of shape (n_trials , n_channels , n_times)

### Preprocessing & Augmentation  
- **Preprocessing** (`utils/preprocessing.py`): band‑pass filtering, channel normalization etc.
- **Augmentation** (`utils/augmentation.py`): Gaussian noise, time shift, amplitude scaling, time warp, channel dropout.

### Model Definitions  
All architectures live in `model/`.  
- **MTCformerV2** and **MTCformerV3** implement convolutional‑attention blocks with sparse gating. V2 is used for ssvep while V3 performs better for MI.  
- See `model/README.md` for block diagrams and detailed layer descriptions.

### Training Loop  
`utils/training.py` provides `train_model()`, which handles:
- Loss computation (cross‑entropy, domain/adversarial losses)  
- Optimizer & scheduler steps  
- Early stopping, checkpoint saving, TensorBoard logging  
- Adversarial training via `utils/gradient_attack.py` (FGSM variants)  
- Domain adaptation (subject‑ID loss) can be toggled or scheduled.

Example scripts:  
- `train/train_mi_model1.py` (MI variant 1)  
- `train/train_ssvep_model.py` (SSVEP)

### Inference & Submission  
`inference/inference.py` loads saved checkpoints, runs on validation or test sets, and writes `submission.csv`. You can choose the “best” checkpoints or most recent snapshots via command‑line flags.


---

## Getting Started

1. **Install dependencies** (tested on Python 3.10):  
   ```bash
   conda create -n eeg_env python=3.10 -y
   conda activate eeg_env
   pip install -r requirements.txt
   ```


##  Train Models

You can either train all models fully on the complete dataset or quickly test the training pipeline on a small subset.

### 🔹 Option 1: Full Training

Run the full training loop for each model:

```bash
python train/train_mi_model1.py
python train/train_ssvep_model.py
```
### 🔹Option 2: Test the Training Pipeline  
   Run a minimal version of the pipeline using only the first 50 samples.  
   Useful for debugging or verifying that the pipeline runs correctly.

   ```bash
   python train/train_mi_model1.py --test_pipeline
   python train/train_ssvep_model.py --test_pipeline
   ```

   > **Note:** This mode is not intended for meaningful training or performance evaluation. It is only for verifying that the pipeline executes correctly on a small subset of the data.

## Run inference

   By default, inference will use the best model checkpoints. If you want to test models from scratch or use the most recently generated checkpoints, you can pass the `--predict_on_best_models` argument as `False`:

   ```sh
   python inference/inference.py
   # or, to use non-best checkpoints:
   python inference/inference.py --predict_on_best_models False
   ```
   **to make predictions with a new test dataset. please replace the files related to older test set with the newer test set.**

   The output will be saved as `submission.csv` (or `submission_regenerated_(non_best).csv` if using non-best checkpoints or freshly trained models).


##  **Computational Environment**

This project was developed and tested under the following setup:

- **Operating System:** Ubuntu 22.04 LTS  
- **Python Version:** 3.10.8  
- **Environment Manager:** Conda  
- **System RAM:** 32 GB  
- **GPU:** NVIDIA RTX 4070 (12 GB VRAM)  
- **CUDA Version:** 12.6  
- **cuDNN Version:** 9.5.1  

---
## **Project Recognition**
This repository secured **1st Place** in the **ICMTC AIC 3** Competition.
[LinkedIn](https://www.linkedin.com/posts/mohammed-a-metwally-aa626532b_appliedinnovationcenter-militarytechnicalcollege-activity-7357003725850247168-jjLx)
