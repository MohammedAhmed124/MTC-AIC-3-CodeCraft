# MTC-AIC EEG Competition Pipeline

This repository contains a complete pipeline for the MTC-AIC EEG competition, including data conversion, preprocessing, model training, and inference for both Motor Imagery (MI) and SSVEP tasks.

## Project Structure

```
.
├── convert_csv_to_fif.py         # Convert raw CSV EEG data to MNE .fif format
├── inference/
│   └── inference.py              # Full inference pipeline for MI and SSVEP
├── model/
│   └── MTCformerV3.py            # MTCFormer model definition
├── data/
│    ├── MI/
|    ├── SSVEP/
│    ├── train.csv
│    ├── validation.csv
│    ├── test.csv
│    └── sample_submission.csv 
├── train/
│   ├── train_mi_model1.py        # Train MI model variant 1
│   ├── train_mi_model2.py        # Train MI model variant 2 (adversarial)
│   ├── train_mi_model3.py        # Train MI model variant 3
│   └── train_ssvep_model.py      # Train SSVEP model
├── utils/
│   ├── augmentation.py           # Data augmentation utilities
│   ├── CustomDataset.py          # Custom EEGDataset class
│   ├── extractors.py             # Data extraction helpers
│   ├── gradient_attack.py        # Adversarial attack utilities
│   ├── preprocessing.py          # Preprocessing functions
│   ├── rank_ensemble.py          # Ensemble ranking utilities
│   └── training.py               # Training and prediction utilities
├── checkpoints/                  # Model checkpoints (created during training)
├── requirements.txt              # Python dependencies
├── submission.csv                # Example submission file
└── README.md                     # Project documentation
```

## Main Components

- **Data Conversion:**
  Use `convert_csv_to_fif.py` to convert raw competition CSV files into `.fif` format for efficient processing with [MNE](https://mne.tools/).

- **Preprocessing & Augmentation:**
  Utilities in `utils/preprocessing.py` and `utils/augmentation.py` handle signal cleaning and data augmentation.

- **Model Training:**
  - MI models: `train/train_mi_model1.py`, `train/train_mi_model2.py`, `train/train_mi_model3.py`
  - SSVEP model: `train/train_ssvep_model.py`
  - All models use the `MTCFormer` architecture (`model/MTCformerV3.py`).

- **Inference:**
  The full inference pipeline is in `inference/inference.py`, producing a submission file.

## Getting Started

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2. **Convert data:**

   ```sh
   python convert_csv_to_fif.py --competitions_data_directory <path_to_competition_data>
   ```

3. **Train models:**

   ```sh
   python train/train_mi_model1.py
   python train/train_mi_model2.py
   python train/train_mi_model3.py
   python train/train_ssvep_model.py
   ```

4. **Run inference:**

   ```sh
   python inference/inference.py
   ```

   The output will be saved as `submission.csv`.

## Notes

- Model checkpoints are saved in the `checkpoints` directory.
- Custom datasets and data loaders are implemented in `utils/CustomDataset.py`.
- Training and evaluation utilities are in `utils/training.py`.

## Citation

If you use this codebase, please cite the original competition and the MTCFormer model.

---

**Authors:**
Mohammed Ahmed Metwally and the CodeCraft Team

For questions, please open an issue or contact the authors.
