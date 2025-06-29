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
│   └── README.md                 # Model architecture details and diagram
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
├── reproducible_env.yml          # Conda reproducible environment file
├── submission.csv                # Example submission file
└── README.md                     # Project documentation
```

## Main Components

- **Data Conversion:**
Easily convert your raw competition CSV files to the efficient `.fif` format for MNE processing using the `convert_csv_to_fif.py` script.
**Tip:** If your competition data is already in a folder named `data`, simply run the script without arguments—it will automatically use that directory.
  
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

   It is recommended to create your environment from the provided `reproducible_env.yml` for best reproducibility:

   ```sh
   conda env create -f reproducible_env.yml
   conda activate Competition_environment
   ```
    
   Alternatively, you can install dependencies directly with pip:

   ```sh
   pip install -r requirements.txt
   ```
   ==============================================================================
   ⚠️ WARNING:
   This project was tested using Python 3.10.8.
   We highly recommend using Conda to create and manage the environment 
   for full reproducibility. 
   Using only pip may result in version mismatches or CUDA incompatibilities.
   ==============================================================================
2. **Convert data:**

   ```sh
   python convert_csv_to_fif.py --competitions_data_directory <path_to_competition_data>
   ```

   If your data is in the `data` directory, you do not need to pass any arguments:

   ```sh
   python convert_csv_to_fif.py
   ```

3. **Train models:**

   ```sh
   python train/train_mi_model1.py
   python train/train_mi_model2.py
   python train/train_mi_model3.py
   python train/train_ssvep_model.py
   ```

4. **Run inference:**

   By default, inference will use the best model checkpoints. If you want to test models from scratch or use the most recently generated checkpoints, you can pass the `--predict_on_best_models` argument as `False`:

   ```sh
   python inference/inference.py
   # or, to use non-best checkpoints:
   python inference/inference.py --predict_on_best_models False
   ```

   The output will be saved as `submission.csv` (or `submission_regenerated_(non_best).csv` if using non-best checkpoints).

## Notes

- Model checkpoints are saved in the `checkpoints` directory.
- Custom datasets and data loaders are implemented in `utils/CustomDataset.py`.
- Training and evaluation utilities are in `utils/training.py`.
- **Model architecture details and diagram are available in [`model/README.md`](model/README.md).**


## Citation

If you use this codebase, please cite the original competition and the MTCFormer model.

---

**Authors:**
Mohammed Ahmed Metwally and the CodeCraft Team

For questions, please open an issue or contact the authors.
