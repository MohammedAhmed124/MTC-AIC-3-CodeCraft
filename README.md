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

##  Main Components

**Data Conversion**

Easily convert your raw competition CSV files to the efficient `.fif` format for MNE processing using the [`convert_csv_to_fif.py`](convert_csv_to_fif.py) script. This conversion ensures compatibility with the rest of the EEG processing pipeline.

Under the hood, this script performs the following steps:

1. **Load Labels & Trial Metadata**  
   Loads metadata from `train.csv`, `validation.csv`, and `test.csv` to determine where each trial is stored and how to label it.

2. **Preprocess Each Trial**  
   For each trial:
   - Reads the raw EEG `.csv` files.
   - Drops irrelevant columns (`Time`, `Battery`, etc.).
   - Computes L2 norms of accelerometer and gyroscope signals.
   - Retains EEG + norm features + `Validation` signal.
   
3. **Quality Filtering**  
   For training and validation data:
   - Skips trials with poor quality (low validation scores or excessive movement).
   - Keeps all test data (no skipping).

4. **Trial Extraction**  
   Extracts individual trial segments from the full session using the provided `extract_trial()` function.

5. **Convert to MNE `.fif` Format**  
   Converts each trial to an `mne.io.RawArray` object with:
   - Proper EEG + misc channel labeling.
   - Embedded metadata (subject ID, validation quality, etc.).
   - Annotations marking bad segments (where `Validation == 0`).

6. **Save Output**
   - Saves each processed trial as a `.fif` file under:
     ```
     data_fif/
       ├── train/
       ├── validation/
       └── test/
           ├── MI/
           └── SSVEP/
     ```
### 🔄 **Data Conversion**

Easily convert your raw competition CSV files to the efficient `.fif` format for MNE processing using the [`convert_csv_to_fif.py`](convert_csv_to_fif.py) script. This conversion ensures compatibility with the rest of the EEG processing pipeline.

Under the hood, this script performs the following steps:

1. **Load Labels & Trial Metadata**  
   Loads metadata from `train.csv`, `validation.csv`, and `test.csv` to determine where each trial is stored and how to label it.

2. **Preprocess Each Trial**  
   For each trial:
   - Reads the raw EEG `.csv` files.
   - Drops irrelevant columns (`Time`, `Battery`, etc.).
   - Computes L2 norms of accelerometer and gyroscope signals.
   - Retains EEG + norm features + `Validation` signal.
   
3. **Quality Filtering**  
   For training and validation data:
   - Skips trials with poor quality (low validation scores or excessive movement).
   - Keeps all test data (no skipping).

4. **Trial Extraction**  
   Extracts individual trial segments from the full session using the provided `extract_trial()` function.

5. **Convert to MNE `.fif` Format**  
   Converts each trial to an `mne.io.RawArray` object with:
   - Proper EEG + misc channel labeling.
   - Embedded metadata (subject ID, validation quality, etc.).
   - Annotations marking bad segments (where `Validation == 0`).

6. **Save Output**
   - Saves each processed trial as a `.fif` file under:
     ```
     data_fif/
       ├── train/
       ├── validation/
       └── test/
           ├── MI/
           └── SSVEP/
     ```


- **Preprocessing & Augmentation:**  
  Utilities in [`utils/preprocessing.py`](utils/preprocessing.py) and [`utils/augmentation.py`](utils/augmentation.py) handle signal cleaning and data augmentation.

- **Model Training:**  
  - MI models: [`train/train_mi_model1.py`](train/train_mi_model1.py), [`train/train_mi_model2.py`](train/train_mi_model2.py), [`train/train_mi_model3.py`](train/train_mi_model3.py)  
  - SSVEP model: [`train/train_ssvep_model.py`](train/train_ssvep_model.py)  
  - All models use the `MTCFormer` architecture defined in [`model/MTCformerV3.py`](model/MTCformerV3.py). with a tutorial at [`model/README.md`](model/README.md).

- **Inference:**  
  The full inference pipeline is in [`inference/inference.py`](inference/inference.py), producing the final submission file.


## Getting Started


1. **Install dependencies:**

   ⚠️ WARNING:
   This project was tested using Python 3.10.8.
   We highly recommend using Conda to create and manage the environment 
   for full reproducibility. 
   Using only pip may result in version mismatches or CUDA incompatibilities.
   
   Option 1: It is recommended to create your environment from this sequence of commands for best reproducibility:
   ```sh
    conda create -n Competition_environment python=3.10
    conda activate Competition_environment
    pip install -r requirements.txt
   ```
    
   Option 2: If you're not a conda user, you can install dependencies directly with pip:

   ```sh
   pip install -r requirements.txt
   ```
  

   Option 3: 
   ```sh
   conda env create -f environment.yml
   conda activate Competition_environment
   ```

   
3. **Convert data:**

   ```sh
   python convert_csv_to_fif.py --competitions_data_directory <path_to_competition_data>
   ```
  
   We recomment adding competition's data inside `data` directory, so you do not need to pass any arguments:

   ```sh
   python convert_csv_to_fif.py
   ```

4. **Train models:**

   Option 1: Full Training  
   Train on the entire dataset.  
   ⚠️ *This may crash on systems with less than 16 GB of RAM.*

   ```bash
   python train/train_mi_model1.py
   python train/train_mi_model2.py
   python train/train_mi_model3.py
   python train/train_ssvep_model.py
   ```

   Option 2: Test the Training Pipeline  
   Run a minimal version of the pipeline using only the first 50 samples.  
   Useful for debugging or verifying that the pipeline runs correctly.

   ```bash
   python train/train_mi_model1.py --test_pipeline
   python train/train_mi_model2.py --test_pipeline
   python train/train_mi_model3.py --test_pipeline
   python train/train_ssvep_model.py --test_pipeline
   ```

   > **Note:** This mode is not intended for meaningful training or performance evaluation. It is only for verifying that the pipeline executes correctly on a small subset of the data.



5. **Run inference:**

   By default, inference will use the best model checkpoints. If you want to test models from scratch or use the most recently generated checkpoints, you can pass the `--predict_on_best_models` argument as `False`:

   ```sh
   python inference/inference.py
   # or, to use non-best checkpoints:
   python inference/inference.py --predict_on_best_models False
   ```

   The output will be saved as `submission.csv` (or `submission_regenerated_(non_best).csv` if using non-best checkpoints).


## 5. **Computational Environment**

This project was developed and tested under the following setup:

- **Operating System:** Ubuntu 22.04 LTS  
- **Python Version:** 3.10.8  
- **Environment Manager:** Conda  
- **System RAM:** 32 GB  
- **GPU:** NVIDIA RTX 4070 (12 GB VRAM)  
- **CUDA Version:** 12.6  
- **cuDNN Version:** 9.5.1  

---

###  Key Libraries

####  Core Scientific Stack
- `numpy==1.26.4`  
- `pandas==2.3.0`  
- `scipy==1.15.2`  
- `scikit-learn==1.2.2`  
- `joblib`  
- `tqdm`  
- `glob2`  
- `ipython==8.37.0`  

####  PyTorch Stack
- `torch==2.7.1`  
- `torchvision==0.22.1`  
- `torchaudio==2.7.1`  
- `torchinfo==1.8.0`  
- `torchview==0.2.7`  
- `torchvis==0.2.0`  
- `torchviz==0.0.3`  

####  EEG / MNE Tools
- `mne==1.9.0`  
- `mne-features==0.3`  

---

To replicate this environment, use the provided `environment.yml` or `requirements.txt`


## Notes

- Model best checkpoints are saved in the [`checkpoints`](checkpoints) directory `(the one at the top level directory)` .
- Custom datasets and data loaders are implemented in [`utils/CustomDataset.py`](utils/CustomDataset.py).
- Training and evaluation utilities are in [`utils/training.py`](utils/training.py).  
- **Model architecture details, tutorial and diagram are available in [`model/README.md`](model/README.md).**
- **preprocessing utilities available in [`utils/preprocessing.py`](utils/preprocessing.py)**
- **augmentation logic available in [`utils/augmentation.py`](utils/augmentation.py)**
- **adversarial gradient attack logic in [`utils/gradient_attack.py`](utils/gradient_attack.py)**
- **Rank Averaging for Ensembling logic in [`utils/rank_ensemble.py`](utils/rank_ensemble.py)**
- **All the theory behind the solution in [`system_description.pdf`](system_description.pdf)**
## Citation

If you use this codebase, please cite the original competition and the MTCFormer model.

---

**Authors:**
Mohammed Ahmed Metwally and the CodeCraft Team

For questions, please open an issue or contact the authors.
