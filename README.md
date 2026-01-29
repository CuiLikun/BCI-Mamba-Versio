# EEG-ATCNet: Attention Temporal Convolutional Network for EEG-based Motor Imagery Classification

This repository contains the implementation of **ATCNet** (Attention Temporal Convolutional Network) for EEG-based Motor Imagery classification, as proposed by **Hamdi Altaheri** et al.

The code is designed to work with the **BCI Competition IV-2a** dataset.

## 📋 Requirements

The project requires the following Python libraries:

- Python 3.x
- TensorFlow (Keras)
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

You can install the dependencies using:

```bash
pip install tensorflow numpy scipy matplotlib scikit-learn
```

## 💾 Dataset

The project uses the **BCI Competition IV-2a** dataset.

1. Download the dataset from [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets).
2. Place the dataset files (e.g., `A01T.gdf`, `A01E.gdf`, etc. or `.mat` versions depending on what `preprocess.py` expects) in a `data/` folder (or configure the path in the script).
   _Note: Check `preprocess.py` for exact data format and path expectations._

## 🚀 Usage

To train and test the model:

1. Open `main_TrainTest.py` or `main_TrainValTest.py`.
2. Configure the `dataset_conf` and `train_conf` dictionaries at the beginning of the format (or `__main__` block) if necessary (e.g., to point to your data directory).
3. Run the script:

```bash
python main_TrainTest.py
```

## 📂 File Structure

- **`main_TrainTest.py`**: Main script for training and testing the model (Subject-Specific evaluation).
- **`main_TrainValTest.py`**: Main script for training, validation, and testing.
- **`models.py`**: Contains the implementation of the **ATCNet** model and other comparison models.
- **`preprocess.py`**: Functions for loading and preprocessing the BCI Competition IV-2a dataset.
- **`preprocess_HGD.py`**: Functions for loading High Gamma Dataset (HGD).
- **`attention_models.py`**: Implementation of various attention mechanisms.
- **`results/`**: Directory where training logs, best models, and confusion matrices are saved.

## 📊 Results (BCI Competition IV-2a)

Test results for 9 subjects (Accuracy & Kappa):

|   Subject   | Accuracy (%) |   Kappa   |
| :---------: | :----------: | :-------: |
|      1      |    81.25     |   0.750   |
|      2      |    59.38     |   0.458   |
|      3      |    87.85     |   0.838   |
|      4      |    68.75     |   0.583   |
|      5      |    75.69     |   0.676   |
|      6      |    56.25     |   0.417   |
|      7      |    81.25     |   0.750   |
|      8      |    80.56     |   0.741   |
|      9      |    82.99     |   0.773   |
| **Average** |  **74.88**   | **0.665** |

> _Note: Results obtained using NVIDIA GeForce RTX 4060 Laptop GPU._

## 📜 License

This project is licensed under the **Apache License 2.0**.
Copyright (C) 2022 King Saud University, Saudi Arabia.

## 🔗 Reference

If you use this code, please credit the original authors:
**Hamdi Altaheri**, King Saud University.
