# Med-BiasX
PyTorch code for MICCAI 2025 paper "Med-BiasX: Robust Medical Visual Question Answering with Language Biases".  
  
# Requirements
* python 3.10
* pytorch 2.6.0
* torchvision 0.21.0

# Installation

**Option 1 – Miniconda / Anaconda (recommended)**

Create and activate the environment in one step using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate Med-BiasX
```

**Option 2 – Create environment manually, then pip-install**

```bash
conda create -n Med-BiasX python=3.10
conda activate Med-BiasX
pip install -r requirements.txt
```

## Data Setup
keep file in the folders set by `utils/config.py`.
More details please refer to [https://github.com/bvyih3/CEDO/data](https://github.com/bvyih3/CEDO/tree/main/data)

## Preprocessing

The preprocessing steps are as follows:

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```

2. process answers and question types, and generate the frequency-based margins:
    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```

## Model training instruction
```
    python main_arcface.py --name test-VQA --gpu 0 --dataset DATASET
   ```
Set `DATASET` to a specfic dataset such as `slake`, `slake-cp`, `vqa-rad`, and `vqa-rad-cp`. 

## Model evaluation instruction
```
    python main_arcface.py --name test-VQA --test
   ```
