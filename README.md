# SciMine Reproducibility Repository

## Folder Structure

- `src/`: Contains source code.
  - `data_preprocessing/`: Scripts for data download and preprocessing.
  - `ranking_scripts/`: Scripts for ranking algorithms.
- `data/`: Data files.
- `conda_env.yml`: Conda environment file.
- `README.md`: This file.

## Setup Instructions

1. **Create Conda Environment**
    ```bash
    conda env create -f conda_env.yml
    conda activate scimine_env
    ```
2. **Download and Preprocess Data**
    ```bash
    python src/data_preprocessing/download_preprocess_data.py
    ```
3. **Run Ranking Scripts**
    ```bash
    python src/ranking_scripts/ranking_script1.py
    python src/ranking_scripts/ranking_script2.py
    ```
## Adding Packages to Conda Environment

Write the package name and version in the `conda_env.yml` file under the `dependencies` section.

## Other commands

- **Remove Conda Environment**
    ```bash
    conda deactivate
    conda env remove -n scimine_env -all
    ```
- **Update Conda Environment**
    ```bash
    conda env update -f conda_env.yml
    ```

- **Specific torch for cuda 12.6**
    ```bash
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
    ```