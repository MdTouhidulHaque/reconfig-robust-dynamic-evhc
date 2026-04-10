# Reconfiguration-Robust Dynamic EVHC

Official repository for the paper:

**Reconfiguration-Robust Spatiotemporal Learning for Dynamic EV Hosting Capacity Estimation in Distribution Feeders**

## Repository purpose

This repository contains the code, data packaging layout, trained models, and evaluation assets of the paper.

## Repository layout

```text
reconfig-robust-dynamic-evhc/
├─ README.md
├─ LICENSE
├─ DATA_LICENSE.md
├─ CITATION.cff
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ .gitattributes
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ sample/
├─ scripts/
├─ models/
├─ results/
└─ docs/
```

## Data layout

- `data/raw/load_data.h5`: original load data used as the input time series source.
- `data/raw/dss_model/`: OpenDSS feeder model files.
- `data/interim/qsts/<scenario>/`: QSTS outputs and metadata.
- `data/processed/graph/<scenario>/`: graph topology tensors and metadata.
- `data/processed/ground_truth/<scenario>/`: per-bus ground-truth EV hosting capacity files.
- `data/processed/ml_ready/<scenario>/`: tensors used by the learning models.

Scenario names are standardized as:

- `base`
- `s1`
- `s2`
- `s3`
- `s4`
- `s5`

## Environment

The preprocessing pipeline depends on **Windows + OpenDSS COM**, because several scripts use `win32com.client` to control OpenDSS directly.

The training and evaluation scripts use **PyTorch** and **PyTorch Geometric**.

Create a fresh environment, then install the paper dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Recommended workflow

1. Put the feeder model in `data/raw/dss_model/`.
2. Put the original HDF5 load file in `data/raw/load_data.h5`.
3. Organize scenario-specific data folders.
4. Build or verify graph files in `data/processed/graph/`.
5. Build ML-ready tensors:
   ```powershell
   python .\scripts\04_build_ml_dataset.py
   ```
6. Train the proposed model:
   ```powershell
   python .\scripts\05_train_3d_ecgat.py
   ```
7. Train baselines:
   ```powershell
   python .\scripts\06_train_dnn_baseline.py
   python .\scripts\07_train_lstm_baseline.py
   ```
8. Run evaluation:
   ```powershell
   python .\scripts\08_evaluate_models.py
   ```

## Notes on large files

The full dataset and trained models exceed convenient GitHub size limits. The setup is as follows:

- GitHub: code, metadata, configuration, small sample files, and lightweight result files
- Zenodo: full data, trained models, and archival release



After creating a public GitHub release, archive that release with Zenodo and replace the placeholders below.

## Citation and archival release

Replace the placeholders in `CITATION.cff` after creating the public release.

GitHub repository: `REPLACE_WITH_GITHUB_URL`  
Zenodo DOI: `REPLACE_WITH_ZENODO_DOI`

## License split

- Code: MIT License
- Data, trained models, and result artifacts created for this project: CC BY 4.0

See `LICENSE` and `DATA_LICENSE.md`.


## Third-party assets

This repository includes or references certain third-party materials that are **not covered** by the repository’s MIT code license or the repository’s CC BY 4.0 license for original released materials.

In particular, the OpenDSS feeder model under `data/raw/dss_model/` and `data/raw/load_data.h5` originates from the Iowa Distribution Test System released by the original authors. Please see:

- `data/raw/dss_model/README_source.md`

Users of those files should follow the original source attribution and any applicable reuse terms.



## Reproducibility note for the paper

A suitable camera-ready sentence is:

> Code, data, and trained models for this study are publicly available at [GitHub link], with an archived release at [Zenodo DOI].

