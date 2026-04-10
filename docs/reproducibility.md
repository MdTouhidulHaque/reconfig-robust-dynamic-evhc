# Reproducibility guide

This document explains how to reproduce the main results in the public repository.

## Repository layout

The repository is organized into five main parts:

- `data/raw/`: raw load data and OpenDSS model files
- `data/interim/`: intermediate simulation outputs
- `data/processed/`: graph tensors, ground-truth EV hosting capacity labels, and ML-ready tensors
- `models/`: trained model checkpoints and scaler files
- `results/`: evaluation tables and figures prepared for the paper or release

## Contents

To make the work reproducible, the release should include:

### Required
- all source code under `scripts/`
- `requirements.txt`
- `configs/paths.yaml`
- the OpenDSS feeder model under `data/raw/dss_model/`
- the raw or releasable load dataset under `data/raw/load_data.h5`
- processed graph files under `data/processed/graph/`
- ground-truth EV hosting capacity files under `data/processed/ground_truth/`
- ML-ready tensors under `data/processed/ml_ready/`

### Optional
- the best trained checkpoint and scaler for the proposed model:
  - `models/3d_ecgat/3d_ecgat_protocol_b_w24_best_state_dict.pt`
  - `models/3d_ecgat/scalers_node_edge_protocol_b.npz`
- the baseline checkpoints and scalers:
  - `models/dnn_baseline/best_dnn_protocol_b.pt`
  - `models/dnn_baseline/scaler_dnn_protocol_b.npz`
  - `models/lstm_baseline/best_lstm_protocol_b.pt`
  - `models/lstm_baseline/scaler_lstm_protocol_b.npz`

### Outputs
- the evaluation CSV file produced by the evaluation script
- the representative weekly overlay figures
- the node-MAE map figures

## Minimal reproducibility versus full reproducibility

### Minimal reproducibility
A reader should:
1. inspect the code,
2. access the data,
3. run the evaluation script on the released best checkpoint, and
4. reproduce the reported test metrics without retraining.

For that goal, the most important files to include are:
- data
- code
- one best checkpoint per released model
- matching scaler files
- evaluation outputs

### Full reproducibility
A reader should:
1. regenerate QSTS outputs,
2. regenerate ground-truth EV hosting capacity labels,
3. rebuild graph tensors and ML-ready tensors,
4. retrain the models, and
5. reproduce the final plots and metrics.

For that goal, include the full pipeline and all required intermediate data.

## Suggested execution order

### 1. Configure paths
Copy `configs/paths.example.yaml` to `configs/paths.yaml` and update paths only if you changed the default repository layout.

### 2. Generate QSTS outputs
Run:
```powershell
python .\scripts\01_generate_qsts.py
```

Expected outputs:
- `data/interim/qsts/base/qsts_base.npz`
- `data/interim/qsts/s1/qsts_s1.npz`
- `data/interim/qsts/s2/qsts_s2.npz`
- `data/interim/qsts/s3/qsts_s3.npz`
- `data/interim/qsts/s4/qsts_s4.npz`
- `data/interim/qsts/s5/qsts_s5.npz`
- one `metadata.json` file beside each scenario output

### 3. Generate graph features
Run:
```powershell
python .\scripts\03_build_graph_features.py
```

Expected outputs for each scenario graph folder:
- `edge_index.pt`
- `edge_attr.pt`
- `graph_meta.json`
- `edge_attr_stats.npz`

### 4. Generate ground-truth EV hosting capacity labels
Run:
```powershell
python .\scripts\02_generate_ground_truth_evhc.py
```

Expected outputs:
- one `ev_hc_bus<id>.npz` file per bus in each scenario folder under `data/processed/ground_truth/`

### 5. Build ML-ready tensors
Run:
```powershell
python .\scripts\04_build_ml_dataset.py
```

Expected outputs in each scenario folder under `data/processed/ml_ready/`:
- `features.pt`
- `targets.pt`
- `load_bus_mask.pt`
- `target_mask.pt`
- `bus_index.json`

### 6. Train models
Run:
```powershell
python .\scripts\05_train_3d_ecgat.py
python .\scripts\06_train_dnn_baseline.py
python .\scripts\07_train_lstm_baseline.py
```

Expected outputs:
- `models/3d_ecgat/scalers_node_edge_protocol_b.npz`
- `models/3d_ecgat/3d_ecgat_protocol_b_w24_best_state_dict.pt`
- `models/dnn_baseline/scaler_dnn_protocol_b.npz`
- `models/dnn_baseline/best_dnn_protocol_b.pt`
- `models/lstm_baseline/scaler_lstm_protocol_b.npz`
- `models/lstm_baseline/best_lstm_protocol_b.pt`

### 7. Evaluate the proposed model
Run:
```powershell
python .\scripts\08_evaluate_models.py
```

The  evaluation script writes to `results/` after evaluation.

Expected outputs:
- `metrics_per_scenario_*.csv`
- `node_MAE_map_*.pdf`
- `weekly_overlay_*.pdf`

## Environment notes

The public `requirements.txt` is intentionally lean and paper-specific.


## Windows and OpenDSS note

The preprocessing and simulation stages use OpenDSS and Windows-oriented dependencies. If a reader wants to rerun the full pipeline, they will need a compatible Windows/OpenDSS setup.

## Permanent archive

The archived reproducibility package associated with this repository is available at:
[<ZENODO_REPRO_DOI>](https://doi.org/https://doi.org/10.5281/zenodo.19491863)

If a separate GitHub release archive was created through Zenodo, it is available at:
[<ZENODO_SOFTWARE_DOI>](https://doi.org/10.5281/zenodo.19491494)

