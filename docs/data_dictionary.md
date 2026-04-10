# Data dictionary

This document describes the main files released with the repository.

## Scenario naming

The public release uses the following scenario names:

- `base`
- `s1`
- `s2`
- `s3`
- `s4`
- `s5`

These correspond to the base feeder configuration and five switching/reconfiguration scenarios described in the paper.

## Raw data

### `data/raw/load_data.h5`
Main raw load dataset used by the pipeline.

Expected role:
- hourly input load data used by QSTS and label generation

### `data/raw/dss_model/`
OpenDSS feeder model files.

Expected contents:
- `Master.dss`
- referenced line, load, transformer, and control definition files

Notes:
- this folder should be self-contained so `Master.dss` can compile without absolute paths

## Intermediate simulation outputs

### `data/interim/qsts/<scenario>/qsts_<scenario>.npz`
Quasi-static time-series outputs for all scenarios.

Examples:
- `data/interim/qsts/base/qsts_base.npz`
- `data/interim/qsts/s1/qsts_s1.npz`

Expected contents:
- hourly bus-level voltage and power-flow related arrays used to build ML-ready tensors

### `data/interim/qsts/<scenario>/metadata.json`
Metadata describing the corresponding QSTS run.

Recommended fields:
- scenario name
- number of hours
- source files used
- timestamp or software version
- relevant solver settings

## Processed graph files

### `data/processed/graph/<scenario>/edge_index.pt`
PyTorch tensor with graph connectivity in COO format.

Expected shape:
- `[2, E]`

### `data/processed/graph/<scenario>/edge_attr.pt`
PyTorch tensor of edge features.

Expected shape:
- `[E, D]`

For the attached graph manifest, each scenario has:
- `E = 880`
- `D = 13`

### `data/processed/graph/<scenario>/graph_meta.json`
Graph metadata file.

Recommended contents:
- edge count `E`
- edge feature dimension `D`
- feature column names
- breaker-state summary for the scenario

### `data/processed/graph/<scenario>/edge_attr_stats.npz`
Mean and standard deviation used for edge-feature inspection or normalization.

Expected contents:
- `mu`
- `sd`

## Ground-truth EV hosting capacity labels

### `data/processed/ground_truth/<scenario>/ev_hc_bus<id>.npz`
Ground-truth EV hosting capacity label for one bus.

Example:
- `data/processed/ground_truth/base/ev_hc_bus1003.npz`

Expected contents:
- time series of bus-level EV hosting capacity values
- array name used in the current cleaned script: `ev_hosting_capacity`

Notes:
- the cleaned dataset builder reads these files by bus ID from the filename
- uses lowercase file names consistently in the public release

## ML-ready tensors

Each scenario folder under `data/processed/ml_ready/` should contain the following files.

### `features.pt`
Node-feature tensor used for model input.

Expected shape:
- `[T, N, F]`

In the cleaned dataset script, the public feature stack is:
1. phase-A voltage magnitude
2. phase-B voltage magnitude
3. phase-C voltage magnitude
4. mean voltage magnitude
5. net active power injection or load-related active feature
6. net reactive power injection or load-related reactive feature
7. transformer active loading feature
8. transformer reactive loading feature
9. hour-of-day sine
10. hour-of-day cosine

So the default cleaned feature dimension is:
- `F = 10`

### `targets.pt`
Target tensor used for supervised learning.

Expected shape:
- `[T, N]`

Contents:
- bus-level EV hosting capacity labels aligned to the canonical bus order

### `load_bus_mask.pt`
Boolean mask identifying buses associated with load points.

Expected shape:
- `[N]`

### `target_mask.pt`
Boolean mask indicating which buses have valid target labels.

Expected shape in the cleaned dataset builder:
- `[N]`

Note:
- this mask is created as `(~np.isnan(y)).any(axis=0)`

### `bus_index.json`
Mapping between canonical bus identifiers and tensor indices.

Expected contents:
- `bus_list`
- `bus_index`

## Model artifacts

### `models/3d_ecgat/scalers_node_edge_protocol_b.npz`
Scaler file for the proposed model.

Expected contents:
- `node_mu`
- `node_sd`
- `edge_mu`
- `edge_sd`

### `models/3d_ecgat/3d_ecgat_protocol_b_w24_best_state_dict.pt`
Best-validation checkpoint for the proposed model.

### `models/dnn_baseline/scaler_dnn_protocol_b.npz`
Scaler for the DNN baseline.

Expected contents:
- `mu`
- `sd`

### `models/dnn_baseline/best_dnn_protocol_b.pt`
Best-validation checkpoint for the DNN baseline.

### `models/lstm_baseline/scaler_lstm_protocol_b.npz`
Scaler for the LSTM baseline.

Expected contents:
- `mu`
- `sd`

### `models/lstm_baseline/best_lstm_protocol_b.pt`
Best-validation checkpoint for the LSTM baseline.

## Evaluation outputs

The evaluation script writes outputs to  `results/`.

### `metrics_per_scenario_*.csv`
Scenario-level evaluation summary.

Expected columns:
- scenario
- nMAE
- nRMSE
- R2
- Over
- Under

### `node_MAE_map_*.pdf`
Per-node mean absolute error map for a scenario.

### `weekly_overlay_*.pdf`
Representative weekly ground-truth versus predicted overlays for selected buses.


