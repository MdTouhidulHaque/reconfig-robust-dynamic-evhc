"""Evaluate the 3D-ECGAT model and generate summary metrics and figures."""

import os
import glob
import re
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import softmax as pyg_softmax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

from pathlib import Path

from common import DEFAULT_PROTOCOL, project_root_from_script, repo_paths, resolve_scenario_dir

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

RAW_NPZ_DIR = PATHS["ground_truth"] / "base"
CKPT_PATH = ""
SCALER_PATH = ""

ALL_SCENARIOS = ["base", "s1", "s2", "s3", "s4", "s5"]
WEEK_BUS_IDS = [1011, 2028, 3147]
WEEK_START_ENDTIME = 7444
WEEK_SCENARIOS = ["s4", "s5"]

X_TICK_STEP = 12
DAY_MARKER_ALPHA = 0.15

WINDOW = DEFAULT_PROTOCOL["window"]
T_TOTAL = DEFAULT_PROTOCOL["total_steps"]
B_VAL_END = DEFAULT_PROTOCOL["val_end"]
USE_TEST_BLOCK = True
OUTDIR_NAME = "results"
ZCLIP = 6.0
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Helper Functions
# ============================

def set_pub_style():
    """Sets styles matching Analyze_Ground_Truth_rev1.py"""
    plt.rcParams.update({
        "font.size": 18,
        "font.weight": "bold",
        "axes.labelsize": 24,
        "axes.labelweight": "bold",
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "legend.fontsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 3.0,
        "axes.linewidth": 2.0,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

def req(path: str) -> str:
    if not os.path.exists(path): raise FileNotFoundError(path)
    return path

def pick_latest_file(root: str, exts, prefer_substr: str = None):
    candidates, preferred = [], []
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(tuple(e.lower() for e in exts)):
                p = os.path.join(d, fn)
                candidates.append(p)
                if prefer_substr and (prefer_substr.lower() in fn.lower()):
                    preferred.append(p)
    pool = preferred if preferred else candidates
    if not pool: return None
    pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pool[0]

def build_bus_index_map(npz_dir, target_ids):
    """
    Scans raw NPZ folder to find indices for a LIST of Bus IDs.
    Returns a dict: {bus_id: index}
    """
    print(f"Scanning raw data for Buses {target_ids} in: {npz_dir}")
    if not os.path.exists(npz_dir):
        print(f"[Warning] Raw directory not found. Cannot map IDs.")
        return {}
        
    files = glob.glob(os.path.join(npz_dir, "EV_HC_Bus*.npz"))
    if not files:
        print("[Warning] No .npz files found.")
        return {}
        
    bus_ids = []
    for f in files:
        match = re.search(r"Bus(\d+)", os.path.basename(f), re.IGNORECASE)
        if match:
            bus_ids.append(int(match.group(1)))
            
    bus_ids.sort()
    
    mapping = {}
    for tid in target_ids:
        if tid in bus_ids:
            idx = bus_ids.index(tid)
            mapping[tid] = idx
            print(f"  -> Found Bus {tid} at index {idx}")
        else:
            print(f"  -> Bus {tid} NOT FOUND in raw files.")
            
    return mapping

# ---------------------------
# Data Loading
# ---------------------------
def load_graph(gdir: str):
    ei = torch.load(req(os.path.join(gdir, "edge_index.pt"))).long()
    cand = ["edge_attr_enriched.pt", "edge_attr.pt"]
    ea = None
    for c in cand:
        p = os.path.join(gdir, c)
        if os.path.exists(p):
            ea = torch.load(p).float()
            break
    if ea is None: raise FileNotFoundError(f"No edge attr found in {gdir}")
    return ei, ea

def load_protocolB_scalers(npz_path: str):
    z = np.load(req(npz_path))
    if "node_mu" not in z:
        raise KeyError(f"Expected 'node_mu' in scaler. Found: {list(z.keys())}")
    node_mu = torch.from_numpy(z["node_mu"]).float().view(1, 1, -1)
    node_sd = torch.from_numpy(z["node_sd"]).float().view(1, 1, -1).clamp_min(1e-6)
    edge_mu = torch.from_numpy(z["edge_mu"]).float().view(1, -1)
    edge_sd = torch.from_numpy(z["edge_sd"]).float().view(1, -1).clamp_min(1e-6)
    return node_mu, node_sd, edge_mu, edge_sd

class WindowDataset(Dataset):
    def __init__(self, X, Y, end_times: np.ndarray, W: int):
        self.X, self.Y, self.W = X, Y, W
        self.end_times = np.asarray(end_times, dtype=np.int64)
        self.end_times = self.end_times[self.end_times >= (W - 1)]
    def __len__(self): return len(self.end_times)
    def __getitem__(self, i):
        t_end = int(self.end_times[i])
        xw = self.X[t_end - self.W + 1 : t_end + 1]
        y  = self.Y[t_end]
        return xw, y, t_end

# ---------------------------
# 3D Graph Model
# ---------------------------
ST_HIDDEN, ST_HEADS, ST_LAYERS, RES_SCALE = 64, 4, 2, 0.3

def build_3d_adj(spatial_edge_index, edge_attr, n_nodes, window):
    dev = spatial_edge_index.device
    spatial_indices = []
    spatial_attrs = []
    for t in range(window):
        offset = t * n_nodes
        spatial_indices.append(spatial_edge_index + offset)
        spatial_attrs.append(edge_attr)
    all_spatial_idx = torch.cat(spatial_indices, dim=1)
    all_spatial_attr = torch.cat(spatial_attrs, dim=0)

    temp_indices = []
    for t in range(window - 1):
        src = torch.arange(t * n_nodes, (t + 1) * n_nodes, device=dev)
        dst = torch.arange((t + 1) * n_nodes, (t + 2) * n_nodes, device=dev)
        temp_indices.append(torch.stack([src, dst], dim=0))
    all_temp_idx = torch.cat(temp_indices, dim=1)
    temp_feat_val = edge_attr.mean(dim=0, keepdim=True).expand(all_temp_idx.size(1), -1)

    full_edge_index = torch.cat([all_spatial_idx, all_temp_idx], dim=1)
    full_edge_attr  = torch.cat([all_spatial_attr, temp_feat_val], dim=0)
    return full_edge_index, full_edge_attr

class FiLM3DECGATBlock(nn.Module):
    def __init__(self, d_node, d_edge, hidden, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dk = hidden // heads
        self.q = nn.Linear(d_node, hidden)
        self.k = nn.Linear(d_node, hidden)
        self.v = nn.Linear(d_node, hidden)
        self.film = nn.Linear(d_edge, 2 * hidden)
        self.out = nn.Linear(hidden, d_node)
        self.ln = nn.LayerNorm(d_node)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        B, N_total, _ = x.shape
        Q = self.q(x).view(B, N_total, self.heads, -1)
        K = self.k(x).view(B, N_total, self.heads, -1)
        V = self.v(x).view(B, N_total, self.heads, -1)
        src, dst = edge_index[0], edge_index[1]
        q_e, k_e, v_e = Q[:, dst], K[:, src], V[:, src]
        gamma, beta = self.film(edge_attr).chunk(2, dim=-1)
        gamma = gamma.view(1, -1, self.heads, self.dk)
        beta  = beta.view(1, -1, self.heads, self.dk)
        k_e = k_e * (1.0 + gamma) + beta
        logits = (q_e * k_e).sum(-1) / math.sqrt(self.dk)
        att = torch.stack([pyg_softmax(logits[b], dst, num_nodes=N_total) for b in range(B)], dim=0)
        msg = (att.unsqueeze(-1) * v_e)
        out = torch.zeros((B, N_total, self.heads, self.dk), device=x.device).index_add_(1, dst, msg)
        y = self.out(out.view(B, N_total, -1))
        return self.ln(x + self.drop(y))

class Model3DECGAT(nn.Module):
    def __init__(self, fin, e_dim):
        super().__init__()
        self.in_proj = nn.Linear(fin, ST_HIDDEN)
        self.blocks = nn.ModuleList([FiLM3DECGATBlock(ST_HIDDEN, e_dim, 128, ST_HEADS) for _ in range(ST_LAYERS)])
        self.head = nn.Sequential(nn.Linear(ST_HIDDEN, 1), nn.Softplus())
    def forward(self, xw, spatial_ei, spatial_ea):
        B, W, N, F = xw.shape
        ei_3d, ea_3d = build_3d_adj(spatial_ei, spatial_ea, N, W)
        x = xw.reshape(B, W * N, F)
        x = self.in_proj(x)
        for blk in self.blocks: x = x + RES_SCALE * (blk(x, ei_3d, ea_3d) - x)
        out = x.view(B, W, N, -1)[:, -1, :, :]
        return self.head(out).squeeze(-1)

# ---------------------------
# Metrics (Percentage)
# ---------------------------
def core_metrics(y_true, y_pred, ok):
    if ok.sum() == 0: 
        return {"MAE": np.nan, "RMSE": np.nan, "nMAE_pct": np.nan, "nRMSE_pct": np.nan, "R2": np.nan}
    
    yt, yp = y_true[ok].astype(float), y_pred[ok].astype(float)
    mae = np.mean(np.abs(yp - yt))
    rmse = np.sqrt(np.mean((yp - yt)**2))
    
    mean_val = np.mean(yt) + 1e-12
    mean_val_abs = np.mean(np.abs(yt)) + 1e-12
    
    nmae_pct = (mae / mean_val_abs) * 100.0
    nrmse_pct = (rmse / mean_val) * 100.0
    r2 = 1.0 - (np.sum((yt - yp)**2) / (np.sum((yt - np.mean(yt))**2) + 1e-12))
    
    return {"MAE": mae, "RMSE": rmse, "nMAE_pct": nmae_pct, "nRMSE_pct": nrmse_pct, "R2": r2}

def over_under_rates(y_true, y_pred, ok):
    if ok.sum() == 0: return np.nan, np.nan
    yt, yp = y_true[ok], y_pred[ok]
    return np.mean(yp > yt), np.mean(yp < yt)

def per_bus_mae(y_true, y_pred, ok2d):
    S, N = y_true.shape
    out = np.full((N,), np.nan)
    for b in range(N):
        m = ok2d[:, b]
        if m.sum() > 0:
            out[b] = np.mean(np.abs(y_pred[m, b] - y_true[m, b]))
    return out

# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def predict_scenario(base_dir, scen, model, node_mu, node_sd, edge_mu, edge_sd, end_times, device, batch_size, zclip):
    node_dir = resolve_scenario_dir(Path(base_dir) / "data" / "processed" / "ml_ready", scen)
    try:
        g_dir = resolve_scenario_dir(Path(base_dir) / "data" / "processed" / "graph", scen)
    except FileNotFoundError:
        g_dir = Path(base_dir) / "data" / "processed" / "graph" / "base"

    X = torch.load(req(os.path.join(node_dir, "features.pt"))).float()
    Y = torch.load(req(os.path.join(node_dir, "targets.pt"))).float()
    M = torch.load(req(os.path.join(node_dir, "target_mask.pt")))
    if Y.dim() == 3: Y = Y.squeeze(-1)
    if isinstance(M, np.ndarray): M = torch.from_numpy(M)
    M = M.float()

    Xn = (X - node_mu) / node_sd
    Xn = torch.nan_to_num(Xn, nan=0.0)
    if zclip > 0: Xn = torch.clamp(Xn, -zclip, zclip)

    ei, ea_raw = load_graph(g_dir)
    ea = (ea_raw - edge_mu) / edge_sd
    ea = torch.nan_to_num(ea, nan=0.0)
    if zclip > 0: ea = torch.clamp(ea, -zclip, zclip)
    
    ds = WindowDataset(Xn, Y, end_times, WINDOW)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ei, ea = ei.to(device), ea.to(device)
    model.eval()

    preds, trues, oks = [], [], []
    for xw, y, t_end in ld:
        xw = xw.to(device)
        p = model(xw, ei, ea).detach().cpu()
        if M.dim() == 1: mb = (M > 0.5).unsqueeze(0).expand(p.shape[0], -1)
        else: mb = (M[t_end] > 0.5)
        preds.append(p.numpy())
        trues.append(y.numpy())
        oks.append(mb.numpy() & np.isfinite(y.numpy()))

    return {"pred": np.concatenate(preds, 0), "true": np.concatenate(trues, 0), "ok": np.concatenate(oks, 0)}

# ---------------------------
# Plotting (Date at Bottom Right)
# ---------------------------
def plot_week_overlay(y_true, y_pred, ok, bus_idx, bus_id, scen, date_str, outpdf):
    yt = y_true[:, bus_idx]
    yp = y_pred[:, bus_idx]
    mk = ok[:, bus_idx]
    
    yt = np.where(mk, yt, np.nan)
    yp = np.where(mk, yp, np.nan)
    
    # 1. Figure Size
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    
    hours = np.arange(168)
    
    # 2. Line Styles
    ax.plot(hours, yt, linestyle="-", color="C0", label="Ground Truth", 
            marker="o", markevery=5, markersize=10, markeredgewidth=1, zorder=2)
    ax.plot(hours, yp, linestyle="-", color="C1", label="Predicted", 
            marker="*", markevery=5, markersize=12, markeredgewidth=1, zorder=2)
    
    # 3. Day Markers
    for x in range(0, 169, 24):
        ax.axvline(x, color="k", alpha=DAY_MARKER_ALPHA, linewidth=2, zorder=1)
        
    ax.set_xlim(0, 167)
    
    # 4. X-Ticks
    step = max(1, int(X_TICK_STEP))
    pos = np.arange(0, 168, step)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{p:d}" for p in pos], rotation=45, ha="right")
    
    # 5. Labels & Title
    ax.set_xlabel("Hour within week (0–167)")
    ax.set_ylabel("EV Hosting capacity (kW)")
    
    label_str = f"Bus ID {bus_id}" if bus_id else f"Bus Index {bus_idx}"
    
    # Title (Clean)
    title_str = f"Weekly overlay — {scen} — {label_str}"
    ax.set_title(title_str, fontweight="bold")
    
    ax.grid(True, linestyle="--", alpha=0.0)
    
    # Legend
    ax.legend(loc="upper right", frameon=True, ncol=2)
    
    # 6. Date Text Box (Bottom Right)
    # Using 'transform=ax.transAxes' puts it relative to the plotting box (0 to 1)
    # (0.98, 0.03) puts it in the bottom-right corner.
    ax.text(0.98, 0.03, f"Week Start: {date_str}", 
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="normal", 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))
    
    plt.tight_layout()
    plt.savefig(outpdf, bbox_inches="tight")
    plt.close()

def node_mae_map(per_node_mae, edge_index, outpdf, title):
    if not HAS_NX: return 
    G = nx.Graph()
    G.add_nodes_from(range(len(per_node_mae)))
    for u, v in edge_index.T: G.add_edge(int(u), int(v))
    plt.figure(figsize=(9.0, 8.0))
    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=per_node_mae, node_size=75, cmap="viridis", linewidths=0.0)
    nx.draw_networkx_edges(G, pos, alpha=0.55, width=1.1)
    cb = plt.colorbar(nodes, fraction=0.046, pad=0.04)
    cb.set_label("Per-node MAE")
    plt.title(title, fontweight="bold")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpdf, bbox_inches="tight")
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    set_pub_style()
    base_dir = str(PROJECT_ROOT)
    outdir = os.path.join(base_dir, OUTDIR_NAME)
    os.makedirs(outdir, exist_ok=True)
    
    outputs_root = str(Path(base_dir) / "models" / "3d_ecgat")
    ckpt = CKPT_PATH or pick_latest_file(outputs_root, (".pt",), "best")
    scaler = SCALER_PATH or pick_latest_file(outputs_root, (".npz",), "scaler")
    
    print(f"Ckpt: {ckpt}\nScaler: {scaler}\nOutdir: {outdir}")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 1. Resolve Bus Indices (Multi-bus support)
    bus_map = build_bus_index_map(RAW_NPZ_DIR, WEEK_BUS_IDS)
    if not bus_map:
        print("[Error] No valid buses found in map. Plots will be skipped for specific buses.")

    # Load Scalers & Model
    node_mu, node_sd, edge_mu, edge_sd = load_protocolB_scalers(scaler)
    
    node_base_dir = Path(base_dir) / "data" / "processed" / "ml_ready" / "base"
    X0 = torch.load(node_base_dir / "features.pt")
    Fdim = X0.shape[-1]; N = X0.shape[1]
    
    g_base = Path(base_dir) / "data" / "processed" / "graph" / "base"
    _, ea0 = load_graph(g_base)
    Edim = ea0.shape[1]

    model = Model3DECGAT(fin=Fdim, e_dim=Edim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    if USE_TEST_BLOCK:
        eval_end_times = np.arange(B_VAL_END + WINDOW, T_TOTAL)
        lbl = "TEST_BLOCK"
    else:
        eval_end_times = np.arange(WINDOW-1, T_TOTAL)
        lbl = "FULL_RANGE"

    # Metrics Loop
    rows = []
    print("\n--- Running Metrics (Percentage) ---")
    for scen in ALL_SCENARIOS:
        print(f"Evaluating {scen}...")
        pack = predict_scenario(base_dir, scen, model, node_mu, node_sd, edge_mu, edge_sd,
                                eval_end_times, device, BATCH_SIZE, ZCLIP)
        
        c = core_metrics(pack["true"], pack["pred"], pack["ok"])
        over, under = over_under_rates(pack["true"], pack["pred"], pack["ok"])
        
        print(f"  MAE={c['MAE']:.4f} nMAE={c['nMAE_pct']:.2f}% RMSE={c['RMSE']:.4f} nRMSE={c['nRMSE_pct']:.2f}% R2={c['R2']:.4f} Over={over:.3f}")
        
        rows.append({
            "Scenario": scen,
            "MAE": c['MAE'], "nMAE (%)": c['nMAE_pct'], 
            "RMSE": c['RMSE'], "nRMSE (%)": c['nRMSE_pct'],
            "R2": c['R2'], "OverPred": over
        })
        
        node_mae = per_bus_mae(pack["true"], pack["pred"], pack["ok"])
        ei_np, _ = load_graph(resolve_scenario_dir(Path(base_dir) / "data" / "processed" / "graph", scen))
        node_mae_map(node_mae, ei_np.numpy(), 
                     os.path.join(outdir, f"node_MAE_map_{scen}_{lbl}.pdf"),
                     f"{scen} Node MAE")

    with open(os.path.join(outdir, f"metrics_per_scenario_{lbl}.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Weekly Plots (Multi-bus)
    print("\n--- Generating Weekly Plots ---")
    
    # Use user-defined week start, or default to Test Block start
    week_start_idx = int(WEEK_START_ENDTIME) if int(WEEK_START_ENDTIME) >= 0 else (int(B_VAL_END) + WINDOW)
    week_end = np.arange(week_start_idx, week_start_idx + 168)
    
    # Calculate Date
    base_ts = pd.Timestamp("2017-01-01")
    start_ts = base_ts + pd.Timedelta(hours=week_start_idx)
    date_str = start_ts.strftime("%Y-%m-%d")
    print(f"Plotting Week Starting: {date_str} (Hour Index {week_start_idx})")
    
    for scen in WEEK_SCENARIOS:
        print(f"Generating overlays for scenario: {scen}")
        # Run inference once per scenario
        pack = predict_scenario(base_dir, scen, model, node_mu, node_sd, edge_mu, edge_sd,
                                week_end, device, BATCH_SIZE, ZCLIP)
        
        # Loop through requested buses
        for bus_id in WEEK_BUS_IDS:
            if bus_id not in bus_map:
                continue # Skip if map failed
            
            bus_idx = bus_map[bus_id]
            if bus_idx >= N:
                print(f"  [Skip] Bus ID {bus_id} maps to index {bus_idx} (>= {N}).")
                continue

            outpdf = os.path.join(outdir, f"weekly_overlay_{scen}_ID{bus_id}_{lbl}.pdf")
            plot_week_overlay(pack["true"], pack["pred"], pack["ok"], bus_idx, bus_id, 
                              scen, date_str, outpdf)
            print(f"  -> Saved: {outpdf}")

    print(f"\nDone. Results saved to: {outdir}")

if __name__ == "__main__":
    main()