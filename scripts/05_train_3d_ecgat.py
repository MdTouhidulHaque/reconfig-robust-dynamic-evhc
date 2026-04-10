"""Train the 3D-ECGAT model using the cleaned repository layout."""

import math
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import softmax as pyg_softmax

from common import DEFAULT_PROTOCOL, ensure_dir, project_root_from_script, resolve_scenario_dir, repo_paths

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

ML_READY = PATHS["ml_ready"]
GRAPH = PATHS["graph"]
OUTPUT_DIR = ensure_dir(PATHS["models"] / "3d_ecgat")

SEED = 1862437
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SCENS_2019 = DEFAULT_PROTOCOL["train_scenarios"]
VAL_SCENS_2019 = DEFAULT_PROTOCOL["val_scenarios"]
TEST_SCENS_2019 = DEFAULT_PROTOCOL["test_scenarios"]

A_TRAIN_END = DEFAULT_PROTOCOL["train_end"]
B_VAL_END = DEFAULT_PROTOCOL["val_end"]
WINDOW = DEFAULT_PROTOCOL["window"]

train_end_times = np.arange(WINDOW - 1, A_TRAIN_END + 1)
val_end_times = np.arange(A_TRAIN_END + WINDOW, B_VAL_END + 1)
test_end_times = np.arange(B_VAL_END + WINDOW, DEFAULT_PROTOCOL["total_steps"])
TRAIN_TIME_IDX_FOR_SCALER = slice(0, A_TRAIN_END + 1)

SCALERS_P = OUTPUT_DIR / "scalers_node_edge_protocol_b.npz"
SAVE_BEST = OUTPUT_DIR / f"3d_ecgat_protocol_b_w{WINDOW}_best_state_dict.pt"

BATCH = 4
EPOCHS = 120
LR = 2e-4
WD = 1e-4
PATIENCE = 12
USE_HUBER = True
HUBER_BETA = 1.0
ST_HIDDEN = 64
ST_HEADS = 4
ST_LAYERS = 2
RES_SCALE = 0.3
EPS_STD = 1e-6
ZCLIP = 6.0
NUM_WORKERS = 0
PIN_MEMORY = True


# =======================
# Utilities
# =======================
def _nan_safe_sum_sumsq_count(x: torch.Tensor):
    """
    Returns (sum, sumsq, count) per feature dimension over the first two dims.
    x: [T, N, F] or [E, F]
    """
    x = x.detach()
    finite = torch.isfinite(x)
    x0 = torch.where(finite, x, torch.zeros_like(x))
    # sum / sumsq along sample dims
    if x.dim() == 3:
        s    = x0.sum(dim=(0, 1)).double()
        ssq  = (x0 * x0).sum(dim=(0, 1)).double()
        cnt  = finite.sum(dim=(0, 1)).double().clamp_min(1.0)
    elif x.dim() == 2:
        s    = x0.sum(dim=0).double()
        ssq  = (x0 * x0).sum(dim=0).double()
        cnt  = finite.sum(dim=0).double().clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported tensor rank for stats: {x.shape}")
    return s, ssq, cnt

def fit_node_scaler(train_scens, year=None):
    """
    Fit node scaler on TRAIN scenarios × TRAIN time block only.
    """
    sum_f  = None
    ssq_f  = None
    cnt_f  = None

    for scen in train_scens:
        node_dir = resolve_scenario_dir(Path(ML_READY), scen)
        X = torch.load(Path(node_dir) / "features.pt").float()  # [T, N, F]
        X = X[TRAIN_TIME_IDX_FOR_SCALER]  # restrict to train time block
        s, ssq, cnt = _nan_safe_sum_sumsq_count(X)

        if sum_f is None:
            sum_f, ssq_f, cnt_f = s, ssq, cnt
        else:
            sum_f += s
            ssq_f += ssq
            cnt_f += cnt

        del X

    mu = (sum_f / cnt_f).float()
    var = (ssq_f / cnt_f).float() - mu * mu
    var = torch.clamp(var, min=EPS_STD**2)
    sd = torch.sqrt(var).clamp_min(EPS_STD)
    return mu, sd

def fit_edge_scaler(train_scens, year=None):
    """
    Fit edge scaler on TRAIN scenario graphs only (global).
    """
    sum_f  = None
    ssq_f  = None
    cnt_f  = None

    for scen in train_scens:
        try:
            g_dir = resolve_scenario_dir(Path(GRAPH), scen)
        except FileNotFoundError:
            g_dir = Path(GRAPH) / "base"

        ei = torch.load(Path(g_dir) / "edge_index.pt").long()
        p  = os.path.join(g_dir, "edge_attr_enriched.pt")
        if not os.path.exists(p):
            p = os.path.join(g_dir, "edge_attr.pt")
        ea = torch.load(p).float()  # [E, De]

        s, ssq, cnt = _nan_safe_sum_sumsq_count(ea)
        if sum_f is None:
            sum_f, ssq_f, cnt_f = s, ssq, cnt
        else:
            sum_f += s
            ssq_f += ssq
            cnt_f += cnt

        del ei, ea

    mu = (sum_f / cnt_f).float()
    var = (ssq_f / cnt_f).float() - mu * mu
    var = torch.clamp(var, min=EPS_STD**2)
    sd = torch.sqrt(var).clamp_min(EPS_STD)
    return mu, sd

def norm_and_clip_node(X, mu, sd):
    Xn = (X - mu.view(1, 1, -1)) / sd.view(1, 1, -1)
    Xn = torch.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    if ZCLIP is not None and ZCLIP > 0:
        Xn = torch.clamp(Xn, -ZCLIP, ZCLIP)
    return Xn

def norm_and_clip_edge(ea, mu, sd):
    ean = (ea - mu.view(1, -1)) / sd.view(1, -1)
    ean = torch.nan_to_num(ean, nan=0.0, posinf=0.0, neginf=0.0)
    if ZCLIP is not None and ZCLIP > 0:
        ean = torch.clamp(ean, -ZCLIP, ZCLIP)
    return ean

# =======================
# Data I/O
# =======================
def load_node_tensors(node_dir):
    X = torch.load(Path(node_dir) / "features.pt").float()
    Y = torch.load(Path(node_dir) / "targets.pt").float()
    M = torch.load(Path(node_dir) / "target_mask.pt")
    if Y.dim() == 3:
        Y = Y.squeeze(-1)
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M)
    return X, Y, M.float()

def load_graph(graph_dir):
    ei = torch.load(os.path.join(graph_dir, "edge_index.pt")).long()
    p = os.path.join(graph_dir, "edge_attr_enriched.pt")
    if not os.path.exists(p):
        p = os.path.join(graph_dir, "edge_attr.pt")
    ea = torch.load(p).float()
    return ei, ea

class WindowDataset(Dataset):
    def __init__(self, X, Y, end_times, W):
        self.X, self.Y, self.W = X, Y, W
        self.end_times = end_times[end_times >= (W - 1)]
    def __len__(self):
        return len(self.end_times)
    def __getitem__(self, i):
        t_end = int(self.end_times[i])
        xw = self.X[t_end - self.W + 1: t_end + 1]  # [W, N, F]
        y  = self.Y[t_end]                           # [N]
        return xw, y, t_end

# =======================
# 3D Graph Infrastructure
# =======================
def build_3d_adj(spatial_edge_index, edge_attr, n_nodes, window):
    """
    Replicate spatial edges across time; add temporal edges (bus,t)->(bus,t+1).
    Returns edge_index [2, E3D], edge_attr [E3D, De].
    """
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

# =======================
# Model
# =======================
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
        self.blocks = nn.ModuleList([
            FiLM3DECGATBlock(ST_HIDDEN, e_dim, 128, ST_HEADS) for _ in range(ST_LAYERS)
        ])
        self.head = nn.Sequential(nn.Linear(ST_HIDDEN, 1), nn.Softplus())

    def forward(self, xw, spatial_ei, spatial_ea):
        B, W, N, F = xw.shape
        ei_3d, ea_3d = build_3d_adj(spatial_ei, spatial_ea, N, W)

        x = xw.reshape(B, W * N, F)
        x = self.in_proj(x)

        for blk in self.blocks:
            x = x + RES_SCALE * (blk(x, ei_3d, ea_3d) - x)

        out = x.view(B, W, N, -1)[:, -1, :, :]
        return self.head(out).squeeze(-1)

# =======================
# Caches (avoid re-loading every epoch)
# =======================
class ScenarioCache:
    def __init__(self, node_mu, node_sd, edge_mu, edge_sd, year=None):
        self.node_mu = node_mu
        self.node_sd = node_sd
        self.edge_mu = edge_mu
        self.edge_sd = edge_sd
        self.year = year
        self.nodes = {}  # scen -> (Xn, Y, M)
        self.graphs = {} # scen -> (ei_gpu, ea_gpu)

    def get_nodes(self, scen):
        if scen in self.nodes:
            return self.nodes[scen]
        node_dir = resolve_scenario_dir(Path(ML_READY), scen)
        X, Y, M = load_node_tensors(node_dir)
        Xn = norm_and_clip_node(X, self.node_mu, self.node_sd)
        self.nodes[scen] = (Xn, Y, M)
        return self.nodes[scen]

    def get_graph(self, scen):
        if scen in self.graphs:
            return self.graphs[scen]
        try:
            g_dir = resolve_scenario_dir(Path(GRAPH), scen)
        except FileNotFoundError:
            g_dir = Path(GRAPH) / "base"
        ei, ea = load_graph(g_dir)
        ea = norm_and_clip_edge(ea, self.edge_mu, self.edge_sd)
        self.graphs[scen] = (ei.to(device), ea.to(device))
        return self.graphs[scen]

# =======================
# Metrics helpers
# =======================
def batch_mask(M, t_end_batch, batch_size):
    """
    Supports:
      - M: [N] constant mask
      - M: [T, N] time-varying mask
    Returns boolean mask [B, N].
    """
    if M.dim() == 1:
        return (M[None, :] > 0.5).expand(batch_size, -1)
    if M.dim() == 2:
        # t_end_batch can be int or tensor [B]
        if isinstance(t_end_batch, int):
            return (M[t_end_batch] > 0.5).unsqueeze(0).expand(batch_size, -1)
        t = torch.as_tensor(t_end_batch, dtype=torch.long)
        return (M[t] > 0.5)
    raise ValueError(f"Unsupported mask shape: {tuple(M.shape)}")

def masked_mae_sum_count(pred, y, mb):
    # pred,y: [B,N], mb: [B,N] bool
    diff = torch.abs(pred - y)
    s = diff[mb].sum()
    c = mb.sum()
    return s, c

def run_eval(model, cache, scen_list, end_times, split_name="VAL"):
    model.eval()
    mae_sum = 0.0
    mae_cnt = 0.0

    with torch.no_grad():
        for scen in scen_list:
            Xn, Y, M = cache.get_nodes(scen)
            ei, ea = cache.get_graph(scen)

            ds = WindowDataset(Xn, Y, end_times, WINDOW)
            ld = DataLoader(ds, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

            for xw, y, t_end in ld:
                xw = xw.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)
                pred = model(xw, ei, ea)

                mb = batch_mask(M.to(device), t_end, xw.size(0))
                s, c = masked_mae_sum_count(pred, y, mb)
                mae_sum += float(s.item())
                mae_cnt += float(c.item())

    mae = mae_sum / max(mae_cnt, 1.0)
    print(f"[{split_name}] MAE = {mae:.5f}  (count={int(mae_cnt)})")
    return mae

# =======================
# Main
# =======================
def main():
    print("=== 3D-ECGAT | Protocol B (unseen topology + unseen time) | NO time features ===")
    print(f"Device: {device}")
    print(f"Scenarios: train={TRAIN_SCENS_2019} | val={VAL_SCENS_2019} | test={TEST_SCENS_2019}")
    print(f"Time blocks (end_times): train[{train_end_times[0]}..{train_end_times[-1]}] "
          f"val[{val_end_times[0]}..{val_end_times[-1]}] test[{test_end_times[0]}..{test_end_times[-1]}]")

    # Sanity checks: disjoint scenario sets
    assert len(set(TRAIN_SCENS_2019) & set(VAL_SCENS_2019)) == 0, "Val scenarios must not be in train"
    assert len(set(TRAIN_SCENS_2019) & set(TEST_SCENS_2019)) == 0, "Test scenarios must not be in train"
    assert len(set(VAL_SCENS_2019) & set(TEST_SCENS_2019)) == 0, "Test scenarios must not be in val"

    # -----------------------
    # Fit / Load scalers
    # -----------------------
    if Path(SCALERS_P).exists():
        z = np.load(str(SCALERS_P))
        node_mu = torch.from_numpy(z["node_mu"]).float()
        node_sd = torch.from_numpy(z["node_sd"]).float()
        edge_mu = torch.from_numpy(z["edge_mu"]).float()
        edge_sd = torch.from_numpy(z["edge_sd"]).float()
        print(f"[scaler] Loaded: {SCALERS_P}")
    else:
        print("[scaler] Fitting node scaler (TRAIN scenarios × TRAIN time block only)...")
        node_mu, node_sd = fit_node_scaler(TRAIN_SCENS_2019)
        print("[scaler] Fitting edge scaler (TRAIN scenario graphs only)...")
        edge_mu, edge_sd = fit_edge_scaler(TRAIN_SCENS_2019)
        np.savez(
            SCALERS_P,
            node_mu=node_mu.numpy(), node_sd=node_sd.numpy(),
            edge_mu=edge_mu.numpy(), edge_sd=edge_sd.numpy()
        )
        print(f"[scaler] Saved: {SCALERS_P}")

    cache = ScenarioCache(node_mu, node_sd, edge_mu, edge_sd)

    # -----------------------
    # Infer dims from one scenario
    # -----------------------
    Xn0, Y0, M0 = cache.get_nodes(TRAIN_SCENS_2019[0])
    _, _, Fdim = Xn0.shape
    ei0, ea0 = cache.get_graph(TRAIN_SCENS_2019[0])
    e_dim = ea0.shape[1]

    model = Model3DECGAT(fin=Fdim, e_dim=e_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_val = float("inf")
    bad = 0

    # -----------------------
    # Training loop
    # -----------------------
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_loss_cnt = 0.0

        # Train: (TRAIN scenarios) × (TRAIN time block)
        for scen in TRAIN_SCENS_2019:
            Xn, Y, M = cache.get_nodes(scen)
            ei, ea = cache.get_graph(scen)

            ds = WindowDataset(Xn, Y, train_end_times, WINDOW)
            ld = DataLoader(ds, batch_size=BATCH, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

            for xw, y, t_end in ld:
                xw = xw.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)
                pred = model(xw, ei, ea)

                mb = batch_mask(M.to(device), t_end, xw.size(0))
                if USE_HUBER:
                    loss = F.smooth_l1_loss(pred[mb], y[mb], beta=HUBER_BETA)
                else:
                    loss = F.l1_loss(pred[mb], y[mb])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tr_loss_sum += float(loss.item()) * xw.size(0)
                tr_loss_cnt += float(xw.size(0))

        tr_loss = tr_loss_sum / max(tr_loss_cnt, 1.0)

        # Val: (VAL scenario) × (VAL time block)
        val_mae = run_eval(model, cache, VAL_SCENS_2019, val_end_times, split_name="VAL")

        print(f"Epoch {ep:02d} | TrainLoss = {tr_loss:.5f} | ValMAE = {val_mae:.5f}")

        sched.step(val_mae)

        if val_mae < best_val:
            best_val = val_mae
            bad = 0
            torch.save(model.state_dict(), str(SAVE_BEST))
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[early stop] patience={PATIENCE} reached.")
                break

    # -----------------------
    # Final test: (TEST scenarios) × (TEST time block)
    # -----------------------
    print("\n--- Final Test Evaluation (Protocol B) ---")
    model.load_state_dict(torch.load(SAVE_BEST, map_location=device))
    model.eval()
    for scen in TEST_SCENS_2019:
        test_mae = run_eval(model, cache, [scen], test_end_times, split_name=f"TEST {scen}")
        print(f"Scenario {scen}: MAE = {test_mae:.5f}")

    print("Done.")

if __name__ == "__main__":
    main()
