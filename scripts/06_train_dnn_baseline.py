"""Train the topology-aware DNN baseline on the cleaned repository layout."""

import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common import DEFAULT_PROTOCOL, ensure_dir, project_root_from_script, resolve_scenario_dir, repo_paths

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

ML_READY = PATHS["ml_ready"]
GRAPH = PATHS["graph"]
BASE_G = PATHS["graph"] / "base"
OUTPUT_DIR = ensure_dir(PATHS["models"] / "dnn_baseline")

TRAIN_SCENS = DEFAULT_PROTOCOL["train_scenarios"]
VAL_SCENS = DEFAULT_PROTOCOL["val_scenarios"]
ALL_SCENS = TRAIN_SCENS + VAL_SCENS + DEFAULT_PROTOCOL["test_scenarios"]

WINDOW = DEFAULT_PROTOCOL["window"]
A_TRAIN_END = DEFAULT_PROTOCOL["train_end"]
B_VAL_END = DEFAULT_PROTOCOL["val_end"]
T_TOTAL = DEFAULT_PROTOCOL["total_steps"]

BATCH = 32
EPOCHS = 100
LR = 1e-3
HIDDEN = 1024
USE_TOPO = True
HUBER_BETA = 1.0
DROPOUT = 0.3
WD = 1e-3

SAVE_BEST = OUTPUT_DIR / "best_dnn_protocol_b.pt"
SCALER_P = OUTPUT_DIR / "scaler_dnn_protocol_b.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1862437
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =======================
# Helpers
# =======================
def load_data(scen):
    d = resolve_scenario_dir(Path(ML_READY), scen)
    X = torch.load(d / "features.pt").float()
    Y = torch.load(d / "targets.pt").float()
    M = torch.load(d / "target_mask.pt")
    if Y.ndim == 3: Y = Y.squeeze(-1)
    if isinstance(M, np.ndarray): M = torch.from_numpy(M)
    return X, Y, M.float()

def load_graph(scen):
    d = resolve_scenario_dir(Path(GRAPH), scen) if (Path(GRAPH) / scen).is_dir() or (Path(GRAPH) / scen.replace("s","scenario-",1)).is_dir() else Path(BASE_G)
    ei = torch.load(d / "edge_index.pt").long().to(device)
    return ei

def neighbor_mean(x, edge_index):
    src, dst = edge_index
    out = torch.zeros_like(x)
    deg = torch.zeros(x.size(0), x.size(1), 1, device=device)
    B, N, FeatDim = x.shape
    ones = torch.ones(edge_index.size(1), 1, device=device)
    for b in range(B):
        out[b].index_add_(0, dst, x[b, src])
        deg[b].index_add_(0, dst, ones)
    return out / deg.clamp_min(1.0)

def fit_scaler(train_scens, train_hours, save_path):
    if Path(save_path).exists():
        z = np.load(str(save_path))
        return torch.from_numpy(z["mu"]), torch.from_numpy(z["sd"])
    xs = []
    th = train_hours.astype(np.int64)
    for s in train_scens:
        X, _, _ = load_data(s)
        xs.append(X[th]) 
    X_all = torch.cat(xs, dim=0)
    mu = X_all.mean(dim=(0,1))
    sd = X_all.std(dim=(0,1)).clamp_min(1e-6)
    np.savez(str(save_path), mu=mu.numpy(), sd=sd.numpy())
    return mu, sd

def compute_metrics_pct(y_true, y_pred, m_bool):
    ok = m_bool & np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0: return {"nMAE": np.nan, "nRMSE": np.nan, "R2": np.nan, "Over": np.nan}
    yt, yp = y_true[ok].astype(np.float64), y_pred[ok].astype(np.float64)
    mae = np.mean(np.abs(yp - yt))
    rmse = np.sqrt(np.mean((yp - yt)**2))
    mean_val = np.mean(yt) + 1e-12
    mean_abs = np.mean(np.abs(yt)) + 1e-12
    nmae_pct = (mae / mean_abs) * 100.0
    nrmse_pct = (rmse / mean_val) * 100.0
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - np.mean(yt))**2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    over = np.mean(yp > yt)
    return {"nMAE": nmae_pct, "nRMSE": nrmse_pct, "R2": r2, "Over": over}

# =======================
# Model
# =======================
class TopoDNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), 
            nn.LayerNorm(HIDDEN), 
            nn.GELU(),
            nn.Dropout(DROPOUT),  # <--- Added Dropout
            
            nn.Linear(HIDDEN, HIDDEN), 
            nn.LayerNorm(HIDDEN), 
            nn.GELU(),
            nn.Dropout(DROPOUT),  # <--- Added Dropout
            
            nn.Linear(HIDDEN, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# =======================
# Dataset
# =======================
class WinDS(Dataset):
    def __init__(self, X, Y, M, idxs):
        self.X, self.Y, self.M, self.idxs = X, Y, M, idxs[idxs >= WINDOW-1]
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        t = int(self.idxs[i])
        return self.X[t-WINDOW+1:t+1], self.Y[t], self.M[t] if self.M.ndim==2 else self.M, t

# =======================
# Main
# =======================
def main():
    print("=== DNN Protocol B (Regularized) | Drop=0.5, WD=1e-3 ===")
    
    train_hrs = np.arange(0, A_TRAIN_END+1)
    mu, sd = fit_scaler(TRAIN_SCENS, train_hrs, SCALER_P)
    mu, sd = mu.view(1, 1, -1), sd.view(1, 1, -1)
    graphs = {s: load_graph(s) for s in ALL_SCENS}
    
    def get_loader(scen, ends, shuffle):
        X, Y, M = load_data(scen)
        Xn = torch.nan_to_num((X - mu) / sd)
        ds = WinDS(Xn, Y, M, ends)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle), X.shape[2]

    train_end = np.arange(WINDOW-1, A_TRAIN_END+1)
    val_end   = np.arange(A_TRAIN_END+WINDOW, B_VAL_END+1)
    test_end  = np.arange(B_VAL_END+WINDOW, T_TOTAL)

    loaders = {}
    F_dim = 0
    for s in TRAIN_SCENS: loaders[s], F_dim = get_loader(s, train_end, True)
    for s in VAL_SCENS:   loaders[s], _ = get_loader(s, val_end, False)

    in_dim = WINDOW * F_dim + (2 * F_dim if USE_TOPO else 0)
    model = TopoDNN(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD) # AdamW for Weight Decay
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    
    best_val = float('inf')

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        nb_tr = 0
        for s in TRAIN_SCENS:
            ei = graphs[s]
            for xw, y, m, t in loaders[s]:
                xw, y, m = xw.to(device), y.to(device), m.to(device)
                m_bool = (m > 0.5)
                
                B, W, N, FeatDim = xw.shape
                x_flat = xw.permute(0, 2, 1, 3).reshape(B, N, W*FeatDim)
                
                feat = [x_flat]
                if USE_TOPO:
                    x_last = xw[:, -1] 
                    nbr = neighbor_mean(x_last, ei)
                    feat.extend([x_last, nbr])
                
                xin = torch.cat(feat, dim=-1)
                pred = model(xin)
                loss = F.smooth_l1_loss(pred[m_bool], y[m_bool], beta=HUBER_BETA)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                tr_loss += loss.item()
                nb_tr += 1
        
        avg_tr = tr_loss / max(nb_tr, 1)

        model.eval()
        va_loss = 0
        nb_va = 0
        with torch.no_grad():
            for s in VAL_SCENS:
                ei = graphs[s]
                for xw, y, m, t in loaders[s]:
                    xw, y, m = xw.to(device), y.to(device), m.to(device)
                    m_bool = (m > 0.5)
                    
                    B, W, N, FeatDim = xw.shape
                    x_flat = xw.permute(0, 2, 1, 3).reshape(B, N, W*FeatDim)
                    feat = [x_flat]
                    if USE_TOPO:
                        x_last = xw[:, -1]
                        nbr = neighbor_mean(x_last, ei)
                        feat.extend([x_last, nbr])
                        
                    xin = torch.cat(feat, dim=-1)
                    pred = model(xin)
                    va_loss += F.smooth_l1_loss(pred[m_bool], y[m_bool], beta=HUBER_BETA).item()
                    nb_va += 1
                    
        avg_va = va_loss / max(nb_va, 1)
        sched.step(avg_va)
        print(f"Epoch {ep}: Train Loss {avg_tr:.4f} | Val Loss {avg_va:.4f}")
        
        if avg_va < best_val:
            best_val = avg_va
            torch.save(model.state_dict(), str(SAVE_BEST))

    print("\n--- Final Test Results ---")
    model.load_state_dict(torch.load(SAVE_BEST))
    model.eval()
    
    for s in ALL_SCENS:
        ld, _ = get_loader(s, test_end, False)
        ei = graphs[s]
        trues, preds, mbools = [], [], []
        with torch.no_grad():
            for xw, y, m, t in ld:
                xw = xw.to(device)
                m_bool = (m > 0.5).numpy()
                B, W, N, FeatDim = xw.shape
                x_flat = xw.permute(0, 2, 1, 3).reshape(B, N, W*FeatDim)
                feat = [x_flat]
                if USE_TOPO:
                    x_last = xw[:, -1]
                    nbr = neighbor_mean(x_last, ei)
                    feat.extend([x_last, nbr])
                xin = torch.cat(feat, dim=-1)
                p = model(xin).cpu().numpy()
                trues.append(y.numpy())
                preds.append(p)
                mbools.append(m_bool)
        
        if len(trues) > 0:
            met = compute_metrics_pct(np.concatenate(trues), np.concatenate(preds), np.concatenate(mbools))
            print(f"{s}: nMAE={met['nMAE']:.2f}% nRMSE={met['nRMSE']:.2f}% R2={met['R2']:.4f} Over={met['Over']:.3f}")

if __name__ == "__main__":
    main()