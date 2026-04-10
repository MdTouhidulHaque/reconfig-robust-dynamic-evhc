"""Train the LSTM baseline on the cleaned repository layout."""

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

from common import DEFAULT_PROTOCOL, ensure_dir, project_root_from_script, resolve_scenario_dir, repo_paths

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

ML_READY = PATHS["ml_ready"]
OUTPUT_DIR = ensure_dir(PATHS["models"] / "lstm_baseline")

TRAIN_SCENS = DEFAULT_PROTOCOL["train_scenarios"]
VAL_SCENS = DEFAULT_PROTOCOL["val_scenarios"]
ALL_SCENS = TRAIN_SCENS + VAL_SCENS + DEFAULT_PROTOCOL["test_scenarios"]

WINDOW = DEFAULT_PROTOCOL["window"]
A_TRAIN_END = DEFAULT_PROTOCOL["train_end"]
B_VAL_END = DEFAULT_PROTOCOL["val_end"]
T_TOTAL = DEFAULT_PROTOCOL["total_steps"]

BATCH = 16
EPOCHS = 40
LR = 5e-4
WD = 1e-4
PATIENCE = 15
CLIP_NORM = 2.0
HUBER_BETA = 1.0
LSTM_HIDDEN = 8
LSTM_LAYERS = 1
DROPOUT = 0.05

USE_GLOBAL_CONTEXT = True
USE_BUS_EMB = True
BUS_EMB_DIM = 16
USE_TEMP_ATT = True
USE_SOFTPLUS_OUT = True
ADD_TIME_FEATS = True

SCALER_P = OUTPUT_DIR / "scaler_lstm_protocol_b.npz"
SAVE_BEST = OUTPUT_DIR / "best_lstm_protocol_b.pt"

SEED = 1862437
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Paths & Settings
# =======================

# =======================
# Helpers
# =======================
def find_node_dir(scen):
    d = resolve_scenario_dir(Path(ML_READY), scen)
    return d

def load_node_tensors(node_dir):
    X = torch.load(node_dir / "features.pt").float()
    Y = torch.load(node_dir / "targets.pt").float()
    M = torch.load(node_dir / "target_mask.pt")
    if Y.dim() == 3: Y = Y.squeeze(-1)
    if isinstance(M, np.ndarray): M = torch.from_numpy(M)
    
    # NEW: Add Cyclical Time Features if requested
    if ADD_TIME_FEATS:
        T_total = X.shape[0]
        # Create time indices 0..T-1
        t_idx = torch.arange(T_total, dtype=torch.float)
        
        # Hour of Day (0-23)
        day_h = t_idx % 24
        h_sin = torch.sin(2 * math.pi * day_h / 24.0).view(-1, 1)
        h_cos = torch.cos(2 * math.pi * day_h / 24.0).view(-1, 1)
        
        # Day of Year (approx 0-364)
        day_y = (t_idx // 24) % 365
        d_sin = torch.sin(2 * math.pi * day_y / 365.0).view(-1, 1)
        d_cos = torch.cos(2 * math.pi * day_y / 365.0).view(-1, 1)
        
        # Expand to (T, N, 4) matches X shape (T, N, F)
        N_nodes = X.shape[1]
        time_feats = torch.cat([h_sin, h_cos, d_sin, d_cos], dim=1).unsqueeze(1).expand(-1, N_nodes, -1)
        
        # Concatenate: X becomes (T, N, F+4)
        X = torch.cat([X, time_feats], dim=-1)
        
    return X, Y, M.float()

def fit_scaler(train_scens, train_hours, save_path):
    if Path(save_path).exists():
        z = np.load(str(save_path))
        return torch.from_numpy(z["mu"]), torch.from_numpy(z["sd"])

    sum_x, sum_x2, cnt = None, None, None
    th = torch.from_numpy(train_hours.astype(np.int64))

    for scen in train_scens:
        X, _, _ = load_node_tensors(find_node_dir(scen))
        Xt = X.index_select(0, th)
        valid = torch.isfinite(Xt)
        x0 = torch.where(valid, Xt, torch.zeros_like(Xt))

        s = x0.sum(dim=(0,1))
        s2 = (x0**2).sum(dim=(0,1))
        c = valid.sum(dim=(0,1)).float()

        if sum_x is None: sum_x, sum_x2, cnt = s, s2, c
        else: sum_x+=s; sum_x2+=s2; cnt+=c

    cnt = cnt.clamp_min(1.0)
    mu = sum_x / cnt
    var = (sum_x2 / cnt) - mu**2
    sd = torch.sqrt(var.clamp_min(1e-12)).clamp_min(1e-6)
    
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
# Dataset / Model
# =======================
class WindowDataset(Dataset):
    def __init__(self, X, Y, end_times, W):
        self.X, self.Y, self.end_times, self.W = X, Y, end_times, W
        self.end_times = self.end_times[self.end_times >= (W - 1)]
    def __len__(self): return len(self.end_times)
    def __getitem__(self, i):
        t = int(self.end_times[i])
        return self.X[t-self.W+1:t+1], self.Y[t], t

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x: [Batch, Seq, Hidden]
        # weights: [Batch, Seq, 1]
        scores = self.query(x)
        weights = F.softmax(scores, dim=1)
        # context: [Batch, Hidden]
        context = (x * weights).sum(dim=1)
        return context

class PerNodeContextLSTM(nn.Module):
    def __init__(self, fin, n_nodes):
        super().__init__()
        # Input Dimension calculation
        fin_aug = fin + (2*fin if USE_GLOBAL_CONTEXT else 0) + (BUS_EMB_DIM if USE_BUS_EMB else 0)
        
        if USE_BUS_EMB: self.bus_emb = nn.Embedding(n_nodes, BUS_EMB_DIM)
        
        self.in_ln = nn.LayerNorm(fin_aug)
        self.lstm = nn.LSTM(fin_aug, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True, dropout=(DROPOUT if LSTM_LAYERS>1 else 0))
        
        if USE_TEMP_ATT:
            self.att = TemporalAttention(LSTM_HIDDEN)
        
        self.out_ln = nn.LayerNorm(LSTM_HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN, 1), nn.Softplus() if USE_SOFTPLUS_OUT else nn.Identity()
        )

    def forward(self, xw):
        # xw: [Batch, Window, Nodes, Features]
        B, W, N, F = xw.shape
        parts = [xw]
        if USE_GLOBAL_CONTEXT:
            g_mu = xw.mean(dim=2, keepdim=True).expand(B, W, N, F)
            g_sd = xw.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6).expand(B, W, N, F)
            parts.extend([g_mu, g_sd])
        if USE_BUS_EMB:
            e = self.bus_emb(torch.arange(N, device=xw.device)).view(1, 1, N, -1).expand(B, W, N, -1)
            parts.append(e)
            
        x = torch.cat(parts, dim=-1)
        # Flatten Batch and Nodes -> [B*N, Window, F_aug]
        x = self.in_ln(x).permute(0, 2, 1, 3).reshape(B*N, W, -1)
        
        out, _ = self.lstm(x) # [B*N, W, Hidden]
        
        if USE_TEMP_ATT:
            ctx = self.att(out) # [B*N, Hidden]
        else:
            ctx = out[:, -1, :] # Last step
            
        return self.head(self.out_ln(ctx)).view(B, N)

# =======================
# Main
# =======================
def main():
    print(f"=== LSTM (Temporal Enhanced) | No Bus Emb, +TimeFeats, +Att ===")
    
    # Time splits
    train_end = np.arange(WINDOW-1, A_TRAIN_END+1)
    val_end   = np.arange(A_TRAIN_END+WINDOW, B_VAL_END+1)
    test_end  = np.arange(B_VAL_END+WINDOW, T_TOTAL)
    train_hrs = np.arange(0, A_TRAIN_END+1)

    # Scaler
    mu, sd = fit_scaler(TRAIN_SCENS, train_hrs, SCALER_P)
    mu, sd = mu.view(1, 1, -1), sd.view(1, 1, -1)

    # Loaders
    def get_loader(scen, ends, shuffle):
        X, Y, M = load_node_tensors(find_node_dir(scen))
        Xn = torch.nan_to_num((X - mu) / sd)
        ds = WindowDataset(Xn, Y, ends, WINDOW)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle), M, X.shape[1], X.shape[2]

    loaders, masks = {}, {}
    N_nodes, F_dim = None, None
    
    # Train Loaders
    for s in TRAIN_SCENS:
        ld, M, N, FeatDim = get_loader(s, train_end, True)
        loaders[s], masks[s] = ld, M
        N_nodes, F_dim = N, FeatDim
    
    # Val Loaders
    for s in VAL_SCENS:
        ld, M, _, _ = get_loader(s, val_end, False)
        loaders[s], masks[s] = ld, M

    # Train
    model = PerNodeContextLSTM(F_dim, N_nodes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    
    best_val = float('inf')
    
    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        nb_tr = 0
        for s in TRAIN_SCENS:
            for xw, y, t in loaders[s]:
                xw, y = xw.to(device), y.to(device)
                m = (masks[s][t] > 0.5).to(device) if masks[s].dim()==2 else (masks[s]>0.5).expand(y.shape).to(device)
                
                pred = model(xw)
                loss = F.smooth_l1_loss(pred[m], y[m], beta=HUBER_BETA)
                
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
                for xw, y, t in loaders[s]:
                    xw, y = xw.to(device), y.to(device)
                    m = (masks[s][t] > 0.5).to(device) if masks[s].dim()==2 else (masks[s]>0.5).expand(y.shape).to(device)
                    pred = model(xw)
                    va_loss += F.smooth_l1_loss(pred[m], y[m], beta=HUBER_BETA).item()
                    nb_va += 1
        
        avg_va = va_loss / max(nb_va, 1)
        sched.step(avg_va)
        print(f"Epoch {ep}: Train Loss {avg_tr:.4f} | Val Loss {avg_va:.4f}")
        
        if avg_va < best_val:
            best_val = avg_va
            torch.save(model.state_dict(), str(SAVE_BEST))
            
    # Final Test
    print("\n--- Final Test Results (All Scenarios on Test Block) ---")
    model.load_state_dict(torch.load(SAVE_BEST))
    model.eval()
    
    for s in ALL_SCENS:
        ld, _, _, _ = get_loader(s, test_end, False)
        # Re-fetch mask from dict just to be safe
        M = masks.get(s)
        if M is None: # Should be loaded by find_node_dir inside get_loader but let's be safe
             _, _, M = load_node_tensors(find_node_dir(s))

        trues, preds, mbools = [], [], []
        with torch.no_grad():
            for xw, y, t in ld:
                xw = xw.to(device)
                if M.dim() == 2:
                    m = (M[t] > 0.5).numpy().astype(bool)
                else:
                    m = (M > 0.5).expand(y.shape).numpy().astype(bool)
                
                p = model(xw).cpu().numpy()
                trues.append(y.numpy())
                preds.append(p)
                mbools.append(m)
        
        if len(trues) > 0:
            met = compute_metrics_pct(np.concatenate(trues), np.concatenate(preds), np.concatenate(mbools))
            print(f"{s}: nMAE={met['nMAE']:.2f}% nRMSE={met['nRMSE']:.2f}% R2={met['R2']:.4f} Over={met['Over']:.3f}")

if __name__ == "__main__":
    main()