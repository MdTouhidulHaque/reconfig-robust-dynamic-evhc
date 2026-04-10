"""Compute per-bus EV hosting capacity and save one NPZ file per bus."""

def guess_label_from_name(path: str):
    stem = Path(path).stem.lower()
    if "scenario-1" in stem or "scenario_1" in stem:
        return "s1"
    if "scenario-2" in stem or "scenario_2" in stem:
        return "s2"
    if "scenario-3" in stem or "scenario_3" in stem:
        return "s3"
    if "scenario-4" in stem or "scenario_4" in stem:
        return "s4"
    if "scenario-5" in stem or "scenario_5" in stem:
        return "s5"
    return "base"

import os, re, json
import numpy as np
import h5py
import win32com.client
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pathlib import Path

from common import ensure_dir, project_root_from_script, repo_paths

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

mat_dir = str(PATHS["raw"])
h5_dir = str(PATHS["raw"])
dss_file = str(PATHS["master_dss"])
results_base = str(ensure_dir(PATHS["ground_truth"]))

# HC step and limits
ev_step_kw   = 3.3
voltage_min  = 0.95
voltage_max  = 1.05

# Search tuning
TOL_KW    = 0.1       # bisection tolerance
MAX_EV_KW = 1000.0    # hard cap for exponential search

# ===========================
# Feeder bus lists (unchanged)
# ===========================
FeederA_buses = list(range(1003, 1018))
FeederB_buses = (
    [2002,2003,2005] + list(range(2008,2012)) + list(range(2014,2019)) + [2020] +
    list(range(2022,2026)) + list(range(2028,2033)) + [2034,2035,2037] +
    list(range(2040,2044)) + list(range(2045,2057)) + list(range(2058,2061))
)
FeederC_buses = (
    [3002,3004] + list(range(3006,3008)) + list(range(3009,3015)) + list(range(3016,3022)) +
    list(range(3023,3030)) + list(range(3031,3040)) + list(range(3041,3046)) +
    list(range(3047,3053)) + [3054] + list(range(3056,3068)) + list(range(3070,3075)) +
    [3077,3078,3081] + list(range(3083,3092)) + list(range(3093,3100)) +
    list(range(3101,3107)) + list(range(3108,3113)) + list(range(3114,3118)) +
    list(range(3120,3133)) + list(range(3134,3139)) + list(range(3141,3156)) +
    list(range(3157,3163))
)
all_load_buses = FeederA_buses + FeederB_buses + FeederC_buses

# offsets
offsetA, offsetB, offsetC = 1001, 2001, 3001

# required dataset names in each H5 group
REQUIRED_KEYS = [
    "FeederA_P", "FeederA_Q",
    "FeederB_P", "FeederB_Q",
    "FeederC_P", "FeederC_Q",
]

# ===========================
# Utilities: MAT (v7.3) & H5
# ===========================
def load_mat_v7p3(filename, var):
    """Load MATLAB v7.3 dataset by name -> np.array, transposed to match existing usage."""
    with h5py.File(filename, "r") as f:
        return np.array(f[var]).T

def _pick_group(hf: h5py.File, preferred: str) -> str:
    """Choose the group to use: preferred if present, else the single group if only one exists."""
    if preferred in hf and isinstance(hf[preferred], h5py.Group):
        return preferred
    groups = [k for k in hf.keys() if isinstance(hf[k], h5py.Group)]
    if len(groups) == 1:
        return groups[0]
    raise ValueError(
        f"No group '{preferred}' and multiple groups present: {groups}. "
        f"Please ensure a top-level group named '{preferred}'."
    )

def load_year_h5(h5_path: str, preferred_group: str):
    """Return feeder P/Q arrays from an H5 file group."""
    with h5py.File(h5_path, "r") as hf:
        gname = _pick_group(hf, preferred_group)
        g = hf[gname]
        for k in REQUIRED_KEYS:
            if k not in g:
                raise KeyError(f"Missing dataset '{k}' in {h5_path} group '{gname}'")
        A_P = np.asarray(g["FeederA_P"]); A_Q = np.asarray(g["FeederA_Q"])
        B_P = np.asarray(g["FeederB_P"]); B_Q = np.asarray(g["FeederB_Q"])
        C_P = np.asarray(g["FeederC_P"]); C_Q = np.asarray(g["FeederC_Q"])
    return A_P, A_Q, B_P, B_Q, C_P, C_Q, gname

def get_n_hours_h5(h5_path: str, preferred_group: str) -> int:
    with h5py.File(h5_path, "r") as hf:
        gname = _pick_group(hf, preferred_group)
        return int(hf[gname]["FeederA_P"].shape[0])

def schema_map(h5_path: str, preferred_group: str):
    """For checks: {key: (shape, dtype)}"""
    out = {}
    with h5py.File(h5_path, "r") as hf:
        gname = _pick_group(hf, preferred_group)
        g = hf[gname]
        for k in REQUIRED_KEYS:
            ds = g[k]
            out[k] = (tuple(ds.shape), str(ds.dtype))
    return out

# ===========================
# 2017: MAT vs H5 comparison
# ===========================
def compare_2017_mat_vs_h5():
    """Print a compact comparison report for 2017 MAT vs 2017 H5."""
    print("\n=== Sanity check: 2017 MAT vs 2017 H5 ===")
    # MAT (2017)
    m_A_P = load_mat_v7p3(os.path.join(mat_dir, "FeederA_P.mat"), "FeederA_P")
    m_A_Q = load_mat_v7p3(os.path.join(mat_dir, "FeederA_Q.mat"), "FeederA_Q")
    m_B_P = load_mat_v7p3(os.path.join(mat_dir, "FeederB_P.mat"), "FeederB_P")
    m_B_Q = load_mat_v7p3(os.path.join(mat_dir, "FeederB_Q.mat"), "FeederB_Q")
    m_C_P = load_mat_v7p3(os.path.join(mat_dir, "FeederC_P.mat"), "FeederC_P")
    m_C_Q = load_mat_v7p3(os.path.join(mat_dir, "FeederC_Q.mat"), "FeederC_Q")

    # H5 (2017)
    h5_2017 = os.path.join(h5_dir, "original_data.h5")
    A_P, A_Q, B_P, B_Q, C_P, C_Q, gname = load_year_h5(h5_2017, "2017")

    pairs = [
        ("FeederA_P", m_A_P, A_P), ("FeederA_Q", m_A_Q, A_Q),
        ("FeederB_P", m_B_P, B_P), ("FeederB_Q", m_B_Q, B_Q),
        ("FeederC_P", m_C_P, C_P), ("FeederC_Q", m_C_Q, C_Q),
    ]

    def _summary(name, X, Y):
        same_shape = X.shape == Y.shape
        max_abs    = float(np.nanmax(np.abs(X - Y))) if same_shape else float("inf")
        mean_abs   = float(np.nanmean(np.abs(X - Y))) if same_shape else float("inf")
        p99_abs    = float(np.nanpercentile(np.abs(X - Y), 99)) if same_shape else float("inf")
        print(f" - {name:<10} shape_mat={X.shape} shape_h5={Y.shape} "
              f"| max|Δ|={max_abs:.6g}  mean|Δ|={mean_abs:.6g}  p99|Δ|={p99_abs:.6g}")

    print(f"Using H5 group: '{gname}'")
    for nm, X, Y in pairs:
        _summary(nm, X, Y)
    print("=== End sanity check ===\n")

# ===========================
# GLOBALS propagated to workers
# ===========================
YEAR_LABEL = None
FA_P = FA_Q = FB_P = FB_Q = FC_P = FC_Q = None
N_HOURS = None
OUT_DIR_YEAR = None  # per-year output folder

# Per-process OpenDSS engine (reused across bus tasks in that worker)
_DSSOBJ = None
_DSSText = None
_DSSCircuit = None

def _init_year_worker(h5_path: str, preferred_group: str, year_label: str, out_dir_year: str):
    """Initializer for worker processes. Loads the year's P/Q arrays once and starts OpenDSS once."""
    global YEAR_LABEL, FA_P, FA_Q, FB_P, FB_Q, FC_P, FC_Q, N_HOURS, OUT_DIR_YEAR
    global _DSSOBJ, _DSSText, _DSSCircuit

    # Load year arrays
    FA_P, FA_Q, FB_P, FB_Q, FC_P, FC_Q, gname = load_year_h5(h5_path, preferred_group)
    YEAR_LABEL   = year_label
    N_HOURS      = int(FA_P.shape[0])
    OUT_DIR_YEAR = out_dir_year

    # Start OpenDSS engine once per worker
    _DSSOBJ     = win32com.client.Dispatch("OpenDSSEngine.DSS")
    _DSSOBJ.Start(0)
    _DSSText    = _DSSOBJ.Text
    _DSSCircuit = _DSSOBJ.ActiveCircuit
    _DSSText.Command = f'Compile "{dss_file}"'
    # Optional: can improve determinism/speed if controls are not essential
    # _DSSText.Command = "Set ControlMode=Static"
    # _DSSText.Command = "Set AutoShow=No"

# ===========================
# Core worker: one target bus
# ===========================
def compute_ev_hc(target_bus: int):
    """Compute per-hour EV hosting capacity for one load bus using the per-process OpenDSS instance.
       Uses exponential search + bisection for fewer Solve() calls."""
    global YEAR_LABEL, FA_P, FA_Q, FB_P, FB_Q, FC_P, FC_Q, N_HOURS, OUT_DIR_YEAR
    global _DSSOBJ, _DSSText, _DSSCircuit

    # pick feeder matrices & offset
    if 1001 < target_bus < 2000:
        P_mat, Q_mat, offset = FA_P, FA_Q, offsetA
    elif 2001 < target_bus < 3000:
        P_mat, Q_mat, offset = FB_P, FB_Q, offsetB
    else:
        P_mat, Q_mat, offset = FC_P, FC_Q, offsetC

    loads = _DSSCircuit.Loads

    def apply_base_loads(t: int):
        # Feeder A
        for b in FeederA_buses:
            loads.Name = f"Load_{b}"
            loads.kW   = float(FA_P[t, b - offsetA])
            loads.kvar = float(FA_Q[t, b - offsetA])
        # Feeder B
        for b in FeederB_buses:
            loads.Name = f"Load_{b}"
            loads.kW   = float(FB_P[t, b - offsetB])
            loads.kvar = float(FB_Q[t, b - offsetB])
        # Feeder C
        for b in FeederC_buses:
            loads.Name = f"Load_{b}"
            loads.kW   = float(FC_P[t, b - offsetC])
            loads.kvar = float(FC_Q[t, b - offsetC])

    def violates(base_kW: float, base_kVar: float, extra_kW: float) -> bool:
        # modify only target bus demand
        loads.Name = f"Load_{target_bus}"
        loads.kW   = base_kW + extra_kW
        loads.kvar = base_kVar
        _DSSText.Command = "Solve"
        if not _DSSTCircuit.Solution.Converged if False else None:  # placeholder to satisfy linter
            pass  # replaced below
        if not _DSSCircuit.Solution.Converged:
            return True
        Vpu = np.array(_DSSCircuit.AllBusVmagPu, dtype=float)
        if (Vpu < voltage_min).any() or (Vpu > voltage_max).any():
            return True
        # transformer overload check (winding 1 rating)
        xf = _DSSCircuit.Transformers
        i = xf.First
        while i:
            _DSSCircuit.SetActiveElement(f"Transformer.{xf.Name}")
            try:
                xf.Wdg = 1
                rated = float(xf.kva)
            except Exception:
                rated = None
            P, Q = _DSSCircuit.ActiveCktElement.Powers[:2]
            S = (P*P + Q*Q) ** 0.5
            if rated is not None and S > rated + 1e-6:
                return True
            i = xf.Next
        return False

    def find_capacity(base_kW: float, base_kVar: float) -> float:
        # Early fail: even base violates
        if violates(base_kW, base_kVar, 0.0):
            return 0.0
        # Exponential search for an upper bound
        lo, hi = 0.0, ev_step_kw
        while hi <= MAX_EV_KW and not violates(base_kW, base_kVar, hi):
            lo, hi = hi, hi * 2.0
        # Bisection between lo (ok) and hi (violate or cap)
        while (hi - lo) > TOL_KW:
            mid = 0.5 * (lo + hi)
            if violates(base_kW, base_kVar, mid):
                hi = mid
            else:
                lo = mid
        return lo

    hc = np.zeros(N_HOURS, dtype=np.float32)
    for t in range(N_HOURS):
        apply_base_loads(t)
        idx       = target_bus - offset
        base_kW   = float(P_mat[t, idx])
        base_kVar = float(Q_mat[t, idx])
        hc[t]     = find_capacity(base_kW, base_kVar)

    # save per bus into the per-year folder
    out = os.path.join(OUT_DIR_YEAR, f"EV_HC_Bus{target_bus}.npz")
    np.savez(out,
             bus_number=target_bus,
             year=YEAR_LABEL,
             ev_hosting_capacity=hc,
             step_kw=ev_step_kw,
             tol_kw=TOL_KW,
             voltage_limits=(voltage_min, voltage_max))
    return target_bus

# ===========================
# Driver for one year
# ===========================
def run_year(h5_path: str, preferred_group: str, year_label: str):
    print(f"\n=== EV-HC: Year {year_label} from {h5_path} (group={preferred_group}) ===")
    try:
        n_hours = get_n_hours_h5(h5_path, preferred_group)
        print(f"Detected hours: {n_hours}")
    except Exception as e:
        print(f"!! Could not read {year_label}: {e}")
        return

    # per-year output folder
    out_dir_year = os.path.join(results_base, year_label)
    os.makedirs(out_dir_year, exist_ok=True)

    # resume logic: check existing files in this year's folder
    done = set()
    pat = re.compile(r"EV_HC_Bus(\d+)\.npz$")
    for fn in os.listdir(out_dir_year):
        m = pat.match(fn)
        if m:
            done.add(int(m.group(1)))

    todo = [b for b in all_load_buses if b not in done]
    print(f"{len(done)} buses already done for {year_label}, {len(todo)} remaining.")

    if not todo:
        print(f"🎉 {year_label}: nothing to do.")
        return

    # Right-size worker count for COM/OpenDSS
    workers = max(4, os.cpu_count() // 2)  # e.g., 12 on a 24-core CPU
    print(f"→ Running {year_label} on {len(todo)} buses with {workers} workers…")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_year_worker,
        initargs=(h5_path, preferred_group, year_label, out_dir_year),
    ) as exe:
        futures = {exe.submit(compute_ev_hc, bus): bus for bus in todo}
        for f in tqdm(as_completed(futures),
                      total=len(futures),
                      unit="bus",
                      desc=f"EV-HC {year_label}"):
            bus = f.result()
            tqdm.write(f"✓ {year_label}: Bus {bus} done")

    print(f"✅ {year_label}: complete.")

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    h5_files = sorted(Path(h5_dir).glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {h5_dir}")

    for h5_path in h5_files:
        label = guess_label_from_name(str(h5_path))
        try:
            run_year(str(h5_path), None, label)
        except Exception as exc:
            print(f"Failed for {h5_path.name}: {exc}")
