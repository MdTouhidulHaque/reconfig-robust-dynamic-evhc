"""Run quasi-static time-series simulation and save cleaned QSTS outputs."""

import os
import re
import math
import json
import glob
import numpy as np
import h5py
import win32com.client
from multiprocessing import get_context, cpu_count

from pathlib import Path

from common import ensure_dir, project_root_from_script, repo_paths, normalize_scenario

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

master_dss = str(PATHS["master_dss"])
data_dir = str(PATHS["raw"])
qsts_root = ensure_dir(PATHS["qsts"])

# Voltage limits for violation checks
voltage_min, voltage_max = 0.95, 1.05

# Load-bus whitelists (same as before). Keep or update as needed.
FeederA_buses = list(range(1003, 1018))
FeederB_buses = (
    [2002,2003,2005]
    + list(range(2008,2012)) + list(range(2014,2019))
    + [2020] + list(range(2022,2026))
    + list(range(2028,2033))
    + [2034,2035,2037] + list(range(2040,2044))
    + list(range(2045,2057)) + list(range(2058,2061))
)
FeederC_buses = (
    [3002,3004]
    + list(range(3006,3008)) + list(range(3009,3015))
    + list(range(3016,3022)) + list(range(3023,3030))
    + list(range(3031,3040)) + list(range(3041,3046))
    + list(range(3047,3053)) + [3054]
    + list(range(3056,3068)) + list(range(3070,3075))
    + [3077,3078,3081] + list(range(3083,3092))
    + list(range(3093,3100)) + list(range(3101,3107))
    + list(range(3108,3113)) + list(range(3114,3118))
    + list(range(3120,3133)) + list(range(3134,3139))
    + list(range(3141,3156)) + list(range(3157,3163))
)
all_load_buses = set(map(str, FeederA_buses + FeederB_buses + FeederC_buses))
offsetA, offsetB, offsetC = 1001, 2001, 3001  # index offsets

# ================== H5 helpers ==================
LABEL_RX = re.compile(r"(scenario[-_]?\d+|base|original|load_data)", re.IGNORECASE)

def guess_label_from_name(fname: str):
    """Infer an output label from the input filename."""
    stem = Path(fname).stem.lower()
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
    if "base" in stem or "original" in stem or "load_data" in stem:
        return "base"
    return stem.replace("synthetic_", "").replace("_feeders", "")

def resolve_group_name(h5_path: str, prefer: str | None):
    """
    Decide which group to read:
      1) If 'prefer' (e.g., '2018') exists, use it
      2) If exactly one 4-digit group exists, use it
      3) else: read from root (None)
    """
    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
        if prefer in keys:
            return prefer
        digit_groups = [k for k in keys if k.isdigit() and len(k) == 4 and isinstance(hf[k], h5py.Group)]
        if len(digit_groups) == 1:
            return digit_groups[0]
        return None

def get_hour_count(h5_path: str, group_name: str | None):
    with h5py.File(h5_path, "r") as hf:
        g = hf[group_name] if group_name is not None else hf
        for key in ["FeederA_P", "FeederB_P", "FeederC_P"]:
            if key in g:
                return int(g[key].shape[0])
    raise RuntimeError(f"Could not determine n_hours in {h5_path}; Feeder*_P datasets missing.")

def load_profiles_slice(h5_path: str, group_name: str | None, start: int, end: int):
    with h5py.File(h5_path, "r") as hf:
        g = hf[group_name] if group_name is not None else hf
        FeederA_P  = g["FeederA_P"][start:end, :]
        FeederA_Q  = g["FeederA_Q"][start:end, :]
        FeederB_P  = g["FeederB_P"][start:end, :]
        FeederB_Q  = g["FeederB_Q"][start:end, :]
        FeederC_P  = g["FeederC_P"][start:end, :]
        FeederC_Q  = g["FeederC_Q"][start:end, :]
    return (FeederA_P, FeederA_Q, FeederB_P, FeederB_Q, FeederC_P, FeederC_Q)

# ================== Circuit metadata ==================
def collect_metadata(DSSText, DSSCircuit):
    node_list = list(DSSCircuit.AllNodeNames)
    bus_names = list(DSSCircuit.AllBusNames)
    if not node_list or not bus_names:
        raise RuntimeError("Circuit not compiled: node_list/bus_names empty.")

    # lines
    li = DSSCircuit.Lines
    line_list, line_numph, line_endpoints, line_phase_mask = [], [], [], []
    ptr = li.First
    while ptr:
        nm = li.Name
        line_list.append(nm)
        DSSCircuit.SetActiveElement(f"Line.{nm}")
        elem = DSSCircuit.ActiveCktElement
        nph = int(elem.NumPhases) if elem is not None else 1
        line_numph.append(nph)
        buses = elem.BusNames or []
        b1 = buses[0].split('.')[0] if buses else ""
        b2 = buses[1].split('.')[0] if len(buses) > 1 else b1
        line_endpoints.append((b1, b2))

        digits = ''.join(buses)
        phs = set(int(tok) for tok in digits.split('.') if tok.isdigit())
        line_phase_mask.append([int(p in phs) for p in (1, 2, 3)])
        ptr = li.Next
    max_ph = max(line_numph) if line_numph else 1

    # transformers
    xf = DSSCircuit.Transformers
    xfmr_list, xfmr_conn = [], []
    ptr = xf.First
    while ptr:
        nm = xf.Name
        xfmr_list.append(nm)
        DSSCircuit.SetActiveElement(f"Transformer.{nm}")
        buses = DSSCircuit.ActiveCktElement.BusNames or []
        if len(buses) >= 2:
            bu, bv = buses[0].split('.')[0], buses[1].split('.')[0]
        elif len(buses) == 1:
            bu = bv = buses[0].split('.')[0]
        else:
            bu = bv = ""
        xfmr_conn.append((bu, bv))
        ptr = xf.Next

    # loads + robust bus map
    ld = DSSCircuit.Loads
    load_list, load_to_bus = [], []
    ptr = ld.First
    while ptr:
        nm = ld.Name
        load_list.append(nm)
        bus1_base = ""
        try:
            DSSCircuit.SetActiveElement(f"Load.{nm}")
            bus_names_elem = DSSCircuit.ActiveCktElement.BusNames or []
            if bus_names_elem:
                bus1_base = bus_names_elem[0].split('.')[0]
        except Exception:
            pass
        if not bus1_base:
            try:
                DSSText.Command = f"? Load.{nm}.Bus1"
                q = DSSText.Result or ""
                if q:
                    bus1_base = q.split('.')[0]
            except Exception:
                pass
        load_to_bus.append((nm, bus1_base))
        ptr = ld.Next

    return {
        "node_list": node_list,
        "bus_names": bus_names,
        "line_list": line_list,
        "line_numph": line_numph,
        "max_ph": max_ph,
        "line_endpoints": line_endpoints,
        "line_phase_mask": line_phase_mask,
        "xfmr_list": xfmr_list,
        "xfmr_conn": xfmr_conn,
        "load_list": load_list,
        "load_to_bus": load_to_bus,
    }

# ================== Worker ==================
def simulate_chunk(args):
    start, end, data_file, group_name, master_dss_local, vmin, vmax = args

    # Per-process COM init
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
    if not DSSObj.Start(0):
        raise RuntimeError("OpenDSS failed to start")
    DSSText    = DSSObj.Text
    DSSCircuit = DSSObj.ActiveCircuit
    DSSText.Command = f'Compile "{master_dss_local}"'
    # DSSText.Command = "Set ControlMode=Static"   # optional determinism

    meta = collect_metadata(DSSText, DSSCircuit)
    node_list       = meta["node_list"]
    line_list       = meta["line_list"]
    line_numph      = meta["line_numph"]
    max_ph          = meta["max_ph"]
    xfmr_list       = meta["xfmr_list"]
    load_list       = meta["load_list"]

    hrs = end - start
    FeederA_P, FeederA_Q, FeederB_P, FeederB_Q, FeederC_P, FeederC_Q = load_profiles_slice(
        data_file, group_name, start, end
    )

    # Allocate locals
    local_bus_Vpu         = np.zeros((hrs, len(node_list)), dtype=np.float32)
    local_line_I_mag      = np.zeros((hrs, len(line_list), 2, max_ph), dtype=np.float32)
    local_load_elem_PQ    = np.zeros((hrs, len(load_list), 2), dtype=np.float32)
    local_xfmr_flow_PQ    = np.zeros((hrs, len(xfmr_list), 2), dtype=np.float32)
    local_taps            = np.zeros((hrs, 3), dtype=np.int16)
    local_volt_viol       = np.zeros(hrs, dtype=bool)
    local_xfmr_ov         = np.zeros(hrs, dtype=bool)
    local_bus_viol_list   = [None]*hrs
    local_xfmr_ov_details = [None]*hrs
    local_notconv         = 0
    local_ok_mask         = np.ones(hrs, dtype=bool)

    for ii in range(hrs):
        # Set loads from the feeder arrays
        for b in FeederA_buses:
            idx = b - offsetA
            p = float(FeederA_P[ii, idx]); q = float(FeederA_Q[ii, idx])
            DSSText.Command = f'Edit Load.Load_{b} kW={p:.6g} kvar={q:.6g}'
        for b in FeederB_buses:
            idx = b - offsetB
            p = float(FeederB_P[ii, idx]); q = float(FeederB_Q[ii, idx])
            DSSText.Command = f'Edit Load.Load_{b} kW={p:.6g} kvar={q:.6g}'
        for b in FeederC_buses:
            idx = b - offsetC
            p = float(FeederC_P[ii, idx]); q = float(FeederC_Q[ii, idx])
            DSSText.Command = f'Edit Load.Load_{b} kW={p:.6g} kvar={q:.6g}'

        # Solve
        DSSText.Command = "Solve"
        if not DSSCircuit.Solution.Converged:
            local_notconv               += 1
            local_ok_mask[ii]            = False
            local_volt_viol[ii]          = True
            local_xfmr_ov[ii]            = True
            local_bus_viol_list[ii]      = ["<not converged>"]
            local_xfmr_ov_details[ii]    = ["<not converged>"]
            local_taps[ii]               = [-1, -1, -1]
            continue

        # Voltages
        Vpu = np.array(DSSCircuit.AllBusVmagPu, dtype=float)
        if Vpu.size != len(node_list):
            if Vpu.size > len(node_list):
                Vpu = Vpu[:len(node_list)]
            else:
                Vpu = np.pad(Vpu, (0, len(node_list) - Vpu.size))
        local_bus_Vpu[ii, :] = Vpu
        bus_viol = [node_list[k] for k, v in enumerate(Vpu) if v < vmin or v > vmax]
        local_volt_viol[ii]     = bool(bus_viol)
        local_bus_viol_list[ii] = bus_viol

        # Line currents (per terminal, per phase)
        for L, nm in enumerate(line_list):
            DSSCircuit.SetActiveElement(f"Line.{nm}")
            cma  = DSSCircuit.ActiveCktElement.CurrentsMagAng
            mags = np.array(cma[0::2], dtype=float)  # magnitudes only
            nph  = min(line_numph[L], max_ph)
            if mags.size >= nph:
                local_line_I_mag[ii, L, 0, :nph] = mags[:nph]
            if mags.size >= 2 * nph:
                local_line_I_mag[ii, L, 1, :nph] = mags[nph: 2*nph]

        # Loads P/Q (terminal 1)
        ld = DSSCircuit.Loads
        j = ld.First
        k = 0
        while j:
            DSSCircuit.SetActiveElement(f"Load.{ld.Name}")
            Pwr = DSSCircuit.ActiveCktElement.Powers
            local_load_elem_PQ[ii, k, 0] = float(Pwr[0])
            local_load_elem_PQ[ii, k, 1] = float(Pwr[1])
            k += 1
            j = ld.Next

        # Transformer flows + overload flag
        xf = DSSCircuit.Transformers
        j = xf.First
        k = 0
        ov = False
        details = []
        while j:
            nm = xf.Name
            DSSCircuit.SetActiveElement(f"Transformer.{nm}")
            P, Q = DSSCircuit.ActiveCktElement.Powers[:2]
            P = float(P); Q = float(Q)
            local_xfmr_flow_PQ[ii, k, 0] = P
            local_xfmr_flow_PQ[ii, k, 1] = Q
            try:
                xf.Wdg = 1
                rated = float(xf.kva)
            except Exception:
                rated = None
            S = (P*P + Q*Q) ** 0.5
            if rated is not None and S > rated + 1e-6:
                ov = True
                details.append((nm, round(S, 2), rated))
            k += 1
            j = xf.Next
        local_xfmr_ov[ii]         = ov
        local_xfmr_ov_details[ii] = details

        # Regulator taps (optional names Reg_contr_A/B/C)
        try:
            for r_i, r in enumerate(['A', 'B', 'C']):
                DSSCircuit.RegControls.Name = f"Reg_contr_{r}"
                local_taps[ii, r_i] = int(DSSCircuit.RegControls.TapNumber)
        except Exception:
            local_taps[ii] = [-1, -1, -1]

    return {
        'start':start, 'end':end,
        'bus_Vpu':local_bus_Vpu,
        'line_I':local_line_I_mag,
        'load_PQ':local_load_elem_PQ,
        'xfmr_PQ':local_xfmr_flow_PQ,
        'taps':local_taps,
        'v_viol':local_volt_viol,
        'b_viol_list':local_bus_viol_list,
        'x_ov':local_xfmr_ov,
        'x_ov_details':local_xfmr_ov_details,
        'notconv':local_notconv,
        'ok_mask': local_ok_mask,
        # metadata echoes
        'node_list':meta["node_list"],
        'bus_names':meta["bus_names"],
        'line_list':meta["line_list"],
        'line_numph':meta["line_numph"],
        'max_ph':meta["max_ph"],
        'line_endpoints':meta["line_endpoints"],
        'line_phase_mask':meta["line_phase_mask"],
        'xfmr_list':meta["xfmr_list"],
        'xfmr_conn':meta["xfmr_conn"],
        'load_list':meta["load_list"],
        'load_to_bus':meta["load_to_bus"],
    }

# ================== Year runner ==================
def run_one_file(h5_path: str):
    # Infer year label from filename or group
    label_guess = guess_year_label_from_name(h5_path)
    group_name  = resolve_group_name(h5_path, label_guess)
    label       = label_guess

    # If no label yet, try to use the group name as label
    if label is None and group_name is not None and group_name.isdigit() and len(group_name) == 4:
        label = group_name
    # Still None? fall back to basename (without extension)
    if label is None:
        label = os.path.splitext(os.path.basename(h5_path))[0]

    n_hours = get_hour_count(h5_path, group_name)

    # Chunking & pool size
    N = min(6, cpu_count())   # moderate parallelism is more stable for OpenDSS COM
    step = math.ceil(n_hours / N)
    cuts = list(range(0, n_hours, step))
    if cuts[-1] != n_hours:
        cuts.append(n_hours)
    chunks = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]

    args = [(s, e, h5_path, group_name, master_dss, voltage_min, voltage_max) for (s, e) in chunks]

    ctx = get_context('spawn')
    with ctx.Pool(N) as p:
        results = p.map(simulate_chunk, args)

    # Use metadata from first chunk
    meta0 = results[0]
    node_list       = list(meta0['node_list'])
    bus_names       = list(meta0['bus_names'])
    line_list       = list(meta0['line_list'])
    line_numph      = list(meta0['line_numph'])
    max_ph          = int(meta0['max_ph'])
    line_endpoints  = list(meta0['line_endpoints'])
    line_phase_mask = np.array(meta0['line_phase_mask'], dtype=np.int8)
    xfmr_list       = list(meta0['xfmr_list'])
    xfmr_conn       = list(meta0['xfmr_conn'])
    load_list       = list(meta0['load_list'])
    load_to_bus     = list(meta0['load_to_bus'])

    # Pre-alloc
    bus_Vpu  = np.zeros((n_hours, len(node_list)), dtype=np.float32)
    line_I   = np.zeros((n_hours, len(line_list), 2, max_ph), dtype=np.float32)
    load_PQ  = np.zeros((n_hours, len(load_list), 2), dtype=np.float32)
    xfmr_PQ  = np.zeros((n_hours, len(xfmr_list), 2), dtype=np.float32)
    taps     = np.zeros((n_hours, 3), dtype=np.int16)
    v_viol   = np.zeros(n_hours, dtype=bool)
    b_viol_list = [None]*n_hours
    x_ov     = np.zeros(n_hours, dtype=bool)
    x_ov_det = [None]*n_hours
    ok_mask  = np.ones(n_hours, dtype=bool)
    notconv  = 0

    # Assemble results
    for res in results:
        s, e = res['start'], res['end']
        bus_Vpu[s:e]     = res['bus_Vpu']
        line_I[s:e]      = res['line_I']
        load_PQ[s:e]     = res['load_PQ']
        xfmr_PQ[s:e]     = res['xfmr_PQ']
        taps[s:e]        = res['taps']
        v_viol[s:e]      = res['v_viol']
        b_viol_list[s:e] = res['b_viol_list']
        x_ov[s:e]        = res['x_ov']
        x_ov_det[s:e]    = res['x_ov_details']
        ok_mask[s:e]     &= res['ok_mask']
        notconv         += res['notconv']

    # Save outputs (year-suffixed)
    out_dir = ensure_dir(qsts_root / label)
    save_npz = out_dir / f"qsts_{label}.npz"
    out = dict(
        dataset_version="v0.4-auto-all-years",
        voltage_limits=(float(voltage_min), float(voltage_max)),
        bus_Vpu=bus_Vpu,
        line_I_mag=line_I,
        load_elem_PQ=load_PQ,
        xfmr_flow_PQ=xfmr_PQ,
        tap_positions=taps,
        unconverged_steps=int(notconv),
        ok_mask=ok_mask,
        voltage_violation_flag=v_viol,
        node_voltage_violations=np.array(b_viol_list, dtype=object),
        xfmr_overload_flag=x_ov,
        xfmr_overload_details=np.array(x_ov_det, dtype=object),
        # metadata
        node_list=np.array(node_list, dtype=object),
        bus_names=np.array(bus_names, dtype=object),
        line_list=np.array(line_list, dtype=object),
        line_numph=np.array(line_numph, dtype=np.int16),
        max_phases=np.array([max_ph], dtype=np.int16),
        line_endpoints=np.array(line_endpoints, dtype=object),
        line_phase_mask=line_phase_mask,
        transformer_list=np.array(xfmr_list, dtype=object),
        transformer_conn=np.array(xfmr_conn, dtype=object),
        load_list=np.array(load_list, dtype=object),
        load_to_bus=np.array(load_to_bus, dtype=object),
        load_bus_whitelist=np.array(sorted(all_load_buses), dtype=object),
    )
    np.savez(str(save_npz), **out)
    print(f"✅ [{label}] QSTS saved → {save_npz}")

    save_json = out_dir / "metadata.json"
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "dataset_version": out["dataset_version"],
            "bus_names": bus_names,
            "load_to_bus": load_to_bus,
            "transformer_conn": xfmr_conn,
            "line_endpoints": line_endpoints,
            "voltage_limits": out["voltage_limits"]
        }, f, indent=2)
    print(f"🗺️  [{label}] Metadata maps saved → {save_json}")

    bad = (~ok_mask).sum()
    print(f"ℹ️  [{label}] Non-converged hours: {bad}/{n_hours} ({100.0*bad/max(1,n_hours):.2f}%)")

# ================== Main: run ALL .h5 in Data3 ==================
if __name__ == "__main__":
    # Find all .h5 under Data3 (e.g., original_data.h5, synthetic_2018_feeders.h5, synthetic_2019_feeders.h5, …)
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {data_dir}")

    print("Discovered H5 files:")
    for p in h5_files:
        print(" -", os.path.basename(p))

    for h5 in h5_files:
        print(f"\n=== Running QSTS for {os.path.basename(h5)} ===")
        run_one_file(h5)
