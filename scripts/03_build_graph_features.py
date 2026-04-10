"""Extract enriched graph edge features from the active OpenDSS feeder model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import win32com.client

from common import ensure_dir, project_root_from_script, repo_paths

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

MASTER_DSS = PATHS["master_dss"]
ML_READY_DIR = PATHS["ml_ready"]
GRAPH_DIR = ensure_dir(PATHS["graph"] / "base")


def bus_base(name: str) -> str:
    return str(name).split(".")[0]


def units_to_km_factor(unit_str: str) -> float:
    unit = (unit_str or "").strip().lower()
    return {
        "km": 1.0,
        "m": 0.001,
        "mi": 1.609344,
        "kft": 0.3048,
        "ft": 0.0003048,
        "in": 0.0000254,
        "cm": 0.00001,
        "none": 1.0,
        "": 1.0,
    }.get(unit, 1.0)


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def query(text, cmd):
    text.Command = cmd
    return text.Result


def load_reference_bus_index():
    for candidate in [ML_READY_DIR / "base" / "bus_index.json", ML_READY_DIR / "s1" / "bus_index.json"]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data["bus_list"], data["bus_index"]
    raise FileNotFoundError(f"Could not find bus_index.json under {ML_READY_DIR}")


def compile_dss():
    dss_obj = win32com.client.Dispatch("OpenDSSEngine.DSS")
    if not dss_obj.Start(0):
        raise RuntimeError("Failed to start OpenDSS COM engine.")
    dss_text = dss_obj.Text
    dss_circuit = dss_obj.ActiveCircuit
    dss_text.Command = f'Compile "{MASTER_DSS}"'
    if not dss_circuit:
        raise RuntimeError("Failed to compile the feeder model.")
    return dss_obj, dss_text, dss_circuit


def extract_edges(bus_index, dss_text, dss_circuit):
    reg_xfmrs = set()
    rc = dss_circuit.RegControls
    ptr = rc.First
    while ptr:
        try:
            elem = rc.Element
            if elem and elem.lower().startswith("transformer."):
                reg_xfmrs.add(elem.split(".", 1)[1])
        except Exception:
            pass
        ptr = rc.Next

    columns = [
        "is_line",
        "is_xfmr",
        "is_regulator",
        "phases",
        "kv_from",
        "kv_to",
        "length_km",
        "r1_ohm_per_km",
        "x1_ohm_per_km",
        "norm_amps",
        "emerg_amps",
        "rating_kva",
        "xhl_percent",
    ]

    edges = []
    attrs = []

    def get_kvbase(busname: str) -> float:
        try:
            dss_circuit.SetActiveBus(busname)
            return safe_float(dss_circuit.ActiveBus.kVBase, np.nan)
        except Exception:
            return np.nan

    lines = dss_circuit.Lines
    ptr = lines.First
    while ptr:
        try:
            name = lines.Name
            dss_circuit.SetActiveElement(f"Line.{name}")
            buses = list(dss_circuit.ActiveCktElement.BusNames or [])
            if len(buses) < 2:
                ptr = lines.Next
                continue
            bus_u = bus_base(buses[0])
            bus_v = bus_base(buses[1])
            if bus_u not in bus_index or bus_v not in bus_index:
                ptr = lines.Next
                continue

            try:
                phases = int(lines.Phases)
            except Exception:
                phases = max(1, (len(dss_circuit.ActiveCktElement.CurrentsMagAng) // 2) // 2)

            length = safe_float(lines.Length, 0.0)
            units = query(dss_text, f"? Line.{name}.Units") or ""
            km_factor = units_to_km_factor(units)
            length_km = length * km_factor

            r1 = safe_float(lines.R1, np.nan)
            x1 = safe_float(lines.X1, np.nan)
            r1_per_km = r1 / (km_factor if km_factor > 0 else 1.0)
            x1_per_km = x1 / (km_factor if km_factor > 0 else 1.0)

            features = [
                1.0,
                0.0,
                0.0,
                float(phases),
                float(get_kvbase(bus_u)),
                float(get_kvbase(bus_v)),
                float(length_km),
                float(r1_per_km),
                float(x1_per_km),
                float(safe_float(lines.NormAmps, np.nan)),
                float(safe_float(lines.EmergAmps, np.nan)),
                0.0,
                0.0,
            ]

            u = bus_index[bus_u]
            v = bus_index[bus_v]
            edges.extend([(u, v), (v, u)])
            attrs.extend([features, features])
        except Exception:
            pass
        ptr = lines.Next

    xfmrs = dss_circuit.Transformers
    ptr = xfmrs.First
    while ptr:
        try:
            name = xfmrs.Name
            dss_circuit.SetActiveElement(f"Transformer.{name}")
            buses = list(dss_circuit.ActiveCktElement.BusNames or [])
            if len(buses) < 2:
                ptr = xfmrs.Next
                continue
            bus_u = bus_base(buses[0])
            bus_v = bus_base(buses[1])
            if bus_u not in bus_index or bus_v not in bus_index:
                ptr = xfmrs.Next
                continue

            is_reg = 1.0 if name in reg_xfmrs else 0.0
            phases = safe_float(query(dss_text, f"? Transformer.{name}.phases"), 3.0)
            norm_amps = safe_float(query(dss_text, f"? Transformer.{name}.NormHkVA"), np.nan)
            rating_kva = safe_float(query(dss_text, f"? Transformer.{name}.kVA"), np.nan)
            xhl = safe_float(query(dss_text, f"? Transformer.{name}.Xhl"), np.nan)

            features = [
                0.0,
                1.0,
                is_reg,
                float(phases),
                float(get_kvbase(bus_u)),
                float(get_kvbase(bus_v)),
                0.0,
                0.0,
                0.0,
                float(norm_amps),
                float(norm_amps),
                float(rating_kva),
                float(xhl),
            ]

            u = bus_index[bus_u]
            v = bus_index[bus_v]
            edges.extend([(u, v), (v, u)])
            attrs.extend([features, features])
        except Exception:
            pass
        ptr = xfmrs.Next

    edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(attrs, dtype=np.float32), dtype=torch.float32)
    return edge_index, edge_attr, columns


def main():
    _, bus_index = load_reference_bus_index()
    _, dss_text, dss_circuit = compile_dss()
    edge_index, edge_attr, columns = extract_edges(bus_index, dss_text, dss_circuit)

    torch.save(edge_index, GRAPH_DIR / "edge_index.pt")
    torch.save(edge_attr, GRAPH_DIR / "edge_attr.pt")
    with open(GRAPH_DIR / "graph_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"E": int(edge_attr.shape[0]), "D": int(edge_attr.shape[1]), "columns": columns}, fh, indent=2)

    mu = edge_attr.mean(dim=0).numpy()
    sd = edge_attr.std(dim=0).numpy()
    np.savez(GRAPH_DIR / "edge_attr_stats.npz", mu=mu, sd=sd)

    print(f"Saved graph files under {GRAPH_DIR}")


if __name__ == "__main__":
    main()
