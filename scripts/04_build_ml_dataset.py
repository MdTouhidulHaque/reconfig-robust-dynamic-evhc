"""Build ML-ready tensors from QSTS outputs and ground-truth EV hosting capacity files."""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_PROTOCOL, project_root_from_script, repo_paths, resolve_scenario_dir, scenario_candidates

PROJECT_ROOT = project_root_from_script(__file__)
PATHS = repo_paths(PROJECT_ROOT)

QSTS_ROOT = PATHS["qsts"]
GROUND_TRUTH_ROOT = PATHS["ground_truth"]
OUT_ROOT = PATHS["ml_ready"]

STRICT_BUS_INDEX = True
REF_BUS_LIST = None
SCENARIOS = ["base", "s1", "s2", "s3", "s4", "s5"]


def extract_digits(value: str) -> str | None:
    match = re.search(r"(\d+)", str(value))
    return match.group(1) if match else None


def sorted_bus_list(names):
    nums, alphas = [], []
    for bus in names:
        digits = extract_digits(bus)
        if digits:
            try:
                nums.append((int(digits), str(bus)))
            except Exception:
                alphas.append(str(bus))
        else:
            alphas.append(str(bus))
    nums.sort()
    alphas.sort()
    return [b for _, b in nums] + alphas


def build_id_to_canonical(bus_names, load_to_bus_kv):
    bus_set = set(str(b) for b in bus_names)
    id_to_canon = {}

    for bus in bus_set:
        digits = extract_digits(bus)
        if digits and digits not in id_to_canon:
            id_to_canon[digits] = bus

    for _, bus in load_to_bus_kv:
        bus = str(bus)
        digits = extract_digits(bus)
        if digits and digits not in id_to_canon and bus in bus_set:
            id_to_canon[digits] = bus

    return id_to_canon


def locate_qsts(scenario: str) -> Path | None:
    scenario_dir = resolve_scenario_dir(QSTS_ROOT, scenario)
    preferred = scenario_dir / f"qsts_{scenario}.npz"
    if preferred.exists():
        return preferred
    matches = list(scenario_dir.glob("*.npz"))
    return matches[0] if matches else None


def locate_ground_truth_dir(scenario: str) -> Path:
    return resolve_scenario_dir(GROUND_TRUTH_ROOT, scenario)


def prepare_one_dataset(tag: str, qsts_path: Path, evhc_dir: Path, out_dir: Path):
    global REF_BUS_LIST

    if qsts_path is None or not qsts_path.exists():
        raise FileNotFoundError(f"Missing QSTS file for {tag}: {qsts_path}")

    if not evhc_dir.is_dir():
        raise FileNotFoundError(f"Missing EVHC directory for {tag}: {evhc_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    q = np.load(qsts_path, allow_pickle=True)
    orig_hours = int(q["bus_Vpu"].shape[0])
    phase_nodes = list(q["node_list"])
    bus_names = list(q["bus_names"])
    load_list = list(q["load_list"])
    load_to_bus_kv = list(q["load_to_bus"])
    xfmr_conn = list(map(tuple, q["transformer_conn"]))
    bus_vpu_phase = q["bus_Vpu"]
    load_elem_pq = q["load_elem_PQ"]
    xfmr_flow_pq = q["xfmr_flow_PQ"]
    ok_mask = q["ok_mask"] if "ok_mask" in q.files else np.ones(orig_hours, dtype=bool)
    ok_mask = ok_mask.astype(bool)
    valid_idx = np.where(ok_mask)[0]

    bus_vpu_phase = bus_vpu_phase[valid_idx]
    load_elem_pq = load_elem_pq[valid_idx]
    xfmr_flow_pq = xfmr_flow_pq[valid_idx]
    hours = int(bus_vpu_phase.shape[0])

    bus_list = sorted_bus_list(bus_names)
    bus_index = {b: i for i, b in enumerate(bus_list)}
    n_buses = len(bus_list)

    if REF_BUS_LIST is None:
        REF_BUS_LIST = list(bus_list)
        print(f"[{tag}] using reference bus order with {len(REF_BUS_LIST)} buses")
    elif bus_list != REF_BUS_LIST and STRICT_BUS_INDEX:
        raise AssertionError(f"bus index mismatch for {tag}")

    id_to_canonical = build_id_to_canonical(bus_names, load_to_bus_kv)

    sum_a = np.zeros((hours, n_buses), dtype=np.float32)
    sum_b = np.zeros((hours, n_buses), dtype=np.float32)
    sum_c = np.zeros((hours, n_buses), dtype=np.float32)
    cnt_a = np.zeros(n_buses, dtype=np.int32)
    cnt_b = np.zeros(n_buses, dtype=np.int32)
    cnt_c = np.zeros(n_buses, dtype=np.int32)

    for j, node in enumerate(phase_nodes):
        parts = str(node).split(".")
        bus = parts[0].replace("bus", "")
        phase = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        if bus not in bus_index or phase is None:
            continue
        i = bus_index[bus]
        if phase == 1:
            sum_a[:, i] += bus_vpu_phase[:, j]
            cnt_a[i] += 1
        elif phase == 2:
            sum_b[:, i] += bus_vpu_phase[:, j]
            cnt_b[i] += 1
        elif phase == 3:
            sum_c[:, i] += bus_vpu_phase[:, j]
            cnt_c[i] += 1

    den_a = np.where(cnt_a > 0, cnt_a, 1).astype(np.float32)
    den_b = np.where(cnt_b > 0, cnt_b, 1).astype(np.float32)
    den_c = np.where(cnt_c > 0, cnt_c, 1).astype(np.float32)

    x_volt_a = (sum_a / den_a).astype(np.float32)
    x_volt_b = (sum_b / den_b).astype(np.float32)
    x_volt_c = (sum_c / den_c).astype(np.float32)

    phase_stacks = []
    if np.any(cnt_a):
        phase_stacks.append(np.where(cnt_a > 0, x_volt_a, np.nan))
    if np.any(cnt_b):
        phase_stacks.append(np.where(cnt_b > 0, x_volt_b, np.nan))
    if np.any(cnt_c):
        phase_stacks.append(np.where(cnt_c > 0, x_volt_c, np.nan))

    if phase_stacks:
        x_volt_mean = np.nanmean(np.stack(phase_stacks, axis=0), axis=0).astype(np.float32)
        x_volt_mean = np.nan_to_num(x_volt_mean, nan=0.0).astype(np.float32)
    else:
        x_volt_mean = np.zeros((hours, n_buses), np.float32)

    load_to_bus = dict((str(k), str(v)) for k, v in load_to_bus_kv)
    x_load_p = np.zeros((hours, n_buses), dtype=np.float32)
    x_load_q = np.zeros((hours, n_buses), dtype=np.float32)
    for j, load_name in enumerate(load_list):
        bus = load_to_bus.get(str(load_name))
        if bus is None:
            continue
        bus = str(bus)
        if bus in bus_index:
            x_load_p[:, bus_index[bus]] += load_elem_pq[:, j, 0].astype(np.float32)
            x_load_q[:, bus_index[bus]] += load_elem_pq[:, j, 1].astype(np.float32)

    x_xfmr_p = np.zeros((hours, n_buses), dtype=np.float32)
    x_xfmr_q = np.zeros((hours, n_buses), dtype=np.float32)
    for j, (bus_u, bus_v) in enumerate(xfmr_conn):
        bus_u = str(bus_u)
        bus_v = str(bus_v)
        p = xfmr_flow_pq[:, j, 0].astype(np.float32)
        qv = xfmr_flow_pq[:, j, 1].astype(np.float32)
        if bus_u in bus_index:
            x_xfmr_p[:, bus_index[bus_u]] += np.where(p >= 0, p, 0.0)
            x_xfmr_q[:, bus_index[bus_u]] += np.where(p >= 0, qv, 0.0)
        if bus_v in bus_index and bus_v != bus_u:
            x_xfmr_p[:, bus_index[bus_v]] += np.where(p < 0, -p, 0.0)
            x_xfmr_q[:, bus_index[bus_v]] += np.where(p < 0, -qv, 0.0)

    hour_index = np.arange(hours, dtype=np.int32) % 24
    sin_h = np.sin(2 * np.pi * hour_index / 24.0).astype(np.float32)
    cos_h = np.cos(2 * np.pi * hour_index / 24.0).astype(np.float32)
    sin_h = np.repeat(sin_h[:, None], n_buses, axis=1)
    cos_h = np.repeat(cos_h[:, None], n_buses, axis=1)

    x = np.stack(
        [
            x_volt_a,
            x_volt_b,
            x_volt_c,
            x_volt_mean,
            x_load_p,
            x_load_q,
            x_xfmr_p,
            x_xfmr_q,
            sin_h,
            cos_h,
        ],
        axis=-1,
    ).astype(np.float32)

    y = np.full((hours, n_buses), np.nan, dtype=np.float32)
    assigned = 0
    skipped_no_map = 0
    pattern = re.compile(r"EV_HC_Bus(\d+)\.npz$", re.IGNORECASE)

    for file_path in evhc_dir.iterdir():
        match = pattern.match(file_path.name)
        if not match:
            continue
        num_id = match.group(1)
        z = np.load(file_path, allow_pickle=True)
        hc_full = z["ev_hosting_capacity"].astype(np.float32)
        hc = hc_full[valid_idx] if hc_full.shape[0] == orig_hours else hc_full
        canonical = id_to_canonical.get(num_id)
        if canonical and canonical in bus_index:
            y[:, bus_index[canonical]] = hc
            assigned += 1
        else:
            skipped_no_map += 1

    load_bus_names = set(str(b) for _, b in load_to_bus_kv if str(b) in bus_index)
    load_bus_mask = np.array([b in load_bus_names for b in bus_list], dtype=bool)
    target_mask = (~np.isnan(y)).any(axis=0)

    torch.save(torch.from_numpy(x), out_dir / "features.pt")
    torch.save(torch.from_numpy(y), out_dir / "targets.pt")
    torch.save(torch.from_numpy(load_bus_mask), out_dir / "load_bus_mask.pt")
    torch.save(torch.from_numpy(target_mask), out_dir / "target_mask.pt")
    with open(out_dir / "bus_index.json", "w", encoding="utf-8") as fh:
        json.dump({"bus_list": bus_list, "bus_index": {b: i for i, b in enumerate(bus_list)}}, fh, indent=2)

    print(
        f"[{tag}] X={tuple(x.shape)} Y={tuple(y.shape)} assigned_labels={assigned} skipped_no_map={skipped_no_map}"
    )


def main():
    print("Preparing ML-ready tensors...")
    any_ok = False
    for scenario in SCENARIOS:
        try:
            qsts_path = locate_qsts(scenario)
            gt_dir = locate_ground_truth_dir(scenario)
            out_dir = OUT_ROOT / scenario
            prepare_one_dataset(scenario, qsts_path, gt_dir, out_dir)
            any_ok = True
        except Exception as exc:
            print(f"[{scenario}] failed: {exc}")

    if not any_ok:
        sys.exit(1)

    print(f"Done. Outputs written under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
