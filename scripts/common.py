from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

SCENARIO_MAP = {
    "base": "base",
    "Base": "base",
    "base_case": "base",
    "s1": "s1",
    "scenario-1": "s1",
    "scenario_1": "s1",
    "scenario1": "s1",
    "s2": "s2",
    "scenario-2": "s2",
    "scenario_2": "s2",
    "scenario2": "s2",
    "s3": "s3",
    "scenario-3": "s3",
    "scenario_3": "s3",
    "scenario3": "s3",
    "s4": "s4",
    "scenario-4": "s4",
    "scenario_4": "s4",
    "scenario4": "s4",
    "s5": "s5",
    "scenario-5": "s5",
    "scenario_5": "s5",
    "scenario5": "s5",
}

LEGACY_SCENARIO_NAME = {
    "base": "Base",
    "s1": "scenario-1",
    "s2": "scenario-2",
    "s3": "scenario-3",
    "s4": "scenario-4",
    "s5": "scenario-5",
}

DEFAULT_PROTOCOL = {
    "train_scenarios": ["base", "s1", "s2"],
    "val_scenarios": ["s3"],
    "test_scenarios": ["s4", "s5"],
    "window": 24,
    "train_end": 6137,
    "val_end": 7420,
    "total_steps": 8760,
}


def project_root_from_script(script_path: str) -> Path:
    return Path(script_path).resolve().parents[1]


def normalize_scenario(name: str) -> str:
    if name not in SCENARIO_MAP:
        raise KeyError(f"Unknown scenario name: {name}")
    return SCENARIO_MAP[name]


def legacy_name(name: str) -> str:
    return LEGACY_SCENARIO_NAME[normalize_scenario(name)]


def scenario_candidates(name: str) -> List[str]:
    norm = normalize_scenario(name)
    out = [norm]
    legacy = LEGACY_SCENARIO_NAME.get(norm)
    if legacy and legacy not in out:
        out.append(legacy)
    if norm == "base":
        out.extend(["base_case"])
    return out


def repo_paths(project_root: Path) -> Dict[str, Path]:
    data_root = project_root / "data"
    return {
        "project_root": project_root,
        "configs": project_root / "configs",
        "docs": project_root / "docs",
        "scripts": project_root / "scripts",
        "models": project_root / "models",
        "results": project_root / "results",
        "data_root": data_root,
        "raw": data_root / "raw",
        "raw_load_h5": data_root / "raw" / "load_data.h5",
        "dss_model_dir": data_root / "raw" / "dss_model",
        "master_dss": data_root / "raw" / "dss_model" / "Master.dss",
        "qsts": data_root / "interim" / "qsts",
        "graph": data_root / "processed" / "graph",
        "ground_truth": data_root / "processed" / "ground_truth",
        "ml_ready": data_root / "processed" / "ml_ready",
        "sample": data_root / "sample",
    }


def load_paths_config(project_root: Path, config_path: Optional[Path] = None) -> Dict[str, str]:
    defaults = {k: str(v) for k, v in repo_paths(project_root).items()}
    if config_path is None:
        config_path = project_root / "configs" / "paths.yaml"
    if not config_path.exists() or yaml is None:
        return defaults
    with open(config_path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    flat = dict(defaults)
    for key, value in user_cfg.items():
        flat[key] = str((project_root / value).resolve()) if isinstance(value, str) and not os.path.isabs(value) else str(value)
    return flat


def first_existing_dir(base_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        candidate = base_dir / name
        if candidate.is_dir():
            return candidate
    return None


def resolve_scenario_dir(base_dir: Path, scenario: str) -> Path:
    candidate = first_existing_dir(base_dir, scenario_candidates(scenario))
    if candidate is None:
        raise FileNotFoundError(f"Could not resolve scenario directory under {base_dir} for {scenario}")
    return candidate


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
