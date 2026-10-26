"""Configuration loading shared by every fleet node."""

from __future__ import annotations

import os

import yaml


def resolve_config_path(explicit: str = "") -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("WAREHOUSE_CONFIG", "")
    if env:
        candidates.append(env)
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("warehouse_fleet")
        candidates.append(os.path.join(share, "config", "warehouse.yaml"))
    except Exception:
        pass
    candidates.extend(
        [
            os.path.join(os.getcwd(), "config", "warehouse.yaml"),
            "/home/arun/warehouse_robotics_isaac/config/warehouse.yaml",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "warehouse.yaml not found. Pass the 'config' parameter or set "
        "WAREHOUSE_CONFIG. Searched: " + ", ".join(c for c in candidates if c)
    )


def load_config(explicit: str = "") -> dict:
    path = resolve_config_path(explicit)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = path
    return cfg


def stations_by_name(cfg: dict) -> dict:
    return {s["name"]: s for s in cfg["scene"]["stations"]}


def amr_specs(cfg: dict) -> list:
    return list(cfg["amrs"])
