"""Run reporting for reproducibility (Reviewer 1: reproducibility).

Every stage opens a `RunReport`, records the resolved config, seeds, git commit
and environment, then dumps counts/metrics to `results/<run_id>/report.json`.
This is what makes every table and figure in the paper traceable to a run.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


class RunReport:
    """Accumulates a JSON-serialisable record of one pipeline stage."""

    def __init__(self, stage: str, config: dict[str, Any], results_dir: str, run_id: str):
        self.stage = stage
        self.run_id = run_id
        self.dir = Path(results_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {
            "stage": stage,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "env": _env_info(),
            "config": config,
            "metrics": {},
            "counts": {},
            "artifacts": {},
        }

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def add_metric(self, key: str, value: Any) -> None:
        self.data["metrics"][key] = value

    def add_count(self, key: str, value: Any) -> None:
        self.data["counts"][key] = value

    def add_artifact(self, key: str, path: str | Path) -> None:
        self.data["artifacts"][key] = str(path)

    def save(self, filename: str = "report.json") -> Path:
        out = self.dir / filename
        out.write_text(json.dumps(self.data, indent=2, default=str))
        return out
