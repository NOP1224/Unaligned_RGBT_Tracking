"""Project-local paths used by the lightweight PMRL evaluator."""

import os
from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    project_root = Path(__file__).resolve().parents[3]
    settings.prj_dir = os.environ.get("PMRL_PROJECT_DIR", str(project_root))
    settings.save_dir = os.environ.get("PMRL_OUTPUT_DIR", str(project_root / "output"))
    settings.network_path = str(Path(settings.save_dir) / "networks")
    settings.results_path = str(Path(settings.save_dir) / "tracking_results")
    settings.result_plot_path = str(Path(settings.save_dir) / "result_plots")
    settings.segmentation_path = str(Path(settings.save_dir) / "segmentation_results")
    return settings
