"""Local paths for PMRL training.

Set the environment variables below before launching training, or edit this
file for a fixed server installation.
"""

import os
from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]
        self.workspace_dir = os.environ.get("PMRL_WORKSPACE", str(project_root))
        self.tensorboard_dir = os.environ.get(
            "PMRL_TENSORBOARD_DIR", str(project_root / "output" / "tensorboard")
        )
        self.pretrained_networks = os.environ.get(
            "PMRL_PRETRAINED_DIR", str(project_root / "pretrained_models")
        )
        self.lasher_unaligned_dir = os.environ.get("LASHER_UNALIGNED_DIR", "")
        self.luart_dir = os.environ.get("LUART_DIR", "")
        self.use_lmdb = False
