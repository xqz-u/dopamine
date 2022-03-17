import os
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))
dopamine_dir = Path(base_dir.parent, "dopamine")

data_dir = Path(base_dir.parent.parent, "resources", "data")
aim_dir = data_dir
scratch_data_dir = data_dir.joinpath("scratch")

scratch_dir = base_dir.joinpath("scratch")
