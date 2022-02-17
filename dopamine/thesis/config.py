import os
from pathlib import Path

# TODO either make all strings or utils.data_dir_from_conf should
# accept Path too
base_dir = Path(os.path.dirname(__file__))
data_dir = Path(base_dir.parent.parent, "resources", "data")
aim_dir = data_dir
dopamine_dir = Path(base_dir.parent, "dopamine")
test_dir = base_dir.joinpath("tests")
