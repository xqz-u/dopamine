import os
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))
data_dir = Path(base_dir.parent.parent, "resources", "data")
