import os
from pathlib import Path

from thesis import config


def pippo(base_directory: str) -> str:
    redundancy_dirs = Path(base_directory).glob("redundancy_*")
    redundancies_gen = (x.name for x in redundancy_dirs)
    latest_redundancy = max(int(r.split("_")[1]) for r in redundancies_gen)
    latest_redundancy_dir = os.path.join(
        base_directory, f"redundancy_{latest_redundancy}"
    )
    return latest_redundancy_dir


pippo(os.path.join(config.base_dir, "tests", "checkpoints"))
