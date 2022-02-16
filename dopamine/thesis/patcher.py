import os
from pathlib import Path
from typing import Tuple

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains.checkpointer import Checkpointer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Checkpointer


def get_latest_ckpt_number(base_directory: str) -> Tuple[int, int]:
    redundancy_dirs = Path(base_directory).glob("redundancy_*")
    redundancies = [int(r.split("_")[1]) for r in (x.name for x in redundancy_dirs)]
    if not redundancies:
        return -1, -1
    latest_redundancy = max(redundancies)
    return (
        latest_redundancy,
        checkpointer.get_latest_checkpoint_number(
            os.path.join(base_directory, f"redundancy_{latest_redundancy}")
        ),
    )


def setup_redundancy(self, redund: int):
    # -1 is given when the experiment just started and signals that no
    # reloading can be performed
    redund = 0 if redund == -1 else redund
    self._base_directory = os.path.join(self.ckpt_dir, f"redundancy_{redund}")
    os.makedirs(self._base_directory, exist_ok=True)


def init(self, *args, **kwargs):
    og_init(self, *args, **kwargs)
    self.ckpt_dir = self._base_directory


Checkpointer.ckpt_dir: str = ""
og_init = Checkpointer.__init__
Checkpointer.__init__ = init
Checkpointer.setup_redundancy = setup_redundancy
