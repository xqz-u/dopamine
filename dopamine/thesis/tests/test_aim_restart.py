import os
import random

import aim
from thesis import config

repo = os.path.join(config.base_dir, "tests", "aim_restart")

# give run hash to resume experiment!
run = aim.Run(repo=repo, experiment="aim_restart", run_hash="pippo")

# for i in range(10):
for i in range(10, 20):
    run.track(random.random(), "random", step=i)
