import os

from aim import Repo
from thesis import config


def main():
    bad_hash = "test_pong@<class 'thesis.agents.DQNAgent.DQNAgent'>@ALE/Pong@v5@0"
    repo = Repo.from_path(str(config.data_dir))

    new_repo_path = os.path.join(config.data_dir, "new_repo")
    new_repo = Repo.from_path(new_repo_path)
    runs_to_copy = [run_.hash for run_ in repo.iter_runs() if run_.hash != bad_hash]
    print(f"Copy runs: {runs_to_copy} from {repo.path} to {new_repo.path}")
    for run_hash in runs_to_copy:
        print(f"copying run {run_hash}...")
        repo.copy_runs([run_hash], new_repo)


# main()
