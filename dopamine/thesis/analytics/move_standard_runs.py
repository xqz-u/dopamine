import os

import aim
from thesis import constants

repo = aim.Repo.from_path(str(constants.aim_dir))
new_repo = aim.Repo.from_path(os.path.join(constants.data_dir, "symposium"))


# FIXME once runs are copied, the aim UI does not display metrics
# anymore, it freezes but there are no errors printed on the backend
standard_algos_hashes = [
    r.hash
    for r in repo.iter_runs()
    if "offline_v1" in r.get("hparams", {}).get("experiment_name", {})
]
# [
#     "0628051d7e434803863e3bde",
#     "09e4968883ce48dc832d0732",
#     "22ae27327f654d53aeb8ee9d",
#     "2f89f59e69c241eca676b49d",
#     "4cb9027a3cf140d5a030987b",
#     "5eeeaa6900df4d388ebc4ff8",
#     "689ba890ddfc4599b8f2348b",
#     "7c5a2fb3ab394b9d9b01fbb9",
#     "923bec69bd794d9fa3e35a97",
#     "950c070056164eaa998dd137",
#     "98567ac5b429469b89390533",
#     "bc328a85723541c4be728028",
#     "c83ac7ffb9274771a0e56341",
#     "d27a8b11677d4df49f2389c6",
#     "e435faa3e4334824889043c5",
#     "e46437fa6dd6464788e49f24",
#     "e6e01694f89340968522f8b8",
#     "e989a85897ae4e7386a3824e",
# ]

if __name__ == "__main__":
    repo.copy_runs(standard_algos_hashes, new_repo)
