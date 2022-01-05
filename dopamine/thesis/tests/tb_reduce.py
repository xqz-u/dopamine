from glob import glob
from typing import List

import pandas as pd
import tensorboard as tb
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator, io_wrapper

from thesis import config

indirs_mine_glob = f"{config.base_dir}/mul_runs/JaxDQN_*"
indirs_mine = glob(indirs_mine_glob)

tf_file_glob = glob(f"{indirs_mine[0]}/*")[0]
io_wrapper.IsSummaryEventsFile(tf_file_glob)
acc_tb = event_accumulator.EventAccumulator(tf_file_glob).Reload()
print(acc_tb.path)
for metric in acc_tb.Tags()["tensors"]:
    print(metric)
    for i, el in enumerate(acc_tb.Tensors(metric)):
        print(f"{i}: {tf.make_ndarray(el.tensor_proto)}")


tb_dev_ids_mine = [
    "8cGgJB9CR9S0ij0SKV8OKQ",
    "MZwdAGzyQ1eo7CxzRR23LQ",
    "06ouz8PxQZm2zes11YM0AQ",
    "tRzaFgdMQAO4AubpboWZMQ",
    "Igzj3QYYTYudhLrOcla56A",
    "HZzUEBp9Qwu2ByGT6F5Yuw",
    "oS43jtENQGOMRk5W26R5eA",
    "3DLF1RF6Q6KyWtxPobx5Vg",
    "2rSNnhWESEuATdA3Czjnow",
    "U1puXSq8RlG5LTYuREyVqQ",
]


def tb_dev_exp_data(exp_id: str) -> pd.DataFrame:
    return tb.data.experimental.ExperimentFromDev(exp_id).get_scalars()


# takes values for HuberLoss and Train/AverageReturns
def hstack_metrics(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    loss = pd.DataFrame()
    returns = pd.DataFrame()
    for i, df in enumerate(dfs):
        loss[f"HuberLoss_{i}"] = df[df["tag"] == "HuberLoss"]["value"]
        returns[f"Return_{i}"] = df[df["tag"] == "Train/AverageReturns"]["value"]
    return loss, returns


dfs_mine = [tb_dev_exp_data(eid) for eid in tb_dev_ids_mine]

loss_mine, returns_mine = hstack_metrics(dfs_mine)

loss_mine_mean = pd.DataFrame(loss_mine.mean(axis=1)).rename(columns={0: "HuberLoss"})
loss_mine_mean.plot()
plt.savefig("loss_mine_mean.png")


returns_mine_mean = pd.DataFrame(returns_mine.mean(axis=1)).rename(
    columns={0: "Train/AverageReturns"}
)
returns_mine_mean.plot()
plt.savefig("returns_mine_mean.png")
