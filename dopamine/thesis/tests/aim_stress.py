import aim
import numpy as np

path = "/home/xqz-u/uni/thesis/dopamine/thesis/tests"
redundancy = 1
epochs = 5
min_steps = 1000

for redund in range(redundancy):
    run = aim.Run(repo=path, experiment=f"redund_{redund}")
    for epoch in range(epochs):
        epoch_steps = 0
        while epoch_steps <= min_steps:
            step = 0
            done = False
            while not done:
                run.track(
                    np.random.rand(),
                    name="random_val",
                    step=step,
                    epoch=epoch,
                    # context={"subset": "train"},
                )
                if step > min_steps // 2:
                    done = bool(np.random.randint(0, 2))
                step += 1
            print(f"{redund}-{epoch}: {step} , {epoch_steps}")
            epoch_steps += step


res = aim.Repo(path)
df = []
# query = "metric.name == 'random_val' and run.experiment == 'pippone'"
query = "metric.name == 'random_val' and run.experiment == 'redund_0'"
for run in res.query_metrics(query).iter_runs():
    el = run.dataframe()
    if el is not None:
        df.append(el)


# aim_logger = aim.Run(repo=path, experiment="pippo")
# aim_logger.track(2, "ciccio", step=2002, epoch=1)
# aim_logger.track(4, "ciccio", step=2002, epoch=2)

# repo = aim.Repo(path)
# query = "metric.name == 'ciccio'"
# df = []
# for run in repo.query_metrics(query).iter_runs():
#     el = run.dataframe()
#     if el is not None:
#         df.append(el)
