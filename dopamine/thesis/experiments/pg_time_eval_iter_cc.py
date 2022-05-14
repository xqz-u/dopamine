from thesis import constants
from thesis.experiments import peregrine_time_train_iter
from thesis.runner import runner


def make_conf():
    conf = peregrine_time_train_iter.make_conf("peregrine_off_time_eval")
    conf["runner"]["experiment"]["schedule"] = "eval"
    return conf


def main():
    conf = make_conf()
    confs = peregrine_time_train_iter.doconfs(conf, constants.data_dir)
    runner.run_experiments(confs, scratch=True)


def main_peregrine():
    runner.run_experiments(
        peregrine_time_train_iter.doconfs(make_conf(), constants.peregrine_data_dir)
    )


if __name__ == "__main__":
    main_peregrine()
