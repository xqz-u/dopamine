from thesis import constants
from thesis.experiments import pg_time_train_iter_cc


# NOTE no aim tracking
def make_conf():
    conf = pg_time_train_iter_cc.make_conf("peregrine_off_time_eval")
    conf["runner"]["experiment"]["schedule"] = "eval"
    return conf


def main():
    pg_time_train_iter_cc.dorun(constants.scratch_data_dir, constants.data_dir)


def main_peregrine():
    pg_time_train_iter_cc.dorun(
        constants.peregrine_data_dir, constants.peregrine_data_dir
    )


if __name__ == "__main__":
    main_peregrine()
    # main()
