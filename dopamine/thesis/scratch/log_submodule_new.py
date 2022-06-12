import logging

from thesis.scratch import log_submodule

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gimme_logs_zio(x):
    logger.info(f"info: {x}**2 = {x**2}")
    logger.debug(f"debug: {x}**2 = {x**2}")


class Ciccio(log_submodule.Pippo):
    def do(self):
        self.logger.info("log child")
