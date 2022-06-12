import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def gimme_logs(x):
    logger.info(f"info: {x}**2 = {x**2}")
    logger.debug(f"debug: {x}**2 = {x**2}")


class Pippo:
    logger: logging.Logger = logging.getLogger(__name__)

    def do(self):
        self.logger.info("log father")
