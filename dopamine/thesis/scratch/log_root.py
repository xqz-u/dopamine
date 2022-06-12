import logging

from thesis import utils
from thesis.scratch import log_submodule, log_submodule_new

utils.setup_root_logging()

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="(PID=%(process)s) [%(asctime)s] [%(levelname)-8s] -- %(message)s -- (%(name)s:%(lineno)s)",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

logging.info("I will call my pal...")
logging.debug("And I wont print")
logging.info(
    "I am the root logging! aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

log_submodule.gimme_logs(3)
log_submodule_new.gimme_logs_zio(4)


x = log_submodule.Pippo()
x.do()

y = log_submodule_new.Ciccio()
y.do()
