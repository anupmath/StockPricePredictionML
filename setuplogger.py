import datetime
import logging


logger_name = "stock_price_prediction"
# create logger for stock price prediction project
logger = logging.getLogger("{}".format(logger_name))
logger.setLevel(logging.DEBUG)
# create file handler which logs debug messages
# log_file_name = "{}_{}".format(logger_name, datetime.datetime.now().strftime("%Y-%B-%d_%H-%M-%S"))
log_file_name = logger_name
fh = logging.FileHandler('{}.log'.format(log_file_name))
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
