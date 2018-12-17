import json
import os
import pandas as pd
import quandl
from setuplogger import logger


class GetData(object):

  def __init__(self, **kwargs):
    quandl.ApiConfig.api_key = kwargs["api_key"]
    self.stock_data_path = "stockdata/stockdata.csv"
    self.stock_data_info = self.get_stock_data_info()

  def get_stock_data(self, update_data=False):
    logger.info("----------------Getting stock data from Quandl----------------")
    stock_ticker_list = sorted(self.stock_data_info["stock_ticker"], key=str.lower)
    stock_ticker_list = list(map(lambda x: "WIKI/{}".format(x), stock_ticker_list))
    # df = quandl.get("WIKI/GOOGL")
    if update_data or not os.path.exists("{}/{}".format(os.getcwd(), self.stock_data_path)):
      df = quandl.get(stock_ticker_list)
      logger.info("Writing stock data to {}".format(self.stock_data_path))
      df.to_csv("{}".format(self.stock_data_path))
    logger.info("Reading stock data from {}".format(self.stock_data_path))
    # df = pd.read_csv("{}".format(self.stock_data_path), index_col="Date")
    df = pd.read_csv("{}".format(self.stock_data_path))
    logger.debug("df.shape = {}".format(df.shape))
    return df, stock_ticker_list

  @staticmethod
  def get_stock_data_info():
    with open("stockdata/stockdatainfo.json", 'r') as ticker_list_f:
      ticker_list_str = ticker_list_f.read()
      stock_data_info = json.loads(ticker_list_str)
    return stock_data_info
