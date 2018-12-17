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
    self._stock_ticker_list = self.get_stock_ticker_list()

  def get_stock_data(self, update_data=False):
    logger.info("----------------Getting stock data from Quandl----------------")
    self.get_stock_ticker_list()
    logger.info("Stock ticker list = {}".format(self._stock_ticker_list))
    # df = quandl.get("WIKI/GOOGL")
    if update_data or not os.path.exists("{}/{}".format(os.getcwd(), self.stock_data_path)):
      df = quandl.get(self._stock_ticker_list)
      logger.info("Writing stock data to {}".format(self.stock_data_path))
      df.to_csv("{}".format(self.stock_data_path))
    logger.info("Reading stock data from {}".format(self.stock_data_path))
    # df = pd.read_csv("{}".format(self.stock_data_path), index_col="Date")
    df = pd.read_csv("{}".format(self.stock_data_path))
    logger.debug("df.shape = {}".format(df.shape))
    return df

  @staticmethod
  def get_stock_data_info():
    with open("stockdata/stockdatainfo.json", 'r') as ticker_list_f:
      ticker_list_str = ticker_list_f.read()
      stock_data_info = json.loads(ticker_list_str)
    return stock_data_info

  def get_stock_ticker_list(self):
    stock_ticker_list = []
    for ticker_domain, ticker_list in self.stock_data_info["stock_tickers"].items():
      stock_tickers = sorted(ticker_list, key=str.lower)
      stock_tickers = list(map(lambda stock_ticker: "{}/{}".format(ticker_domain, stock_ticker), stock_tickers))
      stock_ticker_list += stock_tickers
    return stock_ticker_list

  @property
  def stock_ticker_list(self):
    logger.debug("Getting stock ticker list")
    return self._stock_ticker_list
