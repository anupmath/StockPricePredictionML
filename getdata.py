import json
import os
import pandas as pd
import quandl
from setuplogger import logger


class GetData(object):

  """
  This class pulls stock data from quandl
  Attributes:
    stock_data_path (str): path to the stock data csv file
    stock_data_info (dict): dictionary containing stock ticker symbols and features
    _stock_ticker_list (list): list of stock ticker symbols
  """

  def __init__(self, **kwargs):
    """
    Constructor for the class. Configures the quandl api key and parses a json file to get the stock tickers
    :param kwargs: dictionary containing "api_key" as key and the quandl api key as the value.
    """
    quandl.ApiConfig.api_key = kwargs["api_key"]
    self.stock_data_path = "stockdata/stockdata.csv"
    self.stock_data_info = self.get_stock_data_info()
    self._stock_ticker_list = self.get_stock_ticker_list()

  def get_stock_data(self, update_data=False):
    """
    Get stock data for the ticker symbols in the json file (stockdata/stockdatainfo.json) from quandl
    :param update_data: bool, tells the function whether to pull data everytime or not.
    :return df: Dataframe
    """
    logger.info("----------------Getting stock data from Quandl----------------")
    logger.info("Stock ticker list = {}".format(self._stock_ticker_list))
    # df = quandl.get("WIKI/GOOGL")
    # Pull data if stockdata/stockdata.csv does not exist or if update_data is True.
    if update_data or not os.path.exists("{}/{}".format(os.getcwd(), self.stock_data_path)):
      df = quandl.get(self._stock_ticker_list)
      logger.info("Writing stock data to {}".format(self.stock_data_path))
      # Write the dataframe to a csv fle
      df.to_csv("{}".format(self.stock_data_path))
    logger.info("Reading stock data from {}".format(self.stock_data_path))
    # df = pd.read_csv("{}".format(self.stock_data_path), index_col="Date")
    # Read the data from the csv file
    df = pd.read_csv("{}".format(self.stock_data_path))
    logger.debug("df.shape = {}".format(df.shape))
    return df

  @staticmethod
  def get_stock_data_info():
    """
    Parse the stockdata/stockdatainfo.json file and store it in a dictionary
    :return stock_data_info: dict
    """
    with open("stockdata/stockdatainfo.json", 'r') as ticker_list_f:
      ticker_list_str = ticker_list_f.read()
      stock_data_info = json.loads(ticker_list_str)
    return stock_data_info

  def get_stock_ticker_list(self):
    """
    Get the stock ticker list from the dictionary that has the parsed json file
    :return stock_ticker_list: list
    """
    stock_ticker_list = []
    for ticker_domain, ticker_list in self.stock_data_info["stock_tickers"].items():
      stock_tickers = sorted(ticker_list, key=str.lower)
      stock_tickers = list(map(lambda stock_ticker: "{}/{}".format(ticker_domain, stock_ticker), stock_tickers))
      stock_ticker_list += stock_tickers
    return stock_ticker_list

  @property
  def stock_ticker_list(self):
    """
    Returns the stock ticker list.
    :return _stock_ticker_list: list.
    """
    logger.debug("Getting stock ticker list")
    return self._stock_ticker_list
