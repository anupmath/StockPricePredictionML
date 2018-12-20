import math
import numpy as np
import pandas as pd
from getdata import GetData
from setuplogger import logger


class PreprocessData(object):

	"""
	This class preprocesses the data
	Attributes:
		stock_data_info (dict): dictionary containing stock ticker symbols and features
		original_df_dict (dict): dictionary containing the original stock data frames.
		preprocessed_data_dict (dict): dictionary containing the preprocessed stock data frames.
		ticker_symbol_list (list): list of stock ticker symbols
	"""

	def __init__(self, **kwargs):
		"""
		Constructor for the class.
		"""
		self.stock_data_info = GetData.get_stock_data_info()
		self.original_df_dict = {}
		self.preprocessed_data_dict = {}
		self.ticker_symbol_list = []
		self.future_prediction_pcnt = kwargs["future_prediction_pcnt"]

	def get_df_for_each_ticker(self, df):
		"""
		Get the stock ticker as key and the corresponding dataframe as value in a dictionary
		:param df: dataframe, original dataframe consisting of stock data for all the ticker symbols.
		:return: None
		"""
		# Copy the date column to another dataframe
		df_date = df.iloc[:, 0].copy(deep=True)
		df_date = pd.DataFrame(df_date.values, columns=["Date"])
		# Drop date from original dataframe for easier separation stock data based on ticker symbols
		df = df.drop(columns=["Unnamed: 0"])
		for ticker_symbol in self.ticker_symbol_list:
			feature_list = self.get_feature_list(ticker_symbol.split("/")[0])
			ticker_symbol_columns = list(map(
				lambda x, y: "{} - {}".format(x, y), [ticker_symbol] * len(feature_list), feature_list))
			self.original_df_dict[ticker_symbol] = pd.concat([df_date.iloc[:, 0], df[ticker_symbol_columns]], axis=1)

	def get_feature_list(self, ticker_domain):
		"""
		Get the list of features from the dictionary containing stock data information.
		:param ticker_domain: str, domain from which the stock data was obtained. eg. WIKI, BCB etc.,
		:return feature_list: list, list of features.
		"""
		feature_list = self.stock_data_info["features_list"][ticker_domain]
		return feature_list

	@staticmethod
	def get_high_to_low_pcnt_change(df, ticker_symbol):
		"""
		Get the percentage change between high and low prices of a stock.
		:param df: dataframe
		:param ticker_symbol: str
		:return df: dataframe
		"""
		df["{} - HL_PCT".format(ticker_symbol)] = (df["{} - Adj. High".format(ticker_symbol)] - df["{} - Adj. Low".format(
			ticker_symbol)]) / df["{} - Adj. Close".format(ticker_symbol)] * 100.0
		return df

	@staticmethod
	def get_open_to_close_pcnt_change(df, ticker_symbol):
		"""
		Get the percentage change between open and close prices of a stock.
		:param df: dataframe
		:param ticker_symbol: str
		:return: dataframe
		"""
		df["{} - PCT_change".format(ticker_symbol)] = (df["{} - Adj. Close".format(
			ticker_symbol)] - df["{} - Adj. Open".format(ticker_symbol)]) / df["{} - Adj. Open".format(
			ticker_symbol)] * 100.0
		return df

	def preprocess_data(self, df, ticker_symbol_list):
		"""
		Preprocess stock data
		:param df: dataframe, original dataframe
		:param ticker_symbol_list: list, list of ticker symbols
		:return preprocessed_data_dict: dict, dictionary with ticker symbols as keys, preprocessed stock data dataframes as
		values
		:return original_df_dict: dict, dictionary with ticker symbols as keys, original stock data dataframes as values
		"""
		self.ticker_symbol_list = ticker_symbol_list
		logger.info("----------------Pre-processing data----------------")
		# Extract data frames for each ticker from the original data frame and put it in a dictionary.
		self.get_df_for_each_ticker(df)
		useful_features = ["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]
		for ticker_symbol, original_df in self.original_df_dict.items():
			ticker_domain = ticker_symbol.split("/")[0]
			feature_list = self.get_feature_list(ticker_domain)
			logger.debug("Feature list for {} = {}".format(ticker_symbol, feature_list))
			preprocessed_feature_list = list(map(
				lambda x, x1: "{} - {}".format(x, x1), [ticker_symbol] * len(feature_list), feature_list))
			preprocessed_df = original_df[preprocessed_feature_list].copy(deep=True)
			if ticker_domain in ["WIKI"]:
				# Compute high to low and open to close stock price percentage values and add them to feature list
				preprocessed_df = self.get_high_to_low_pcnt_change(preprocessed_df, ticker_symbol)
				preprocessed_df = self.get_open_to_close_pcnt_change(preprocessed_df, ticker_symbol)
				preprocessed_feature_list = list(map(
					lambda x, x1: "{} - {}".format(x, x1), [ticker_symbol] * len(useful_features), useful_features))
				preprocessed_df = preprocessed_df[preprocessed_feature_list]
			# Forecast column labels depending on the domain
			forecast_col_labels = {
				"WIKI": "{} - Adj. Close".format(ticker_symbol),
				"BCB": "{} - Value".format(ticker_symbol),
				"NASDAQOMX": "{} - Index Value".format(ticker_symbol)
			}
			preprocessed_df.dropna(inplace=True)
			preprocessed_df["label"] = preprocessed_df[forecast_col_labels[ticker_domain]]
			X_forecast = np.array(preprocessed_df.drop(["label"], 1))
			# Number of future data points to be predicted.
			forecast_out = int(math.ceil(self.future_prediction_pcnt * 0.01 * len(preprocessed_df)))
			preprocessed_df = preprocessed_df.iloc[0: int((1 - self.future_prediction_pcnt * 0.01) * len(preprocessed_df)), :]
			preprocessed_df["label"] = preprocessed_df["label"].shift(-forecast_out)
			preprocessed_df.dropna(inplace=True)
			X = np.array(preprocessed_df.drop(["label"], 1))
			X = X[:-forecast_out]
			y = np.array(preprocessed_df["label"])
			y = y[:-forecast_out]
			self.preprocessed_data_dict[ticker_symbol] = [X, X_forecast, y]
		return self.preprocessed_data_dict, self.original_df_dict
