import math
import numpy as np
import pandas as pd
from getdata import GetData
from setuplogger import logger


class PreprocessData(object):

	def __init__(self):
		self.stock_data_info = GetData.get_stock_data_info()
		self.feature_list = self.stock_data_info["features"]
		self.original_df_dict = {}
		self.preprocessed_data_dict = {}
		self.ticker_symbol_list = []

	def get_df_for_each_ticker(self, df):
		df_date = df["Date"].copy(deep=True)
		df_date = pd.DataFrame(df_date.values, columns=["Date"])
		#Drop date from original dataframe for easier separation of data for all stocks
		df = df.drop(["Date"], 1)
		df_col_nos = list(range(0, df.shape[1], 12))
		for ticker_symbol, df_col_no in zip(self.ticker_symbol_list, df_col_nos):
			df_with_date = pd.concat([df_date.iloc[:, 0], df.iloc[:, df_col_no:df_col_no + 12]], axis=1)
			self.original_df_dict[ticker_symbol] = df_with_date

	@staticmethod
	def get_high_to_low_pcnt_change(df, ticker_symbol):
		df["{} - HL_PCT".format(ticker_symbol)] = (df["{} - Adj. High".format(ticker_symbol)] - df["{} - Adj. Low".format(
			ticker_symbol)]) / df["{} - Adj. Close".format(ticker_symbol)] * 100.0
		return df

	@staticmethod
	def get_open_to_close_pcnt_change(df, ticker_symbol):
		df["{} - PCT_change".format(ticker_symbol)] = (df["{} - Adj. Close".format(
			ticker_symbol)] - df["{} - Adj. Open".format(ticker_symbol)]) / df["{} - Adj. Open".format(
			ticker_symbol)] * 100.0
		return df

	def preprocess_data(self, df, ticker_symbol_list):
		self.ticker_symbol_list = ticker_symbol_list
		logger.info("----------------Pre-processing data----------------")
		# Extract data frames for each ticker from the original data frame and put it in a dictionary.
		self.get_df_for_each_ticker(df)
		# df_list = [df.iloc[:, i:i + 12] for i in range(0, df.shape[1], 12)]
		useful_features = ["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]
		logger.debug("Feature list = {}".format(self.feature_list))
		for ticker_symbol, original_df in self.original_df_dict.items():
			preprocessed_feature_list = list(map(
				lambda x, y: "{} - {}".format(x, y), [ticker_symbol] * len(self.feature_list), self.feature_list))
			original_df = original_df[preprocessed_feature_list]
			original_df = self.get_high_to_low_pcnt_change(original_df, ticker_symbol)
			original_df = self.get_open_to_close_pcnt_change(original_df, ticker_symbol)
			preprocessed_feature_list = list(map(
				lambda x, y: "{} - {}".format(x, y), [ticker_symbol] * len(useful_features), useful_features))
			original_df = original_df[preprocessed_feature_list]
			forecast_col = "{} - Adj. Close".format(ticker_symbol)
			# df.fillna(value=-99999, inplace=True)
			original_df.dropna(inplace=True)
			forecast_out = int(math.ceil(0.01 * len(original_df)))
			original_df["label"] = original_df[forecast_col].shift(-forecast_out)
			X = np.array(original_df.drop(["label"], 1))
			X_forecast = X[-forecast_out:]
			X = X[:-forecast_out]
			y = np.array(original_df["label"])
			y = y[:-forecast_out]
			self.preprocessed_data_dict[ticker_symbol] = [X, X_forecast, y]
		return self.preprocessed_data_dict, self.original_df_dict
