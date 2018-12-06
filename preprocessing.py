import math
import numpy as np
from sklearn import preprocessing, svm
from getdata import GetData
from setuplogger import logger


class PreprocessData(object):

	def __init__(self, **kwargs):
		self.stock_data_info = GetData.get_stock_data_info()

	def preprocess_data(self, df):
		logger.info("Preprocessing data")
		processed_data_list = []
		df_list = [df.iloc[:, i:i + 12] for i in range(0, df.shape[1], 12)]
		feature_list = self.stock_data_info["features"]
		logger.debug("Feature list = {}".format(feature_list))
		for df in df_list:
			ticker_symbol = df.columns[0].split(" ")[0]
			processed_feature_list = list(map(
				lambda x, y: "{} - {}".format(x, y), [ticker_symbol] * len(feature_list), feature_list))
			df = df[processed_feature_list]
			df["{} - HL_PCT".format(ticker_symbol)] = (df["{} - Adj. High".format(ticker_symbol)] - df["{} - Adj. Low".format(
				ticker_symbol)]) / df["{} - Adj. Close".format(ticker_symbol)] * 100.0
			df["{} - PCT_change".format(ticker_symbol)] = (df["{} - Adj. Close".format(
				ticker_symbol)] - df["{} - Adj. Open".format(
				ticker_symbol)]) / df["{} - Adj. Open".format(ticker_symbol)] * 100.0
			df = df[["{} - Adj. Close".format(ticker_symbol), "{} - HL_PCT".format(ticker_symbol),
							 "{} - PCT_change".format(ticker_symbol), "{} - Adj. Volume".format(ticker_symbol)]]
			forecast_col = "{} - Adj. Close".format(ticker_symbol)
			df.fillna(value=-99999, inplace=True)
			forecast_out = int(math.ceil(0.01 * len(df)))
			df["label"] = df[forecast_col].shift(-forecast_out)
			X = np.array(df.drop(["label"], 1))
			X = preprocessing.scale(X)
			X_forecast = X[-forecast_out:]
			X = X[:-forecast_out]
			df.dropna(inplace=True)
			y = np.array(df["label"])
			processed_data_list.append([X, X_forecast, y])
		return processed_data_list, df_list
