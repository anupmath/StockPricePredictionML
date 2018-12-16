import datetime
import numpy as np
from setuplogger import logger


class Predictions(object):

	def __init__(self, **kwargs):
		self.test_result_dict = {}

	@staticmethod
	def get_forecast_set_df(df_copy, forecast_set, ticker_symbol):
		df_copy["{} - Forecast".format(ticker_symbol)] = np.nan
		logger.debug("type(df.iloc[-1].name = {}".format(type(df_copy.iloc[-1].name)))
		last_date = datetime.datetime.strptime(df_copy.iloc[-1].name, "%Y-%m-%d") if isinstance(df_copy.iloc[-1].name, str) \
			else df_copy.iloc[-1].name
		logger.debug("last_date = {}".format(last_date))
		last_unix = last_date.timestamp()
		one_day = 86400
		next_unix = last_unix + one_day
		for i in forecast_set:
			next_date = datetime.datetime.fromtimestamp(next_unix)
			next_unix += 86400
			df_copy.loc[next_date] = [np.nan for _ in range(len(df_copy.columns) - 1)] + [i]
		return df_copy

	def make_prediction(self, model_name, model_for_each_ticker_dict, preprocessed_data_dict, original_df_dict):
		logger.info("----------------Predicting future prices using the {} model----------------".format(model_name))
		forecast_df_dict = {}
		for ticker_symbol, model in model_for_each_ticker_dict.items():
			logger.info("Predicting future prices for {}".format(ticker_symbol))
			df_copy = original_df_dict[ticker_symbol].copy(deep=True)
			df_copy.dropna(inplace=True)
			X_forecast = preprocessed_data_dict[ticker_symbol][1]
			logger.debug("len(X_forecast) = {}".format(len(X_forecast)))
			forecast_set = model.predict(X_forecast)
			forecast_df_dict[ticker_symbol] = self.get_forecast_set_df(df_copy, forecast_set, ticker_symbol)
		return forecast_df_dict

	def make_predictions(self, models_dict, preprocessed_data_dict, original_df_dict):
		for model_name, model_for_each_ticker_dict in models_dict.items():
			self.test_result_dict[model_name] = self.make_prediction(
				model_name, model_for_each_ticker_dict, preprocessed_data_dict, original_df_dict)
		return self.test_result_dict
