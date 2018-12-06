import datetime
import numpy as np
from setuplogger import logger


class TestModels(object):

	def __init__(self, **kwargs):
		pass

	def test_model(self, model_name, model_for_each_stock_list, preprocessed_data_list, df_list):
		logger.info("Testing the {} model".format(model_name))
		forecast_df_list = []
		for model, preprocessed_data, df in zip(model_for_each_stock_list, preprocessed_data_list, df_list):
			df_copy = df.copy(deep=True)
			ticker_symbol = df_copy.columns[0].split(" ")[0]
			X_forecast = preprocessed_data[1]
			logger.debug("len(X_forecast) = {}".format(len(X_forecast)))
			forecast_set = model.predict(X_forecast)
			df_copy["{} - Forecast".format(ticker_symbol)] = np.nan
			logger.debug("type(df.iloc[-1].name", type(df_copy.iloc[-1].name))
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
			forecast_df_list.append(df_copy)
		return forecast_df_list

	def test_models(self, models_dict, preprocessed_data_list, df_list):
		test_result_dict = {}
		for model_name, model_for_each_stock_list in models_dict.items():
			test_result_dict[model_name] = self.test_model(
				model_name, model_for_each_stock_list, preprocessed_data_list, df_list)
		return test_result_dict
