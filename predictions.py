from setuplogger import logger


class Predictions(object):

	"""
	This class uses the built models to predict future stock prices.
	Attributes:
		future_predictions_dict (dict): dictionary containing model_names as keys and the dataframes with future stock
		price predictions as values.
	"""

	def __init__(self):
		"""Constructor for the class"""
		self.future_predictions_dict = {}

	def make_prediction(self, model_name, model_for_each_ticker_dict, preprocessed_data_dict, original_df_dict):
		"""
		Make future stock price prediction.
		:param model_name: str, name of the model.
		:param model_for_each_ticker_dict: dict.
		:param preprocessed_data_dict: dict.
		:param original_df_dict: dict.
		:return:
		"""
		logger.info("----------------Predicting future prices using the {} model----------------".format(model_name))
		forecast_df_dict = {}
		for ticker_symbol, model in model_for_each_ticker_dict.items():
			ticker_symbol = ticker_symbol.replace("_", "/")
			logger.info("Predicting future prices for {}".format(ticker_symbol))
			df_copy = original_df_dict[ticker_symbol].copy(deep=True)
			df_copy.dropna(inplace=True)
			X_forecast = preprocessed_data_dict[ticker_symbol][1]
			logger.debug("len(X_forecast) = {}".format(len(X_forecast)))
			forecast_set = model.predict(X_forecast)
			df_copy["{} - Forecast".format(ticker_symbol)] = forecast_set
			forecast_df_dict[ticker_symbol] = df_copy
		return forecast_df_dict

	def make_predictions(self, models_dict, preprocessed_data_dict, original_df_dict):
		"""
		Make future stock price predictions for all the stocks using all the models.
		:param models_dict:
		:param preprocessed_data_dict:
		:param original_df_dict:
		:return future_predictions_dict: dict. dictionary containing model name as key and dataframes containing predicted
		future stock prices as values.
		"""
		for model_name, model_for_each_ticker_dict in models_dict.items():
			self.future_predictions_dict[model_name] = self.make_prediction(
				model_name, model_for_each_ticker_dict, preprocessed_data_dict, original_df_dict)
		return self.future_predictions_dict
