from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from setuplogger import logger


class BuildModels(object):

	def __init__(self, **kwargs):
		self.models_dict = {}
		self.model_scores_dict = {}

	def build_model(self, model_name, preprocessed_data_dict):
		logger.info("Building model using {}.".format(model_name))
		model_dict = {}
		model_scores_dict = {}
		for ticker_symbol, preprocessed_data in preprocessed_data_dict.items():
			[X, X_forecast, y] = preprocessed_data
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
			if model_name == "Linear Regression":
				model = LinearRegression(n_jobs=-1)
				model.fit(X_train, y_train)
			elif model_name == "SVR":
				model = SVR(kernel="linear", gamma="scale")
				model.fit(X_train, y_train)
			elif model_name == "Decision Tree":
				model = DecisionTreeRegressor(random_state=0)
				model.fit(X_train, y_train)
			else:
				raise Exception("Model = {} is not supported".format(model_name))
			confidence_score = model.score(X_test, y_test)
			logger.info("-------Confidence score for {} model for {} = {}".format(model_name, ticker_symbol, confidence_score))
			model_dict[ticker_symbol] = model
			model_scores_dict[ticker_symbol] = confidence_score
		return model_dict

	def build_models(self, model_names, preprocessed_data_dict):
		for model_name in model_names:
			model_dict, model_scores_dict = self.build_model(model_name, preprocessed_data_dict)
			self.models_dict[model_name] = model_dict
			self.model_scores_dict[model_name] = model_scores_dict
		return self.models_dict, self.model_scores_dict

	def get_built_models(self):
		if self.models_dict:
			return self.models_dict
		else:
			logger.info("No models found. Run build_models first and then call this method.")
			exit(1)
