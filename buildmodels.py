from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from setuplogger import logger


class BuildModels(object):

	def __init__(self, **kwargs):
		pass

	def build_model(self, model_name, preprocessed_data_list):
		logger.info("Building model using {}.".format(model_name))
		model_list = []
		for preprocessed_data in preprocessed_data_list:
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
			confidence = model.score(X_test, y_test)
			logger.info("-------Confidence score for {} model = {}".format(model_name, confidence))
			model_list.append(model)
		return model_list

	def build_models(self, model_names, preprocessed_data_list):
		models_dict = {}
		for model_name in model_names:
			models_dict[model_name] = self.build_model(model_name, preprocessed_data_list)
		return models_dict
