from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, train_test_split, TimeSeriesSplit
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from setuplogger import logger


class BuildModels(object):

	def __init__(self):
		self.built_models_dict = {}
		self.model_scores_dict = {}
		# self.models_dict = {
		# 	"Decision Tree Regressor": DecisionTreeRegressor(random_state=0, max_depth=5),
		# 	"Linear Regression": LinearRegression(n_jobs=-1),
		# 	"Random Forest Regressor": RandomForestRegressor(max_depth=5, random_state=0, n_estimators=10),
		# 	# "SVR": SVR(kernel="linear", gamma="scale")
		# 	"SVR": SVR(kernel="linear")
		# }
		self.models_dict = {
			"Decision Tree Regressor": DecisionTreeRegressor(),
			"Linear Regression": LinearRegression(),
			"Random Forest Regressor": RandomForestRegressor(),
			# "SVR": SVR(kernel="linear", gamma="scale")
			"SVR": SVR()
		}
		self.parameters_dict = {
			"Decision Tree Regressor": {"max_depth": [1, 2, 5, 10, 20, 50]},
			"Linear Regression": {"n_jobs": [None, -1]},
			"Random Forest Regressor": {"max_depth": [2, 5, 10, 20], "n_estimators": [10, 50, 100]},
			"SVR": {"kernel": ["rbf", "linear"], "degree": [1, 2, 3, 4], "gamma": ["auto_deprecated", "scale"]}
		}

	def build_model(self, model_name, preprocessed_data_dict):
		logger.info("----------------Building model using {}----------------".format(model_name))
		model_dict = {}
		model_scores_dict = {}
		for ticker_symbol, preprocessed_data in preprocessed_data_dict.items():
			[X, X_forecast, y] = preprocessed_data
			# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
			tscv = TimeSeriesSplit(n_splits=5)
			optimized_model, cv_scores = self.optimize_hyperparameters(X, y, self.parameters_dict[model_name], model_name, tscv)
			model = make_pipeline(StandardScaler(), optimized_model)
			X_train, X_test, y_train, y_test = self.get_train_and_test_data(X, y, cv_scores)
			model.fit(X_train, y_train)
			confidence_score = model.score(X_test, y_test)
			logger.info("Training score for {} = {}".format(ticker_symbol, confidence_score))
			logger.debug("Cross validation scores for {} = {}".format(ticker_symbol, cv_scores["test_score"]))
			logger.info("Cross validation score for {} = {} +/- {}".format(
				ticker_symbol, cv_scores["test_score"].mean(), cv_scores["test_score"].std() * 2))
			logger.debug("Cross validation scoring time = {}s".format(cv_scores["score_time"].sum()))
			model_dict[ticker_symbol] = model
			model_scores_dict[ticker_symbol] = confidence_score
		return model_dict, model_scores_dict

	def build_models(self, model_names, preprocessed_data_dict):
		for model_name in model_names:
			model_dict, model_scores_dict = self.build_model(model_name, preprocessed_data_dict)
			self.built_models_dict[model_name] = model_dict
			self.model_scores_dict[model_name] = model_scores_dict
		return self.built_models_dict, self.model_scores_dict

	def get_built_models(self):
		if self.built_models_dict:
			return self.built_models_dict
		else:
			logger.info("No models found. Run build_models first and then call this method.")
			exit(1)

	@staticmethod
	def get_best_split_index(scores):
		best_split_index_tuple = np.where(scores["test_score"] == max(scores["test_score"]))
		best_split_index = int(best_split_index_tuple[0])
		return best_split_index

	def get_train_and_test_data(self, X, y, scores):
		tscv = TimeSeriesSplit(n_splits=5)
		X_train, X_test, y_train, y_test = [None, None, None, None]
		split_data = []
		for train_indices, test_indices in tscv.split(X):
			# print("train_data size = {}".format(len(train_index) * 100 /len(self.X)))
			X_train, X_test = X[train_indices], X[test_indices]
			y_train, y_test = y[train_indices], y[test_indices]
			split_data.append((X_train, X_test, y_train, y_test))
		# Get the index of the score that is maximum from the cross val scores list
		best_split_index = self.get_best_split_index(scores)
		X_train, X_test, y_train, y_test = split_data[best_split_index]
		logger.debug("Optimized train_data size = {}".format(len(X_train) * 100 / len(X)))
		# print("X_train = {}".format(X_train))
		return X_train, X_test, y_train, y_test

	def optimize_hyperparameters(self, X, y, parameters_dict, model_name, cv_iterator):
		# Parameter grid
		# p_grid = {
		# 	"alpha": [0.1, 0.5, 1, 1.5]
		# }
		logger.debug("Optimizing hyper-parameters")
		model = self.models_dict[model_name]
		# Hyperparameter optimization
		optimized_model = GridSearchCV(estimator=model, param_grid=parameters_dict, cv=cv_iterator)
		cv_score = cross_validate(optimized_model, X=X, y=y, cv=cv_iterator)
		return optimized_model, cv_score
