import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from setuplogger import logger


class BuildModels(object):

	"""
	This class builds different machine learning models for the given stock data.
	Attributes:
		built_models_dict (dict): dictionary containing model names as keys and built model objects as values.
		model_scores_dict (dict): dictionary containing model names as keys and scores for those models as values.
		saved_models_dir (str): saved models directory name
		models_dict (dict): dictionary containing model names as keys and sklearn model objects as values.
		parameters_dict (dict): dictionary containing model names as keys and hyperparameters dictionaries as values.
	"""

	def __init__(self):
		"""
		Constructor for the class.
		"""
		self.built_models_dict = {}
		self.model_scores_dict = {}
		self.saved_models_dir = "saved_models"
		self.models_dict = {
			"Decision Tree Regressor": DecisionTreeRegressor(),
			"Linear Regression": LinearRegression(),
			"Random Forest Regressor": RandomForestRegressor(),
			"SVR": SVR()
		}
		self.parameters_dict = {
			"Decision Tree Regressor": {"max_depth": [1, 2, 5]},
			"Linear Regression": {"n_jobs": [None, -1]},
			"Random Forest Regressor": {"max_depth": [2, 5], "n_estimators": [10, 100, 200]},
			"SVR": {"kernel": ["rbf", "linear"], "degree": [3], "gamma": ["scale"]}
		}

	def build_model(self, model_name, preprocessed_data_dict, force_build):
		"""
		Build machine learning models using different supervised learning regression algorithms
		:param model_name: str, name of the model to be built.
		:param preprocessed_data_dict: dict.
		:param force_build: bool, if True, will force the function to build the model, even if there is a saved model
		which was built before that is available.
		:return model_dict: dict, dictionary containing model name as key and the built model object as value.
		:return model_scores_dict: dictionary containing model name as key and the model training score as value.
		"""
		logger.info("----------------Building model using {}----------------".format(model_name))
		model_dict = {}
		model_scores_dict = {}
		for ticker_symbol, preprocessed_data in preprocessed_data_dict.items():
			[X, X_forecast, y] = preprocessed_data
			if force_build or not os.path.exists(
					"{}/{}/{}_{}_model.pickle".format(os.getcwd(), self.saved_models_dir, model_name,
																						ticker_symbol.replace("/", "_"))):
				# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
				# Create a cv iterator for splitting train and test data using TimeSeriesSplit
				tscv = TimeSeriesSplit(n_splits=5)
				# Optimize the hyperparameters based on the cross validation scores
				optimized_model, cv_scores = self.optimize_hyperparameters(X, y, self.parameters_dict[model_name], model_name,
																																	 tscv)
				model = make_pipeline(StandardScaler(), optimized_model)
				X_train, X_test, y_train, y_test = self.get_train_and_test_data(X, y)
				model.fit(X_train, y_train)
				self.save_to_pickle_file(model_name, ticker_symbol, model, "model")
				self.save_to_pickle_file(model_name, ticker_symbol, cv_scores, "cv_scores")
			else:
				model = self.load_from_pickle_file(model_name, ticker_symbol, "model")
				cv_scores = self.load_from_pickle_file(model_name, ticker_symbol, "cv_scores")
				X_train, X_test, y_train, y_test = self.get_train_and_test_data(X, y)
			# Training score
			confidence_score = model.score(X_test, y_test)
			logger.info("Training score for {} = {}".format(ticker_symbol, confidence_score))
			logger.debug("Cross validation scores for {} = {}".format(ticker_symbol, cv_scores["test_score"]))
			logger.info("Cross validation score for {} = {} +/- {}".format(
				ticker_symbol, cv_scores["test_score"].mean(), cv_scores["test_score"].std() * 2))
			logger.debug("Cross validation scoring time = {}s".format(cv_scores["score_time"].sum()))
			model_dict[ticker_symbol] = model
			model_scores_dict[ticker_symbol] = confidence_score
		return model_dict, model_scores_dict

	def build_models(self, model_names, preprocessed_data_dict, force_build=False):
		"""
		Build models for the given model names.
		:param model_names: list, list of model names
		:param preprocessed_data_dict: dict.
		:param force_build: bool, force the model to be build even if there is a saved model available
		:return built_models_dict: dictionary containing model names as keys and built model objects as values.
		:return model_scores_dict: dictionary containing model names as keys and training scores for those models as values.
		"""
		for model_name in model_names:
			model_dict, model_scores_dict = self.build_model(model_name, preprocessed_data_dict, force_build)
			self.built_models_dict[model_name] = model_dict
			self.model_scores_dict[model_name] = model_scores_dict
		return self.built_models_dict, self.model_scores_dict

	def get_built_models(self):
		"""
		Get the models if they are already build.
		:return built_models_dict: dictionary containing model names as keys and built model objects as values.
		"""
		if self.built_models_dict:
			return self.built_models_dict
		else:
			logger.info("No models found. Run build_models first and then call this method.")
			exit(1)

	def get_train_and_test_data(self, X, y):
		"""
		Get the train and test data sets.
		:param X: ndarray.
		:param y: array
		:return X_train, X_test, y_train, y_test: arrays, train and test data.
		"""
		tscv = TimeSeriesSplit(n_splits=5)
		split_data = []
		for train_indices, test_indices in tscv.split(X):
			X_train, X_test = X[train_indices], X[test_indices]
			y_train, y_test = y[train_indices], y[test_indices]
			split_data.append((X_train, X_test, y_train, y_test))
		# Get cross validation score for the last index as it will have the most training data which is good for time
		# series data
		best_split_index = -1
		X_train, X_test, y_train, y_test = split_data[best_split_index]
		logger.debug("Optimized train_data size = {}".format(len(X_train) * 100 / len(X)))
		return X_train, X_test, y_train, y_test

	def optimize_hyperparameters(self, X, y, parameters_dict, model_name, cv_iterator):
		"""
		Optimize hyperparameters based on the cross validation score.
		:param X: ndarray.
		:param y: array.
		:param parameters_dict: dict, dictionary containing model names as keys and list of hyperparameters as values.
		:param model_name: str, name of the model
		:param cv_iterator: iterator, split train and test data.
		:return:
		"""
		logger.debug("Optimizing hyper-parameters")
		model = self.models_dict[model_name]
		# Hyperparameter optimization
		optimized_model = GridSearchCV(estimator=model, param_grid=parameters_dict, cv=cv_iterator)
		cv_score = cross_validate(optimized_model, X=X, y=y, cv=cv_iterator)
		return optimized_model, cv_score

	def save_to_pickle_file(self, model_name, ticker_symbol, obj_to_be_saved, obj_name):
		"""
		Save the built model to a pickle file.
		:param model_name: str, name of the model.
		:param ticker_symbol: str, ticker symbol.
		:param obj_to_be_saved: object, model object.
		:param obj_name: str, name of the built model object.
		:return None:
		"""
		logger.info("Saving {} model for {} to pickle file".format(model_name, ticker_symbol))
		pickle_out = open("{}/{}_{}_{}.pickle".format(
			self.saved_models_dir, model_name, ticker_symbol.replace("/", "_"), obj_name), "wb")
		pickle.dump(obj_to_be_saved, pickle_out)
		pickle_out.close()

	def load_from_pickle_file(self, model_name, ticker_symbol, obj_name):
		"""
		Load the built model from a pickle file.
		:param model_name: str, name of the model.
		:param ticker_symbol: str, ticker symbol.
		:param obj_name: str, name of the built model object.
		:return loaded_obj: object, model object.
		"""
		logger.info("Loading {} model for {} from pickle file".format(model_name, ticker_symbol))
		pickle_in = open("{}/{}_{}_{}.pickle".format(
			self.saved_models_dir, model_name, ticker_symbol.replace("/", "_"), obj_name), "rb")
		loaded_obj = pickle.load(pickle_in)
		return loaded_obj
