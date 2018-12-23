import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, TimeSeriesSplit
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
			"Decision Tree Regressor": {"max_depth": [200]},
			"Linear Regression": {"n_jobs": [None, -1]},
			"Random Forest Regressor": {"max_depth": [200], "n_estimators": [100]},
			"SVR": {"kernel": ["rbf", "linear"], "degree": [3], "gamma": ["scale"]}
		}
		self.saved_models_path = "{}/{}".format(os.getcwd(), self.saved_models_dir)
		self.learning_curve_dir_path = "{}/learning_curve_plots".format(os.getcwd())
		os.mkdir(self.saved_models_path) if not os.path.exists(self.saved_models_path) else None
		os.mkdir(self.learning_curve_dir_path) if not os.path.exists(self.learning_curve_dir_path) else None

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
		curr_dir = os.getcwd()
		for ticker_symbol, preprocessed_data in preprocessed_data_dict.items():
			[X, X_forecast, y] = preprocessed_data
			tscv = TimeSeriesSplit(n_splits=5)
			ticker_symbol = ticker_symbol.replace("/", "_")
			if force_build or not os.path.exists(
					"{}/{}_{}_model.pickle".format(self.saved_models_path, model_name,	ticker_symbol)):
				# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
				# Create a cv iterator for splitting train and test data using TimeSeriesSplit
				# Optimize the hyperparameters based on the cross validation scores
				optimized_model = self.optimize_hyperparameters(model_name, tscv)
				model = make_pipeline(StandardScaler(), optimized_model)
				X_train, X_test, y_train, y_test = self.get_train_and_test_data(X, y, tscv)
				model.fit(X_train, y_train)
				self.save_to_pickle_file(model_name, ticker_symbol, model, "model")
			else:
				model = self.load_from_pickle_file(model_name, ticker_symbol, "model")
				X_train, X_test, y_train, y_test = self.get_train_and_test_data(X, y, tscv)
			# Training score
			confidence_score = model.score(X_test, y_test)
			# Plot learning curves
			title = "{}_{}_Learning Curves".format(model_name, ticker_symbol)
			save_file_path = "{}/learning_curve_plots/{}_{}.png".format(curr_dir, model_name, ticker_symbol)
			# Create the CV iterator
			self.plot_learning_curve(model, title, X, y, save_file_path, cv=tscv)
			# Cross validation
			cv_scores = cross_validate(model, X=X, y=y, cv=tscv)
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

	def get_train_and_test_data(self, X, y, tscv):
		"""
		Get the train and test data sets.
		:param X: ndarray.
		:param y: array
		:param tscv: iterator, TimeSeriesSplit iterator
		:return X_train, X_test, y_train, y_test: arrays, train and test data.
		"""
		split_data = []
		for train_indices, test_indices in tscv.split(X):
			X_train, X_test = X[train_indices], X[test_indices]
			y_train, y_test = y[train_indices], y[test_indices]
			split_data.append((X_train, X_test, y_train, y_test))
		# Get cross validation score for the last index as it will have the most training data which is good for time
		# series data
		best_split_index = -1
		X_train, X_test, y_train, y_test = split_data[best_split_index]
		logger.debug("Last train_data size = {}".format(len(X_train) * 100 / len(X)))
		return X_train, X_test, y_train, y_test

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
			self.saved_models_dir, model_name, ticker_symbol, obj_name), "rb")
		loaded_obj = pickle.load(pickle_in)
		return loaded_obj

	def optimize_hyperparameters(self, model_name, cv_iterator):
		"""
		Optimize hyperparameters based on the cross validation score.
		:param model_name: str, name of the model
		:param cv_iterator: iterator, split train and test data.
		:return:
		"""
		logger.debug("Optimizing hyper-parameters")
		parameters_dict = self.parameters_dict[model_name]
		model = self.models_dict[model_name]
		# Hyperparameter optimization
		optimized_model = GridSearchCV(estimator=model, param_grid=parameters_dict, cv=cv_iterator)
		return optimized_model

	def plot_learning_curve(self, estimator, title, X, y, save_file_path, ylim=None, cv=None,
													train_sizes=np.linspace(.1, 1.0, 5)):
		"""
		Generate a simple plot of the test and training learning curve.

		Parameters
		----------
		estimator : object type that implements the "fit" and "predict" methods
				An object of that type which is cloned for each validation.

		title : string
				Title for the chart.

		X : array-like, shape (n_samples, n_features)
				Training vector, where n_samples is the number of samples and
				n_features is the number of features.

		y : array-like, shape (n_samples) or (n_samples, n_features), optional
				Target relative to X for classification or regression;
				None for unsupervised learning.

		ylim : tuple, shape (ymin, ymax), optional
				Defines minimum and maximum yvalues plotted.

		cv : int, cross-validation generator or an iterable, optional
				Determines the cross-validation splitting strategy.
				Possible inputs for cv are:
					- None, to use the default 3-fold cross-validation,
					- integer, to specify the number of folds.
					- An object to be used as a cross-validation generator.
					- An iterable yielding train/test splits.

				For integer/None inputs, if ``y`` is binary or multiclass,
				:class:`StratifiedKFold` used. If the estimator is not a classifier
				or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

				Refer :ref:`User Guide <cross_validation>` for the various
				cross-validators that can be used here.
		train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
		"""
		logger.info("Plotting {}".format(title))
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(
			estimator, X, y, cv=cv, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
										 color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
										 color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

		plt.legend(loc="best")
		plt.savefig("{}".format(save_file_path))
		plt.close()

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
			self.saved_models_dir, model_name, ticker_symbol, obj_name), "wb")
		pickle.dump(obj_to_be_saved, pickle_out)
		pickle_out.close()
