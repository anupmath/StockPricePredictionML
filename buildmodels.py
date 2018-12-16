from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from setuplogger import logger


class BuildModels(object):

	def __init__(self, **kwargs):
		self.models_dict = {}
		self.model_scores_dict = {}

	def build_model(self, model_name, preprocessed_data_dict):
		logger.info("----------------Building model using {}----------------".format(model_name))
		model_dict = {}
		model_scores_dict = {}
		models_dict = {
			"Decision Tree": DecisionTreeRegressor(random_state=0),
			"Linear Regression": LinearRegression(n_jobs=-1),
			"Random Forest Regressor": RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100),
			"SVR": SVR(kernel="linear", gamma="scale")
		}
		model = make_pipeline(StandardScaler(), models_dict[model_name])
		# model = make_pipeline(models_dict[model_name])
		for ticker_symbol, preprocessed_data in preprocessed_data_dict.items():
			[X, X_forecast, y] = preprocessed_data
			# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
			X_train, X_test, y_train, y_test = self.get_train_and_test_data(X, y)
			model.fit(X_train, y_train)
			confidence_score = model.score(X_test, y_test)
			tscv = TimeSeriesSplit(n_splits=5)
			scores = cross_validate(model, X, y, cv=tscv)
			logger.info("Confidence score for {} = {}".format(ticker_symbol, confidence_score))
			logger.info("Cross validation scores for {} = {} +/- {}".format(
				ticker_symbol, scores["test_score"].mean(), scores["test_score"].std() * 2))
			logger.debug("Cross validation scoring time = {}s".format(scores["score_time"].sum()))
			model_dict[ticker_symbol] = model
			model_scores_dict[ticker_symbol] = confidence_score
		return model_dict, model_scores_dict

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

	@staticmethod
	def get_train_and_test_data(X, y):
		tscv = TimeSeriesSplit(n_splits=5)
		X_train, X_test, y_train, y_test = [None, None, None, None]
		for train_indices, test_indices in tscv.split(X):
			# print("train_data size = {}".format(len(train_index) * 100 /len(self.X)))
			X_train, X_test = X[train_indices], X[test_indices]
			y_train, y_test = y[train_indices], y[test_indices]
		logger.debug("train_data size = {}".format(len(X_train) * 100 / len(X)))
		# print("X_train = {}".format(X_train))
		return X_train, X_test, y_train, y_test
