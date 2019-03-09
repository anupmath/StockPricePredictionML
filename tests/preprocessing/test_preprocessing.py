import json
from src.preprocessing import PreprocessData


class TestPreprocessing(object):

	def test_get_feature_list(self):
		expected_feature_list_dict = {"WIKI": ["Open", "High", "Low", "Close", "Volume"],
																	"BCB": ["Index Value"],
																	"NASDAQOMX": ["Index Value"]}
		preprocess_data = PreprocessData(future_prediction_pcnt=1,
																		 stock_data_info_path="tests/test_stockdatainfo.json")
		assert all([features == preprocess_data.get_feature_list(ticker_domain)
								for ticker_domain, features in expected_feature_list_dict.items()])
