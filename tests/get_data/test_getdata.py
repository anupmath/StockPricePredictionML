import json
from src.getdata import GetData


class TestGetData(object):

	def test_get_stock_data_info(self):
		with open("tests/test_stockdatainfo.json", 'r') as ticker_list_f:
			ticker_list_str = ticker_list_f.read()
			stock_data_info = json.loads(ticker_list_str)
		assert stock_data_info == GetData.get_stock_data_info("tests/test_stockdatainfo.json")

	def test_get_stock_ticker_list(self):
		test_stock_ticker_list = ["WIKI/ABCD", "WIKI/EFGH", "WIKI/IJKL", "BCB/1234", "BCB/AAAA", "BCB/BBBB",
															"NASDAQOMX/BBBB", "NASDAQOMX/CCCC"]
		get_data = GetData(api_key="FdZ_rxZEDApZ6esukiax", stock_data_path="stockdata/stockdata.csv",
											 stock_data_info_path="tests/test_stockdatainfo.json")
		assert test_stock_ticker_list == get_data.get_stock_ticker_list()
