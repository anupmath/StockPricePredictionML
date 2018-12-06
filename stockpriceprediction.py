from matplotlib import style
import matplotlib.pyplot as plt
import sys
from buildmodels import BuildModels
from getdata import GetData
from preprocessing import PreprocessData
from setuplogger import logger
from testmodels import TestModels


class StockPricePrediction(object):

  def __init__(self):
    pass

  def plot_forecast(self, forecast_df_list):
    logger.info("Plotting forecast")
    for model_name, df_list in forecast_df_list.items():
      for df in df_list:
        ticker_symbol = df.columns[0].split(" ")[0]
        style.use("ggplot")
        df["{} - Adj. Close".format(ticker_symbol)].plot()
        df["{} - Forecast".format(ticker_symbol)].plot()
        # plt.legend(loc=4)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Forecast for {}".format(model_name))
        plt.show()

  def main(self):
    logger.info("------------------Started Stock Price Prediction------------------")
    get_data = GetData(api_key=sys.argv[1])
    preprocess_data = PreprocessData()
    build_models = BuildModels()
    test_models = TestModels()
    df = get_data.get_stock_data(update_data=False)
    preprocessed_data_list, df_list = preprocess_data.preprocess_data(df)
    models_dict = build_models.build_models(["Linear Regression", "SVR", "Decision Tree"], preprocessed_data_list)
    forecast_df_dict = test_models.test_models(models_dict, preprocessed_data_list, df_list)
    self.plot_forecast(forecast_df_dict)


if __name__ == "__main__":
  stock_price_prediction = StockPricePrediction()
  stock_price_prediction.main()


