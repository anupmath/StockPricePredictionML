import math
from matplotlib import style
import matplotlib.pyplot as plt
import os
import sys
from buildmodels import BuildModels
from getdata import GetData
from preprocessing import PreprocessData
from setuplogger import logger
from predictions import Predictions


class StockPricePrediction(object):

  """
  This class pull stock data from quandl and makes future predictions.
  """

  def __init__(self):
    """Constructor for the class"""
    self.current_dir = os.getcwd()
    self.stock_price_plots_dir = "{}/stock_price_plots/".format(os.getcwd())
    os.mkdir("{}/stock_price_plots".format(self.current_dir)) if not os.path.exists("{}/stock_price_plots".format(
      self.current_dir)) else None

  def plot_forecast(self, forecast_df_dict, original_df_dict, future_prediction_pcnt=1):
    """
    Plots the actual data and the forecast data in the dataframe.
    :param forecast_df_dict: dict, dictionary containing model names as keys and dictionaries containing
     ticker symbols as keys and preprocessed dataframes containing forecast data as values.
    :param original_df_dict: dict, dictionary containing ticker symbols as keys and original dataframes as values.
    :param future_prediction_pcnt: float, Number of dates/data points into the future for which the stock price is to
    be predicted as a percentage of the number of dates/data points for which historical data which is already
    available
    :return: None
    """
    for model_name, df_dict in forecast_df_dict.items():
      logger.info("----------------Plotting stock prices for {} model----------------".format(model_name))
      for ticker_symbol, df in df_dict.items():
        ticker_domain = ticker_symbol.split("/")[0]
        original_df = original_df_dict[ticker_symbol].dropna().reset_index()
        df = df.reset_index()
        forecast_col_labels = {
          "WIKI": "{} - Adj. Close".format(ticker_symbol),
          "BCB": "{} - Value".format(ticker_symbol),
          "NASDAQOMX": "{} - Index Value".format(ticker_symbol)
        }
        logger.info("----------------Plotting stock prices for {}".format(ticker_symbol))
        # Number of future data points to be predicted.
        forecast_out = int(math.ceil(future_prediction_pcnt * 0.01 * len(df)))
        original_df["Date"] = original_df["Date"].shift(-forecast_out)
        df["{} - Forecast".format(ticker_symbol)].plot(color='b')
        original_df[forecast_col_labels[ticker_domain]].plot(color='g')
        plt.legend(loc="best")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Forecast for {} model for {}".format(model_name, ticker_symbol))
        # fig = plt.figure()
        plt.savefig("{}/{}_{}.png".format(self.stock_price_plots_dir, model_name, ticker_symbol.replace("/", "_")))
        plt.clf()
        plt.close()

  def main(self):
    """
    Pulls stock data for the ticker symbols from a json file and pull data from quandl, preprocesses the data
    and then build different supervised learning machine learning models and predicts future stock price.
    :return: None
    """
    logger.info("------------------Started Stock Price Prediction-----------------")
    # Create instances of all the classes used for stock prediction
    get_data = GetData(api_key=sys.argv[1])
    # Number of dates/data points into the future for which the stock price is to be predicted as a percentage of the
    # number of dates/data points for which historical data which is already available
    future_prediction_pcnt = 1
    preprocess_data = PreprocessData(future_prediction_pcnt=future_prediction_pcnt)
    build_models = BuildModels()
    forecast_prices = Predictions()
    # Get data from quandl.
    df = get_data.get_stock_data(update_data=False)
    # Preprocess data
    preprocessed_data_dict, original_df_dict = preprocess_data.preprocess_data(df, get_data.stock_ticker_list)
    models_list = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    # Build models
    models_dict, model_scores_dict = build_models.build_models(models_list, preprocessed_data_dict, force_build=False)
    # Predict future stock prices
    forecast_df_dict = forecast_prices.make_predictions(models_dict, preprocessed_data_dict, original_df_dict)
    self.plot_forecast(forecast_df_dict, original_df_dict, future_prediction_pcnt)


if __name__ == "__main__":
  stock_price_prediction = StockPricePrediction()
  stock_price_prediction.main()


