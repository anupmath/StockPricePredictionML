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

  def __init__(self):
    pass

  def plot_forecast(self, forecast_df_dict):
    current_dir = os.getcwd()
    os.mkdir("{}/stock_price_plots".format(current_dir)) if not os.path.exists("{}/stock_price_plots".format(
      current_dir)) else None
    for model_name, df_dict in forecast_df_dict.items():
      logger.info("----------------Plotting stock prices for {} model----------------".format(model_name))
      for ticker_symbol, df in df_dict.items():
        forecast_col_labels = {
          "WIKI": "{} - Adj. Close".format(ticker_symbol),
          "BCB": "{} - Value".format(ticker_symbol),
          "NASDAQOMX": "{} - Index Value".format(ticker_symbol)
        }
        logger.info("----------------Plotting stock prices for {}".format(ticker_symbol))
        style.use("ggplot")
        df.plot(x="Date", y=[forecast_col_labels[ticker_symbol.split("/")[0]], "{} - Forecast".format(ticker_symbol)],
                color=['g', 'b'])
        plt.legend(loc="best")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Forecast for {} model for {}".format(model_name, ticker_symbol))
        # fig = plt.figure()
        plt.savefig("{}/stock_price_plots/{}_{}.png".format(current_dir, model_name, ticker_symbol.replace("/", "_")))
        plt.clf()
        plt.close()

  def main(self):
    logger.info("------------------Started Stock Price Prediction-----------------")
    get_data = GetData(api_key=sys.argv[1])
    preprocess_data = PreprocessData()
    build_models = BuildModels()
    forecast_prices = Predictions()
    df = get_data.get_stock_data(update_data=False)
    preprocessed_data_dict, original_df_dict = preprocess_data.preprocess_data(df, get_data.stock_ticker_list)
    models_list = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    # models_list = ["Linear Regression"]
    models_dict, model_scores_dict = build_models.build_models(models_list, preprocessed_data_dict)
    forecast_df_dict = forecast_prices.make_predictions(models_dict, preprocessed_data_dict, original_df_dict)
    self.plot_forecast(forecast_df_dict)


if __name__ == "__main__":
  stock_price_prediction = StockPricePrediction()
  stock_price_prediction.main()


