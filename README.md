# Stock Price Predictor

Objective

Predict future stock prices using the historical stock data by applying different supervised learning machine learning algorithms.

Workflow

• Acquire historical stock data from Quandl: The plan is to acquire the stock data of some of the popular stocks in sectors
  like energy, finance, health care, pharmaceutical, technology and also the S&P500 index data from Quandl.
• Preprocess the acquired stock data: Cleaning of data based on model requirement.
• Extract relevant features: Open, close, high, low, volume, market cap, P/E ratio etc.
• Build machine learning models: Build machine learning models using Linear Regression, SVM, Decision Trees etc. 
• Validate the models: Validate the models using RMSE, AUC, etc.
• Predict future stock prices using the models.

Data Source
• Quandl: https://www.quandl.com/

Project setup instructions
• pip install -r requirements.txt
• python3 stockprediction.py <quandl-api-key>
