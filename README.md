# Stock Price Predictor

### Objective

Predict future stock prices using the historical stock data by applying different supervised learning machine learning algorithms.

### Workflow

• Acquire historical stock data from Quandl: The plan is to acquire the stock data of some of the popular stocks in sectors
  like energy, finance, health care, pharmaceutical, technology and also the S&P500 index data from Quandl.
  
• Preprocess the acquired stock data: Cleaning of data based on model requirement.

• Extract relevant features: Open, close, high, low, volume etc.

• Build machine learning models: Build machine learning models using Linear Regression, SVM, Decision Trees etc. 

• Validate the models: Evaluate the models by finding out the cross validation scores.

• Predict future stock prices using the models.


### Data Source

• Quandl: https://www.quandl.com/


### Project setup instructions

```
• pip3 install -r requirements.txt

• python3 stockpriceprediction.py <quandl-api-key>
```
If you do not want to pull new data from quandl, pass a random string (eg. a) instead of the quandl-api-key. If you want to pull new data, then you have to create a free account on quandl and use your api key for quandl.
