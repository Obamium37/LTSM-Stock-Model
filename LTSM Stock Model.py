import yfinance as yf
import pandas as pd
from datetime import date
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from datetime import timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


today = date.today()


Stocks = ['AAPL', 'AMZN', 'BAC', 'META', 'MFST', 'NFLX', 'SHW', 'TSLA']

for stock in Stocks:

    tickerSymbol = stock

    tickerData = yf.Ticker(tickerSymbol)

    tickerDf = tickerData.history(period='1d', start='2012-5-31', end=today)
    tickerDf = tickerDf.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis = 1)
    tickerDf = tickerDf.reset_index()
    tickerDf['Date'] = pd.to_datetime(tickerDf['Date']).dt.date
    df = tickerDf


    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil( len(dataset) *.93)


    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(70, len(train_data)):
        
        x_train.append(train_data[i-70:i, 0])
        y_train.append(train_data[i, 0])

        
    if i<= 70:
        
        print(x_train)
        print(y_train)


    x_train, y_train = np.array(x_train), np.array(y_train )

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    model=Sequential()
    model.add(LSTM(units=50,return_sequences=True, input_shape=(np.shape(x_train)[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))


    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(x_train, y_train, batch_size=1, epochs=1)


    test_data = scaled_data[training_data_len - 70:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(70 , len(test_data)):
        
        x_test.append(test_data[i-70:i,0])

    x_test = np.array(x_test)



    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))


    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))



    yesterday = today - timedelta(days = 1)


    tickerSymbol2 = stock

    tickerData2 = yf.Ticker(tickerSymbol)

    tickerDf2 = tickerData.history(period='1d', start='2012-5-31', end=yesterday)

    new_df = tickerDf2.filter(['Close'])

    last_70_days = new_df[-70:].values

    last_70_days_scaled = scaler.transform(last_70_days)

    X_test = []

    X_test.append(last_70_days_scaled)

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)

    pred_price = scaler.inverse_transform(pred_price)

    print(pred_price)



    tickerSymbol3 = stock

    tickerData3 = yf.Ticker(tickerSymbol)

    tickerDf3 = tickerData.history(period='1d', start=yesterday, end=today)


    filter = tickerDf3.filter(['Close'])

    prev_price = filter.values

    
    pred_price = pred_price.item()
    print("Predicted price of today is: " )
    print(pred_price)
    print("Close price of yesterday is ")
    print(tickerDf3['Close'])



    API_KEY = "PKURCN50OFCIQOOCQQKK"
    SECRET_KEY = "LafENNv5eFM8hJNBfbpeq0Q2SX9XWolxGH4xTiFC"

    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

    if pred_price >= prev_price:

        market_order_data = MarketOrderRequest(
                            symbol='AAPL',
                            qty=5,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
        market_order = trading_client.submit_order(market_order_data)
        for property_name, value in market_order:
            print(f"\"{property_name}\": {value}")