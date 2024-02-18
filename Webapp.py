#import Module

import streamlit as st , pandas as pd ,numpy as np , yfinance as yf

import plotly.express as px

from datetime import datetime 
from datetime import timedelta 
from datetime import date 

from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge


from sklearn.metrics import r2_score, mean_absolute_error


class KeyValue:
    def __init__(self, key, value):
        self.key = key
        self.value = value

class PredictionModel:
    def __init__(self, rscore, mea,lst):
        self.rscore = rscore
        self.mea = mea
        self.lst = lst


def get_IndexBySymbol(values, name):
    ind=0
    n=0
    for val in values:        
        if val==name:        
            ind=n
            return ind
        n += 1
    return ind
    


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


def model_engineList(model, num):
    list = []

    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    p1 = PredictionModel(0,0,list) 
    p1.rscore=r2_score(y_test, preds)
    p1.mea=mean_absolute_error(y_test, preds)

    
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 0
    df = pd.DataFrame(columns=['Day','Price'])

    for i in forecast_pred:
        #list.append(KeyValue(f'Day {day}',i))
       # df.loc[day,day+1] = i
       x1 = datetime.now()+timedelta(days=day+1)
       df.loc[len(df)] = [x1.strftime("%d/%m/%y"),i]       
       day =day+ 1
    p1.lst=df
    return p1


#title
st.title("Stock Dashboard")

stockexchange = st.sidebar.radio(
    "Select Stock Exchange ",
    ["NIFTY 50","TSX 60"],
     horizontal=True
)



x = datetime.now()+timedelta(weeks=-730)
companyName=''
index=0


if stockexchange == 'TSX 60':
    stocks=pd.read_html("https://en.wikipedia.org/wiki/S%26P/TSX_60")
    df = pd.DataFrame(stocks[0], columns=['Company', 'Symbol'])
    name = df['Company'].tolist()
    symbol = df['Symbol'].tolist()
    dic = dict(zip(symbol, name))
    ticker = st.sidebar.selectbox('Choose a Stocks', symbol, format_func=lambda x: dic[x])
    index=get_IndexBySymbol(symbol,ticker)
    companyName=list(name)[index]
    startdate=st.sidebar.date_input("Start Date",x)
    enddate=st.sidebar.date_input("End Date")
    data=yf.download(ticker,start=startdate,end=enddate)   
else:
    stocks=pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
    df = pd.DataFrame(stocks[2], columns=['Company name', 'Symbol'])
    name = df['Company name'].tolist()
    symbol = df['Symbol'].tolist()
    dic = dict(zip(symbol, name))
    ticker = st.sidebar.selectbox('Choose a Stocks', symbol, format_func=lambda x: dic[x])
    index=get_IndexBySymbol(symbol,ticker)
    companyName=list(name)[index]
    startdate=st.sidebar.date_input("Start Date",x)
    enddate=st.sidebar.date_input("End Date")
    data=yf.download(ticker+'.NS',start=startdate,end=enddate) 

scaler = StandardScaler()

fig=px.line(data,x = data.index, y = data['Adj Close'], title=companyName)
st.plotly_chart(fig)

pricing_data,prediction,techInd=st.tabs(["Pricing Data","Prediction"," Technical Indicator"])

with pricing_data:
    st.header("Pricing Data for Last 10 days")
    dataP=data.tail(n=10)
    dataP['% Change']=(data['Adj Close']/data['Adj Close'].shift(1))-1
    dataP.dropna(inplace=True)
    st.write(dataP)
    annualR=dataP['% Change'].mean()*252*100
    st.write('Annual Return is ',annualR,'%')

    stdev= np.std( dataP['% Change'] )*np.sqrt(252)
    st.write('Standard Deviation is is ',stdev*100,'%')




from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor




with prediction:
    st.header("Prediction for Next 5 days")
    num = int(5)
    LR,RFR,ETR,KNR,XR,BR,SR,CB,LGBM =st.tabs(["Linear","RandomForest"," ExtraTrees","KNeighbors","XGB"," BayesianRidge"," SGD","CatBoost","LGBM"])
    
    with LR:
        st.write('Linear Regression')
        datalist=[]
        engine = LinearRegression()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with RFR:
        st.write('Random Forest')
        datalist=[]
        engine = RandomForestRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with ETR:
        st.write('Extra Trees')
        datalist=[]
        engine = ExtraTreesRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with KNR:
        st.write('KNeighbors')
        datalist=[]
        engine = KNeighborsRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with XR:
        st.write('XGB')
        datalist=[]
        engine = XGBRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with BR:
        st.write('BayesianRidge')
        datalist=[]
        engine = BayesianRidge()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with SR:
        st.write('SGD')
        datalist=[]
        engine = SGDRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    
    with CB:
        st.write('CatBoost')
        datalist=[]
        engine = CatBoostRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)
    with LGBM:
        st.write('LGBMRegressor')
        datalist=[]
        engine = LGBMRegressor()
        datalist=model_engineList(engine, num)      
        st.write(datalist.lst)

with techInd:
    st.header('Technical Indicators')
    cp,bb,macd,rsi,sma,ema=st.tabs(["Close Price","BollingerBands"," MACD","RSI","SMA"," EMA"])
    
    with cp:
        st.write('Close Price')
        st.line_chart(data.Close)
    
    with bb:
         # Bollinger bands
        bb_indicator = BollingerBands(data.Close)
        bb = data
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        # Creating a new dataframe
        bb = bb[['Close', 'bb_h', 'bb_l']]
        st.write('BollingerBands')
        st.line_chart(bb)
    
    with macd:
        st.write('Moving Average Convergence Divergence')
        # MACD
        macd = MACD(data.Close).macd()
        st.line_chart(macd)

    with rsi:
        st.write('Relative Strength Indicator')
        # RSI
        rsi = RSIIndicator(data.Close).rsi()
        st.line_chart(rsi)    
    with sma:
        st.write('Simple Moving Average')
        # SMA
        sma = SMAIndicator(data.Close, window=14).sma_indicator()
        st.line_chart(sma)  
        
    with ema:
        st.write('Expoenetial Moving Average')
        # EMA
        ema = EMAIndicator(data.Close).ema_indicator()
        st.line_chart(ema)



    



   