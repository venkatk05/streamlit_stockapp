# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:47:30 2021

@author: venka
"""
# =============================================================================
# LIBRARIES IMPORTED FOR RUNNING CODE. 
# USE <streamlit run stock_project.py> to run file in conda command prompt
# =============================================================================
import streamlit as st
import yfinance as yf
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request

# =============================================================================
# 
# FUNCTIONS FOR THE MAIN CODE
# 
# =============================================================================


def get_data(ticker):
    data=yf.download(ticker,start,todays)   #getting stock data from yfinance API
    data.reset_index(inplace=True)
    return data

@st.cache(suppress_st_warning=True)
def delta_stock(ticker,start,end):
    data1=yf.download(stock,start,end)  #calculating overall stock price change between 2 dates 
    data1.reset_index(inplace=True)
    epsilon = 1e-6   
    d=data1.iloc[0]['Close']
    d1=data1.iloc[-1]['Close']
    s=d
    e=d1
    difference = round(e - s, 2)
    change = round(difference / (s + epsilon) * 100, 2)
    e = round(e, 2)
    cols = st.columns(2)
    (color, marker) = ("green", "+") if difference >= 0 else ("red", "-")
    cols[0].markdown(
            f"""<p style="font-size: 120%;margin-left:5px">{ticker} \t Latest price:${e}</p>""",
            unsafe_allow_html=True)
    cols[1].markdown(
            f"""<p style="color:{color};font-size:120%;margin-right:3px">{marker} \t {difference} {marker} {change} % </p>""",
            unsafe_allow_html=True) 


@st.cache(suppress_st_warning=True)
def plot_stock():         #plotting historic stock open and close prices                          
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name='Opening Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],name='Closing Price'))
    fig.layout.update(template='plotly_dark',xaxis_rangeslider_visible=True,showlegend=True)
    st.markdown("**Stock Open/Close Prices**")
    st.plotly_chart(fig,use_container_width=True)


@st.cache(suppress_st_warning=True)
def candlestick_stock(data):           #candlestock pattern graph
    fig=go.Figure(data=go.Candlestick(x=data['Date'],
                                      open=data['Open'],
                                      high=data['High'],
                                      low=data['Low'],
                                      close=data['Close'],
                                      name=stock))
    fig.update_xaxes(type='category')
    fig.update_layout(template='plotly_dark')
    st.markdown("\n**Stock Candlestick Graph**")
    st.markdown('''Candlestick charts are a technical tool that packs data for multiple time frames into single price bars. 
                This makes them more useful than traditional open-high, low-close bars or simple lines that connect the dots of closing prices.''')
    st.plotly_chart(fig, use_container_width=True)

    
@st.cache(suppress_st_warning=True)
def bb_band(df):                       #Bollinger band pattern for a stock
    indicator_bb = BollingerBands(df['Close'])
    bb = df
    fig=go.Figure()
    bb['bb_h'] = indicator_bb.bollinger_hband()
    bb['bb_l'] = indicator_bb.bollinger_lband()
    st.markdown("**Stock Bollinger Band**")
    st.markdown('''A Bollinger Band is a technical analysis tool defined by a set of trendlines plotted 
                two standard deviations (positively and negatively) away from the stock's closing price, but which can be adjusted to user preferences''')
    fig.add_trace(go.Scatter(x=df['Date'], y=bb['bb_h'], name='Upper band'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Closing Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=bb['bb_l'], name='Lower Band'))
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
  
    
@st.cache(suppress_st_warning=True)
def macd_chart(df):
    ef = MACD(df['Close'])._emafast   # Volume and Moving Average Convergence Divergence Chart
    es = MACD(df['Close'])._emaslow 
    st.markdown('***Stock Moving Average Convergence Divergence (MACD)***\n')
    st.markdown('''Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows 
                   the relationship between the 26-period exponential moving average (EMA) from the 12-period EMA.''')
    fig=make_subplots(vertical_spacing = 0, rows=2, cols=1, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=df['Date'], y=ef, name='26 day EMA'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=es, name='12 day EMA'),row=1,col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],name='Volume', marker_color='#109618'),row=2,col=1)
    fig.update_layout(template='plotly_dark',xaxis_rangeslider_visible=False,
                  xaxis=dict(zerolinecolor='black'))
    fig.update_xaxes(showline=False, linewidth=1, linecolor='white', mirror=False) 
    st.plotly_chart(fig, use_container_width=True)

@st.cache(suppress_st_warning=True)
def rsi_chart(df):                 #Relative Strength Index performance chart
    rsi = RSIIndicator(df['Close']).rsi()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=rsi, name='RSI'))
    fig.add_trace(go.Scatter(x=df['Date'], y=[70] * len(df['Date']),
                name='Overbought', marker_color='#109618',
                line = dict(dash='solid'))) 
    fig.add_trace(go.Scatter(x=df['Date'], y=[30] * len(df['Date']),
                name='Oversold', marker_color='#F55959',
                line = dict(dash='solid')))
    fig.update_layout(template='plotly_dark')
    st.markdown("**Stock Relative Strength Index **")
    st.markdown('''The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset. 
                The RSI is displayed as an oscillator and can have a reading from 0 to 100.''')
    st.plotly_chart(fig, use_container_width=True)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8') #Convert the dataframe tables into csv file
   

def get_cash_flow(tick):           
    cf=tickerData.get_cashflow()   #Returns cash flow statement of a stock
    cf.columns=['FY21','FY20','FY19','FY18']
    st.subheader(f"\nCash Flow statement of {tickerData.info['shortName']}")
    st.write(cf)
    csv=convert_df(cf)
    st.download_button("Download Cash Flow Statement", data=csv,  #Downloads the data into csv file 
                       file_name=f"{tick.info['shortName']}_Cash Flow.csv" ,mime='text/csv')
    

def get_balance_sheet(tick): 
    bs=tick.get_balance_sheet()  #Returns balance sheet statement of a stock
    bs.columns=['FY21','FY20','FY19','FY18']
    st.subheader(f"\nBalance Sheet of {tick.info['shortName']}")
    st.write(bs)
    csv=convert_df(bs)
    st.download_button("Download Balance Sheet Statement", data=csv, 
                       file_name=f"{tick.info['shortName']}_Balance sheet.csv" ,mime='text/csv')
    


def get_stake_holders(tick):    
    inv=tickerData.get_institutional_holders() #Returns top stakeholders of a stock
    st.subheader(f"\nTop Stakeholders of {tickerData.info['shortName']}")
    st.write(inv.head(10))


def get_dividends(tick):
    ds=tickerData.get_actions()   #Returns dividends & stock splits of a stock 
    st.subheader(f"\n Dividends and Stock Splits of {tickerData.info['shortName']}")
    st.write(ds.tail(10))

    
def get_financial(tick):
    ff=tick.get_financials()      #Returns overall financials of a stock
    ff.columns=['FY21','FY20','FY19','FY18']
    ff=ff.dropna()
    st.subheader(f"\n Financials Statement of {tick.info['shortName']}")
    st.write(ff)
    csv=convert_df(ff)
    st.download_button("Download Financials Statement", data=csv, file_name=f"{tick.info['shortName']}_Financials.csv" ,mime='text/csv')


def get_info(tick):       #Displays general info of a stock
    stock_logo= '<img src=%s>' % tick.info['logo_url']
    st.markdown(stock_logo, unsafe_allow_html=True)
    stock_name = tick.info['longName']
    st.header('**%s**' % stock_name)
    st.subheader(f"Sector:{tick.info['sector']}")
    stock_summary=tick.info['longBusinessSummary']
    st.info(stock_summary)
            
@st.cache(suppress_st_warning=True)
def sentiment_analysis(ticker):       #Sentiment Analysis on latest news of stock
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})  #getting the url access from finviz for news data
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser') #BeautifulSoup parses the response in .html format
    news_table = html.find(id='news-table')    ##extracting news from the website
    news_tables[ticker] = news_table
    parsed_data = []
    for ticker, news_table in news_tables.items():
    
        for row in news_table.findAll('tr'):
    
            title = row.a.text
            date_data = row.td.text.split(' ')
    
            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
    
            parsed_data.append([ticker, date, time, title])   # final parsed dataframe for sentiment analysis 
    
    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
    st.markdown("** Multiple News instances collected for each day")
    st.write(df)
    vader = SentimentIntensityAnalyzer()   #NLTK package which analyzes the sentiment of news titles
    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date
    
    mean_df = df.groupby(['ticker', 'date']).mean()  #Finding average compound scores for each date 
    mean_df=mean_df.reset_index(inplace=False)
    st.markdown("** Computed compund sentimental scores for analysis ")
    st.write(mean_df)
    fig=go.Figure([go.Bar(x=mean_df['date'], y=mean_df['compound'])])
    st.plotly_chart(fig)               #Plotting sentiment scores based on news titles 

    
def forecast_price(stock,period):    
    data=get_data(stock)            #building model using the fb prophet model to forecast time-series data
    df_train=data[['Date','Close']]
    df_train=df_train.rename(columns={'Date':'ds', 'Close':'y'})
    m=Prophet()        
    m.fit(df_train)
    future=m.make_future_dataframe(periods=period)
    forecast=m.predict(future)
    st.subheader('Forecast data:')
    st.write(forecast.tail())
    st.markdown('**Forecast Plot**')   
    fig1= plot_plotly(m, forecast)   
    fig1.update_layout(template='plotly_dark')
    st.plotly_chart(fig1)
    st.markdown('**Forecast Components**')
    fig2= m.plot_components(forecast)          #forecast components 
    st.write(fig2)
    
# =============================================================================
# 
#                               MAIN CODE
#
# =============================================================================

# =============================================================================
# Dashboard navigation 
# =============================================================================
st.sidebar.header("Dashboard")
option=st.sidebar.selectbox("Navigation Panel", ("Stock Fundamentals","Technical Analysis","Sentimental Analysis","Forecasting"))
st.title("Stock Analysis Application")
# =============================================================================
# Different selection options
# =============================================================================
if option=='Technical Analysis':    #Technical Analysis Page
    st.header(option)
    st.image('https://images.axios.com/xZxRe3Wn_XMfYm5sCalZuwX9Mc8=/0x0:1920x1080/1920x1080/2020/06/09/1591702479502.gif')
    try:
        stock=st.text_input('Enter Stock Ticker for Analysis:','AAPL')
        start=st.date_input('Start Date', value=pd.to_datetime('2018-01-01'))
        todays=st.date_input('End Date', value=pd.to_datetime('today'))
        data=get_data(stock)
        delta_stock(stock,start,todays)
        st.markdown(f"**View Technical Indicators of {stock} :**")   #Checkbox options to choose different technical indicators
        c1= st.checkbox(f"{stock} CandleStick Pattern")      
        c2= st.checkbox(f"{stock} Bollinger Bands")
        c3= st.checkbox(f"{stock} Volume & Moving Average Convergence/Divergence")
        c4= st.checkbox(f"{stock} Relative Strength index")
        if c1:     
            candlestick_stock(data)  #CandleStick chart
        if c2:
            bb_band(data)           #Bollinger Band chart
        if c3:
            macd_chart(data)        #Volume & MACD chart
        if c4:
            rsi_chart(data)         #RSI chart
    except KeyError:      #error handling if no/incorrect ticker symbol entered
        st.warning("Enter a valid ticker!")
          
if option=='Stock Fundamentals':
    st.header(option)
    st.image('https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy-downsized-large.gif')
    try:
        stock=st.text_input('Enter Stock Ticker for Fundamentals:','AAPL')
        tickerData = yf.Ticker(stock)   
        get_info(tickerData)
        st.markdown('**Select to view:**')  #Checkbox options to choose different stock fundamentals
        opt_1 = st.checkbox(f"Stock Financials of {tickerData.info['shortName']}")
        opt_2 = st.checkbox(f"Balance Sheet Statement of {tickerData.info['shortName']}")
        opt_3 = st.checkbox(f"Cash Flow Statement of {tickerData.info['shortName']}")
        opt_4 = st.checkbox(f"Top Stakeholders of {tickerData.info['shortName']}")
        opt_5 = st.checkbox(f"Dividends & Stock Spilts of {tickerData.info['shortName']}")
        opt_6 = st.checkbox(f"Historical Stock Data of {tickerData.info['shortName']}")
        if opt_1:
            get_financial(tickerData)
        if opt_2:    
            get_balance_sheet(tickerData)
        if opt_3:
            get_cash_flow(tickerData)
        if opt_4:
            get_stake_holders(tickerData)
        if opt_5:
            get_dividends(tickerData)
        if opt_6:
            st.subheader("\nStock Data:\n")
            start=st.date_input('Start Date', value=pd.to_datetime('2018-01-01'))
            todays=st.date_input('End Date', value=pd.to_datetime('today'))
            data=get_data(stock)
            csv=convert_df(data)  
            st.download_button("Download Historical Data", data=csv, 
                               file_name=f"{tickerData.info['shortName']}_Stock_Prices.csv" ,mime='text/csv')
            st.write(data.tail(10))
            plot_stock()
    except KeyError:      #error handling if no/incorrect ticker symbol entered
        st.warning("Enter a valid ticker!")
        
if option=='Sentimental Analysis':
    st.header(option)
    st.image('https://media2.giphy.com/media/l0HlDDyxBfSaPpU88/giphy.gif')
    try:
        stock=st.text_input('Enter Ticker:','MSFT')  
        tick=yf.Ticker(stock)
        st.markdown('**Choose to see:**')
        ch1=st.checkbox(f"Trending News of {tick.info['shortName']}")   #GUI option to select choice
        ch2=st.checkbox(f"Sentimental Analysis of {tick.info['shortName']}")
        if ch1:
            news=tick.get_news()
            st.subheader(f"\nTop {tick.info['shortName']} News \n")
            for n in news:          #scrapping news headlines for a ticker
               l=n['link']      
               link= f"[Read more]({l})"
               st.markdown(f"**{n['title']}**"+'\n')
               st.write(link)
        if ch2:
            st.subheader("Sentiment Analysis for ")
            sentiment_analysis(stock)     #performing sentimental analysis for the stock news
    except KeyError:
        st.warning("Enter a valid ticker!")
        
if option=='Forecasting':
    st.header(option)
    st.image('https://s.wsj.net/public/resources/images/ON-DE793_201909_G_20190830121038.gif')
    stock=st.text_input('Enter Stock Ticker for Analysis:','AAPL')
    start=st.date_input('Start Date', value=pd.to_datetime('2018-01-01'))
    todays=st.date_input('End Date', value=pd.to_datetime('today'))
    months=st.slider("Months of Prediction",1,24)
    period=months*30
    if st.button("Forecast!"):    #forecast button to forecast data
        forecast_price(stock,period)


    


