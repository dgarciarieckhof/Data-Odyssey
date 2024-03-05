# Retrieve data from API
# Universe: US listed stocks, larger than 300MM in market cap, over 1M average volume
# Subset: The top 500 companies stocks by market cap
# Time spam: 2013/03 - 2023/03

# Libs
import os
import sys
import glob
import requests
import time
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

# General parameters
endDate = datetime.today()
initDate = endDate - relativedelta(years=10)
endDate = endDate.strftime('%Y-%m-%d')
initDate = initDate.strftime('%Y-%m-%d')

# Retrieve stocks that match the requirements from finviz
screener = {}
for i in tqdm(np.arange(start=1,stop=22*68,step=20)):
    # Set table page view
    url = 'https://finviz.com/screener.ashx?v=111&f=cap_smallover,sh_avgvol_o1000&o=-marketcap&r='+str(i)
    # Request page
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    html = soup(webpage, "html.parser")
    # Create table
    table = pd.read_html(str(html), attrs = {'class': 'table-light'})[0]
    tableNames = table.loc[0,].to_list()
    table = table.loc[1:,]
    table.columns = tableNames
    # Append table to dict
    screener['page'+str(i)] = table
    time.sleep(1)

# Data is already sorted by marketcap in decreasing order
screener = pd.concat(screener.values(), ignore_index=True)
screener.drop(['No.'],inplace=True,axis=1)
screener = screener.loc[screener.duplicated(keep='first') == False,]
screener.reset_index(drop=True,inplace=True)
screener.to_feather('data/external/universe.feather')

# Select the top 500 stocks based on market cap
stocksName = screener.loc[:499,'Ticker'].to_list()

# Retrieve price data for the stocks
stocksData = {}
fails = []
i = 1
for stock in tqdm(stocksName):
    # Retrieve data
    try:
        dataDict = yf.download(stock, start=initDate, end=endDate, interval='1d')
        dataDict = dataDict.reset_index(drop=False)
        dataDict['stock'] = stock
        stocksData[stock] = dataDict
    except:
        fails.append(stock)
    i = i + 1
    # Rest every 50 calls
    if i % 50 == 0:
        time.sleep(30) 

stockDF = pd.concat(stocksData.values(), ignore_index=True)
stockDF.reset_index(drop=True,inplace=True)
stockDF.to_feather('data/external/subsetprices.feather')