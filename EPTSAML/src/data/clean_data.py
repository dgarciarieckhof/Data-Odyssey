# Clean retrieve data from API

# Libs
import os
import sys
import glob
import requests
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load FinViz data
finvizDF = pd.read_feather('data/external/universe.feather')

# Cleaned numeric variables
def cleanMC(x):
    """Converts a string with a numerical value and a suffix character denoting the unit to a float value.
    Args:
        x (str): A string containing a numerical value and a suffix character denoting the unit.     
    Returns:
        float: The converted float value.
    """    
    # Extract the numerical part of the string by slicing the string from the beginning up to the last character
    num = float(x[:-1])
    # Extract the suffix character by slicing the last character of the string
    val = x[-1]
    # Convert the numerical value based on the suffix character and return the result
    if val == 'B':
        return num
    elif val == 'M':
        return num/(10**3)
    elif val == 'K':
        return num/(10**6)
    else:
        return num/(10**9)

def cleanPCT(x):
    """Converts a string with a percentage value to a float value.
    Args:
        x (str): A string containing a percentage value.
    Returns:
        float: The converted float value as a percentage rounded to four decimal places.
    """    
    # Extract the numerical part of the string by slicing the string from the beginning up to the last character
    num = float(x[:-1])
    # Extract the suffix character by slicing the last character of the string
    val = x[-1]
    # Conver the string percentage value into a float value
    return np.round(num/100,4)

# Convert market cap into floats and express it in billions
finvizDF['Market Cap'] = finvizDF['Market Cap'].apply(lambda x: cleanMC(x))
# Convert the price to earning into float
finvizDF['P/E'] = finvizDF['P/E'].apply(lambda x: np.nan if x == '-' else float(x))
# Convert prices into floats
finvizDF['Price'] = finvizDF['Price'].apply(lambda x: np.nan if x == '-' else float(x))
# Convert Change into a float
finvizDF['Change'] = finvizDF['Change'].apply(lambda x: cleanPCT(x))
# Convert volume to be expressed in millions
finvizDF['Volume'] = finvizDF['Volume'].apply(lambda x: float(x)/(10**6))

# Stored the clean output
finvizDF.to_feather('data/processed/universe.feather')