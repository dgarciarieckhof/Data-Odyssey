import re
import time
import random
import string
import pickle
import profile
import requests
import math as mt
import numpy as np
import pandas as pd
from tqdm import tqdm
from lxml import etree
from operator import ne
from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
import src.functions.utils as utils
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains

tqdm.pandas()

# General parameters
header = ({'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'})

# Get the S&P 500 companies
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
page = requests.get(url,header)

soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find('table', {'class': 'wikitable sortable'})

data = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    company_symbol = cells[0].text.strip()
    company_name = cells[1].text.strip()
    company_link = 'https://en.wikipedia.org' + cells[1].find('a')['href']
    company_sector = cells[2].text.strip()
    company_sub_industry = cells[3].text.strip()
    company_location = cells[4].text.strip()
    date_added = cells[5].text.strip()
    cik = cells[6].text.strip()
    founded = cells[7].text.strip()
    data.append((company_symbol, company_name, company_link, company_sector, company_sub_industry, company_location, date_added, cik, founded))

df = pd.DataFrame(data, columns=['symbol', 'security', 'link', 'gics_sector', 'gics_sub_industry', 'hq_loc', 'date_added', 'cik', 'founded'])

# correct links
df['link'] = df['link'].apply(lambda x: x.replace('%26', '&'))
df['link'] = df['link'].apply(lambda x: x.replace('%27', "'"))
df['link'] = df['link'].apply(lambda x: x.replace('%C3%A9', "é"))
df['link'] = df['link'].apply(lambda x: x.replace('%E2%80%93', "–"))

# Set up the driver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--incognito")
options.add_argument(f'user-agent={header["User-Agent"]}')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)
df['description_'] = df['link'].progress_apply(lambda x: utils.scrape_description(x,driver))
driver.quit()

# Clean description variable
df['description_'] = df['description_'].apply(lambda x: re.sub(r'\[\d+\]', '', x))
df['description_'] = df['description_'].apply(lambda x: x.replace('\n', ' '))
df['description_'] = df['description_'].apply(lambda x: re.sub(' +', ' ', x).strip())

# define function to replace commas in numbers
def replace_commas(match):
    return match.group().replace(",", "")

df['description_'] = df['description_'].apply(lambda x: re.sub(r'\d+,\d+', replace_commas, x))

# Save the output 
df.to_pickle('data/external/sp500_companies.pkl')