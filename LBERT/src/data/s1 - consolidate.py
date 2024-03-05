'''
The code is generating an enhanced training set by combining data from two existing datasets, filtering unnecessary data, processing text data,
and creating an aggregated label for events. The resulting dataset is stored as a CSV file.
'''
# Enhanced train set
import os
import re
import ast
import string
import nltk
import xmltodict
import pandas as pd
import numpy as np
from datetime import datetime
from unidecode import unidecode
from collections import defaultdict
from bs4 import BeautifulSoup 

# Load datasets: assetMonitEvent and cityWireNews to generate the dataset for training the initial model
ameDFv1 = pd.read_excel('data/raw/assetMonitEvent v1.0.xlsx',sheet_name='Events')
ameDFv2 = pd.read_excel('data/raw/assetMonitEvent v2.0.xlsx',sheet_name='Events')
cityDFv1 = pd.read_excel('data/raw/cityWireNews v1.0.xlsx',sheet_name='Historical')
cityDFv2 = pd.read_excel('data/raw/cityWireNews v2.0.xlsx',sheet_name='Historical')

ameDFv1['date'] = pd.to_datetime(ameDFv1['Date'], origin='1899-12-30', unit='D')
ameDFv1['date'] = ameDFv1['date'].dt.strftime('%Y-%m-%d')
ameDFv2['date'] = ameDFv2['Date'].dt.strftime('%Y-%m-%d')

cityDFv1['date'] = pd.to_datetime(cityDFv1['Publication Date'], origin='1899-12-30', unit='D')
cityDFv1['date'] = cityDFv1['date'].dt.strftime('%Y-%m-%d')
cityDFv2['date'] = pd.to_datetime(cityDFv2['Publication Date'], origin='1899-12-30', unit='D')
cityDFv2['date'] = cityDFv2['date'].dt.strftime('%Y-%m-%d')

ameDF = pd.concat([ameDFv1,ameDFv2],axis=0).reset_index(drop=True)
cityDF = pd.concat([cityDFv1,cityDFv2],axis=0).reset_index(drop=True)

# Clean dataset: remove articles with missing titles, unnecesary columns, rename columns, parse variables, and merge datasets 
ameDF = ameDF[['date','Event','Event type','Lead Type','Source']]
ameDF.columns = ['date','title','label','opportunity','source']

ameDF = ameDF.iloc[ameDF['title'].notna().index,:]
ameDF = ameDF.sort_values(by='date',ascending=True).reset_index(drop=True)
ameDF = ameDF.drop_duplicates(['title','label'],keep='last').reset_index(drop=True)
ameDF.title = ameDF.title.apply(lambda x: '' if pd.isna(x) else unidecode(x))

cityDF.columns = ['link','author','title','description','datePub','content','date']
cityDF = cityDF[['date','author','title','description','content','link']]
cityDF = cityDF.iloc[cityDF['title'].notna().index,:]
cityDF = cityDF.sort_values(by='date',ascending=True).reset_index(drop=True)
cityDF = cityDF.drop_duplicates(['title','description'],keep='last').reset_index(drop=True)

cityDF.author = cityDF.author.apply(lambda x: '' if pd.isna(x) else x.title()).apply(lambda x: x.replace('\n',''))
cityDF.author = cityDF.author.apply(lambda x: x.replace(', ',' and ')).apply(lambda x: unidecode(x))

cityDF.title = cityDF.title.apply(lambda x: '' if pd.isna(x) else unidecode(x))
cityDF.description = cityDF.description.apply(lambda x: '' if pd.isna(x) else unidecode(x))

dataDF = ameDF.merge(cityDF,how='left',on=['title','date'])
dataDF = dataDF.loc[dataDF.content.notna(),].drop_duplicates(['title','description'],keep='last').reset_index(drop=True)

# Generate ID var each article based on the url
pattern = r'a[0-9]{6,7}'
for idx, cols in dataDF.iterrows():
    url = dataDF.loc[idx,'link']
    id = re.findall(pattern, url)
    if len(id) !=0:
        dataDF.loc[idx,'id'] = id[0]
    else:
        dataDF.loc[idx,'id'] = ''

dataDF = dataDF[['date','id','title','label','opportunity','source','author','description','content', 'link']]

# Parse the text from the content and count how many links they are referring to
content = []
num_links = []
for idx,row in dataDF.iterrows():
    # Parse the html format
    soup = BeautifulSoup(dataDF['content'][idx], features="html.parser")
    # Count the number of links present in the content section
    links = soup.find_all('a')
    num_links.append(len(links))
    # Remove unnecesary elements from the content
    for script in soup(["script", "style"]):
        script.extract()
    # Get all the text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = unidecode(text).replace('  ',' ')
    content.append(text)

dataDF.loc[:,'content'] = content
dataDF.loc[:,'nurl'] = num_links

# Clean text from title, description,and content
patterns = ['Exclusive: ','Exclusive : ','\(Update\) ','Updated: ','NEW YORK: ','Citywire + rated ','Citywire AAA-rated manager ','Citywire A-rated ','Citywire AA-rated ','Citywire AAA-rated ','\nCitywire Verdict:\n','p>']
replace_pattern = '|'.join(patterns)
def remove_patterns(text):
    return re.sub(replace_pattern, '', text)

dataDF['title'] = dataDF['title'].apply(remove_patterns)
dataDF['description'] = dataDF['description'].apply(remove_patterns)
dataDF['content'] = dataDF['content'].apply(remove_patterns)

dataDF['title'] = dataDF['title'].replace("’", "'").replace("‘", "'").replace("`", "'")
dataDF['description'] = dataDF['description'].replace("’", "'").replace("‘", "'").replace("`", "'")
dataDF['content'] = dataDF['content'].replace("’", "'").replace("‘", "'").replace("`", "'")

dataDF['title'] = dataDF['title'].apply(lambda x: x.strip())
dataDF['description'] = dataDF['description'].apply(lambda x: x.strip())
dataDF['content'] = dataDF['content'].apply(lambda x: x.strip())

# Store the results
dataDF.to_csv('data/interim/s1_consolidate.csv',index=False)




