import os
import re
import umap
import time
import nltk
import spacy
import torch
import pickle
import string
import math as mt
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from concurrent import futures
from nltk.corpus import stopwords
import src.utils.utils as utils
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer

tqdm.pandas()

# -----------------
# Load data
dataDF = pd.read_csv('data/interim/s1_consolidate.csv')

# -----------------
# Clean content for each title
texts = dataDF['title'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

container = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    for idx, cl, ch in tqdm(executor.map(utils.clean_text, content.items()), total=len(content)):
        container[idx] = {'light': cl, 'heavy':ch}

dataDF['title_cl'] = [sub_dict['light'] for entry_key, sub_dict in container.items()]
dataDF['title_ch'] = [sub_dict['heavy'] for entry_key, sub_dict in container.items()]

# Clean content for each description
texts = dataDF['description'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

container = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    for idx, cl, ch in tqdm(executor.map(utils.clean_text, content.items()), total=len(content)):
        container[idx] = {'light': cl, 'heavy':ch}

dataDF['description_cl'] = [sub_dict['light'] for entry_key, sub_dict in container.items()]
dataDF['description_ch'] = [sub_dict['heavy'] for entry_key, sub_dict in container.items()]

# Clean content for each article
texts = dataDF['content'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

container = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    for idx, cl, ch in tqdm(executor.map(utils.clean_text, content.items()), total=len(content)):
        container[idx] = {'light': cl, 'heavy':ch}

dataDF['content_cl'] = [sub_dict['light'] for entry_key, sub_dict in container.items()]
dataDF['content_ch'] = [sub_dict['heavy'] for entry_key, sub_dict in container.items()]

# Generate full text variable for all the text variables formats
dataDF['fulltext'] = dataDF['title'] + '\n' + dataDF['description'] + '\n' + dataDF['content']
dataDF['fulltext_cl'] = dataDF['title_cl'] + '\n' + dataDF['description_cl'] + '\n' + dataDF['content_cl']
dataDF['fulltext_ch'] = dataDF['title_ch'] + '\n' + dataDF['description_ch'] + '\n' + dataDF['content_ch']

# How long are the sentences in full text
dataDF['fulltext_maxlen'] = dataDF['fulltext'].apply(lambda x: np.max([len(i) for i in x.split('\n')]))
dataDF['fulltext_cl_maxlen'] = dataDF['fulltext_cl'].apply(lambda x: np.max([len(i) for i in x.split('\n')]))
dataDF['fulltext_ch_maxlen'] = dataDF['fulltext_ch'].apply(lambda x: np.max([len(i) for i in x.split('\n')]))

# How long are the sentences in only content
dataDF['content_maxlen'] = dataDF['content'].apply(lambda x: np.max([len(i) for i in x.split('\n')]))
dataDF['content_cl_maxlen'] = dataDF['content_cl'].apply(lambda x: np.max([len(i) for i in x.split('\n')]))
dataDF['content_ch_maxlen'] = dataDF['content_ch'].apply(lambda x: np.max([len(i) for i in x.split('\n')]))

# -----------------
# Create sentence embeddings using finbert tokenizer, lengt 512 without special tokens
# Transpose the embedding matrix, each row is a vector that corresponds to an article
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

# Full text
texts = dataDF['fulltext'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.sentence_embedding, tokenizer=tokenizer)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embFt_df = pd.DataFrame(embeddings)
embFt_df = embFt_df.T

# Full text light cleaning
texts = dataDF['fulltext_cl'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.sentence_embedding, tokenizer=tokenizer)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embFtCl_df = pd.DataFrame(embeddings)
embFtCl_df = embFtCl_df.T

# Full text heavy cleaning
texts = dataDF['fulltext_ch'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.sentence_embedding, tokenizer=tokenizer)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embFtCh_df = pd.DataFrame(embeddings)
embFtCh_df = embFtCh_df.T

# Content text
texts = dataDF['content'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.sentence_embedding, tokenizer=tokenizer)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embCt_df = pd.DataFrame(embeddings)
embCt_df = embCt_df.T

# Content text light cleaning
texts = dataDF['content_cl'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.sentence_embedding, tokenizer=tokenizer)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embCtCl_df = pd.DataFrame(embeddings)
embCtCl_df = embCtCl_df.T

# Content text heavy cleaning
texts = dataDF['content_ch'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.sentence_embedding, tokenizer=tokenizer)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embCtCh_df = pd.DataFrame(embeddings)
embCtCh_df = embCtCh_df.T

# Indetifying similar categories
# Distance Matrices -> Reduce categories
temp = embCtCh_df.copy()
temp = temp.iloc[:,np.where(temp.mean(axis=0) != 0)[0]]
temp['labels'] = dataDF['label'].values
temp = temp.groupby('labels').mean()
distFt_dF = np.sqrt(0.5*(1-temp.T.corr()))

# -----------------
# Create sentence embeddings using a sentence transformer
modelName = 'paraphrase-distilroberta-base-v1'

# Full text
texts = dataDF['fulltext'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embFt_df = pd.DataFrame(embeddings)
embFt_df = embFt_df.T

# Full text light cleaning
texts = dataDF['fulltext_cl'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embFtCl_df = pd.DataFrame(embeddings)
embFtCl_df = embFtCl_df.T

# Full text heavy cleaning
texts = dataDF['fulltext_ch'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embFtCh_df = pd.DataFrame(embeddings)
embFtCh_df = embFtCh_df.T

# Content text
texts = dataDF['content'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embCt_df = pd.DataFrame(embeddings)
embCt_df = embCt_df.T

# Content text light cleaning
texts = dataDF['content_cl'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embCtCl_df = pd.DataFrame(embeddings)
embCtCl_df = embCtCl_df.T

# Content text heavy cleaning
texts = dataDF['content_ch'].values
idx = range(0,len(texts))
content = {idx:val for idx, val in zip(idx,texts)}

embeddings = {}
num_workers = 3
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for idx, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[idx] = embedding

embCtCh_df = pd.DataFrame(embeddings)
embCtCh_df = embCtCh_df.T

# Indetifying similar categories
# Distance Matrices -> Reduce categories
temp = embCtCh_df.copy()
temp = temp.iloc[:,np.where(temp.mean(axis=0) != 0)[0]]
temp['labels'] = dataDF['label'].values
temp = temp.groupby('labels').mean()
distFt_dF = np.sqrt(0.5*(1-temp.T.corr()))

# -----------------
# Reduce categories based on findings using sentence embeddings
newLabs = {
    'label': ['Changes in the Investment Team','Changes in the regulatory framework','Closure','Drop in AUM','Drop in ESG score','Fund merger','Hard closure','Launch of new products','Liquidation of a fund','Morningstar downgrade','Negative press','News at organisational level','Not a displacement opportunity','Portfolio manager change','RFP opportunity','Soft closure','Sustained underperformance'],
    'label_f': ['1. Change Inv. Team','5. Others Opps.','2. Closure','3. Underperformance','5. Others Opps.','2. Closure','2. Closure','4. Not Disp. Opp.','2. Closure','3. Underperformance','5. Others Opps.','1. Change Inv. Team','4. Not Disp. Opp.','1. Change Inv. Team','5. Others Opps.','2. Closure','3. Underperformance'],
    'code': ['CIT','OOP','CLS','UDP','OOP','CLS','CLS','NOP','CLS','UDP','OOP','CIT','NOP','CIT','OOP','CLS','UDP']
}
newLabs = pd.DataFrame(newLabs)
dataDF = dataDF.merge(newLabs,how='left',on='label')

# Distance Matrices -> Reduce categories
temp = embFt_df.copy()
temp = temp.iloc[:,np.where(temp.mean(axis=0) != 0)[0]]
temp['labels'] = dataDF['label_f'].values
temp = temp.groupby('labels').mean()
distFt_dF = np.sqrt(0.5*(1-temp.T.corr()))

# Store the results
dataDF.to_csv('data/processed/s1_traindata.csv',index=False)