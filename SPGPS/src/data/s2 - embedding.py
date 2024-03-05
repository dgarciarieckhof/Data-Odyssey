import os
import re
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
from umap.umap_ import UMAP
from functools import partial
from concurrent import futures
from nltk.corpus import stopwords
import src.functions.utils as utils
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

tqdm.pandas()

# Load nltk corpus
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
compsDF = pd.read_pickle('data/external/sp500_companies.pkl')

# Clean the text variable
compsDF['description_cleaned'] = compsDF['description_'].progress_apply(utils.clean_text)

# Generate embeddings
ndim = np.quantile(compsDF['description_cleaned'].apply(len), q=np.linspace(0,1,11))[7] # percentile 80 for string len
if ndim <= 768:
    modelName = 'paraphrase-distilroberta-base-v1'
else:
    modelName = 'roberta-large-nli-stsb-mean-tokens'

cleaned_texts = compsDF['description_cleaned'].values
symbols = compsDF['symbol'].values
content = {idx:val for idx, val in zip(symbols,cleaned_texts)}

embeddings = {}
num_workers = 2
with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    fn = partial(utils.get_embeddings, modelName=modelName)
    for sym, embedding in tqdm(executor.map(fn, content.items()), total=len(content)):
        embeddings[sym] = embedding

embeddingsDF = pd.DataFrame(embeddings)

# Remove companies without description
comps = compsDF.loc[compsDF['description_'] != 'empty','symbol'].values
embeddingsDF = embeddingsDF.loc[:,comps]

# Transpose the embedding matrix, each row is a vector that corresponds to a company
embeddingsDF = embeddingsDF.T

# Store the results
embeddingsDF.to_pickle('data/processed/comp_embeddings.pkl')