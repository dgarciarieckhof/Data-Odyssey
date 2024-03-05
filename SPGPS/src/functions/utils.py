import os
import re
import time
from concurrent import futures
from datetime import datetime
from operator import ne
from pathlib import Path
import pickle
import string

import numpy as np
import pandas as pd
import requests
import torch

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy

from lxml import etree
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import src.functions.utils as utils
import math as mt
import random


def create_embeddings(text, modelName):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentenceTransformer(modelName, device=device)
    sentences = nltk.sent_tokenize(text)
    sentence_embeddings = model.encode(sentences)
    article_embedding = np.mean(sentence_embeddings, axis=0)
    return article_embedding

def get_embeddings(content_text,modelName):
    sym, text = content_text
    embedding = create_embeddings(text,modelName)
    return sym, embedding


def scrape_description(link, driver):
    # Retrieve data
    text = ""
    try:
        driver.get(link)
        infobox = driver.find_element(By.XPATH, '//table[@class="infobox vcard"]')
        # find all siblings of the infobox element
        siblings = infobox.find_elements(By.XPATH,'following-sibling::*')
        # loop through the siblings until the desired h2 element is found
        for sibling in siblings:
            if sibling.tag_name == 'h2':
                break
            elif sibling.tag_name == 'p':
                text += sibling.text + '\n'
        time.sleep(1)
    except:
        text = 'empty'
    return text

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove characters between parentheses and the parentheses themselves
    text = re.sub(r'\([^)]*\)', '', text)    
    # Tokenize into words
    words = word_tokenize(text)
    # Remove stop words and noise words
    stop_words = set(stopwords.words('english'))
    noise_words = set(['also', 'however', 'although', 'since', 'therefore'])
    words = [w for w in words if not w in stop_words.union(noise_words)]
    # Lemmatize remaining words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    # Join words back into a string
    cleaned_text = ' '.join(words)

    return cleaned_text
