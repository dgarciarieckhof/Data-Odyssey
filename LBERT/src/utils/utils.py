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
from torch import Tensor
from functools import partial
from concurrent import futures
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from nltk.stem import WordNetLemmatizer
from typing import List, Optional, Tuple, Any, Union
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer, BatchEncoding, PreTrainedTokenizerBase

# Wrangling
def clean_text(texts: dict):
    """
    Cleans and preprocesses the text segments in parallel.
    Args:
        texts (dict): A dictionary containing the index and text segments.
    Returns:
        tuple: A tuple containing the index and the cleaned and preprocessed light and heavy text segments.
    """
    idx, text = texts
    # Convert to lowercase
    text = text.lower()
    # Remove characters between parentheses and the parentheses themselves
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r"'", '', text)
    # Split the text into segments
    segments = re.split(r'\n', text)
    # Light and heavy pre-processing
    light = []
    heavy = []
    for segment in segments:
        # Tokenize into words
        words = word_tokenize(segment)
        # Remove stop words and noise words
        stop_words = set(stopwords.words('english'))
        noise_words = set(['also', 'however', 'although', 'since', 'therefore'])
        words = [w for w in words if w not in stop_words.union(noise_words)]
        segmentL = ' '.join(words).replace(' %', '%').replace(" '", "'").replace(" ,", ",").replace(" .", ".")
        light.append(segmentL)
        # Lemmatize remaining words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        segmentH = ' '.join(words).replace(' %', '%').replace(" '", "'").replace(" ,", ",").replace(" .", ".")
        heavy.append(segmentH)
    # Output
    outputL = '\n'.join(light)
    outputH = '\n'.join(heavy)
    return idx, outputL, outputH

# Sentence embeddings
def create_embeddings(text, modelName):
    """
    Creates embeddings for the given text using the specified model.
    Args:
        text (str): The input text to create embeddings for.
        modelName (str): The name of the SentenceTransformer model to use.
    Returns:
        np.ndarray: The article embedding.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Create the SentenceTransformer model
    model = SentenceTransformer(modelName, device=device)
    # Tokenize the text into sentences
    # sentences = nltk.sent_tokenize(text)
    sentences = text.split('\n')
    # Encode the sentences to get sentence embeddings
    sentence_embeddings = model.encode(sentences)
    # Calculate the article embedding by taking the mean of all sentence embeddings
    article_embedding = np.mean(sentence_embeddings, axis=0)
    return article_embedding

def get_embeddings(content_text, modelName):
    """
    Retrieves embeddings for the content text using the specified model.
    Args:
        content_text (tuple): A tuple containing the symbol and the text of the content.
        modelName (str): The name of the SentenceTransformer model to use.
    Returns:
        tuple: A tuple containing the symbol and the embedding of the content.
    """
    idx, text = content_text
    # Create the embedding for the text using the specified model
    embedding = create_embeddings(text, modelName)
    return idx, embedding

# BERT for long text
def sentence_embedding(texts: dict, tokenizer):
    """
    Computes sentence embeddings for the given text using the specified tokenizer.
    Args:
        texts (dict): A dictionary containing the index and text.
        tokenizer: The tokenizer object.
    Returns:
        tuple: A tuple containing the index and the computed sentence embeddings.
    """
    idx, text = texts
    # Split the text into sentences
    sentences = text.split('\n')
    doc_segment = []
    # Embed each sentence and ensure the tensor has 512 dimensions
    for sentence in sentences:
        tokens = tokenizer(sentence, add_special_tokens=False, truncation=True, padding=False, return_tensors='pt')
        input_ids = tokens['input_ids'][0]
        if len(input_ids) <= 512:
            n = 512 - len(input_ids)
            input_ids = torch.cat([input_ids, torch.Tensor([0] * n)])
        doc_segment.append(input_ids)
    # Calculate the mean among all embeddings in the text
    doc_segment = torch.stack(doc_segment)
    embedding = doc_segment.numpy().mean(axis=0)
    # Return the embedding
    return idx, embedding