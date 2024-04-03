import os
import re
import nltk
import torch
import spacy
import random
import hdbscan
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from plotnine import *
from tqdm import trange
from hyperopt import hp
from umap.umap_ import UMAP
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
from mizani.formatters import percent_format, label_comma
from sklearn.feature_extraction.text import CountVectorizer
from hyperopt import STATUS_OK, fmin, tpe, space_eval, Trials


# Create n-grams bases on a text
def generate_ngrams(text, n):
    """
    Generate n-grams from the given text.
    Args:
        text (str): The input text.
        n (int): The value of n for n-grams.
    Returns:
        list: A list of n-grams.
    """
    # Split the text into tokens (words)
    tokens = text.split()  
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        ngrams.append(" ".join(ngram))
    return ngrams

# Generate clusters
def generate_clusters(text_embeddings, n_neighbors=15, n_components=5, min_cluster_size=10, min_samples=None, cluster_selection_method='eom', metric='euclidean', random_state=None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    Args:
        text_embeddings (array-like): Embeddings of texts/documents
        n_neighbors (int): Number of neighbors for UMAP
        n_components (int): Number of components for UMAP
        min_cluster_size (int): Minimum cluster size for HDBSCAN
        min_samples (int or None): Minimum number of samples in a cluster for HDBSCAN
        cluster_selection_method (str): Cluster selection method for HDBSCAN
        metric (str): Distance metric for HDBSCAN
        random_state (int or None): Random state for reproducibility        
    Returns:
        clusters (hdbscan.HDBSCAN): HDBSCAN cluster object
    """
    # Reduce embedding dimensionality with UMAP
    umap = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=random_state)
    umap_embeddings = umap.fit_transform(text_embeddings)
    # Generate HDBSCAN cluster object
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,                             
                                 cluster_selection_method=cluster_selection_method, 
                                 metric=metric,)
    clusters = clustering.fit(umap_embeddings)
    return clusters

# Scoring function
def score_clusters(clusters, prob_threshold=0.05):
    """
    Compute the label count and cost of a given cluster from HDBSCAN results.
    Args:
        clusters (HDBSCAN): Cluster object obtained from HDBSCAN
        prob_threshold (float): Probability threshold for considering cluster assignment
    Returns:
        label_count (int): Number of unique cluster labels
        cost (float): Cost of cluster assignment based on the given probability threshold
    """
    # Extract cluster labels and calculate label count
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    # Calculate the cost of cluster assignment
    total_num = len(cluster_labels)
    below_threshold = np.count_nonzero(clusters.probabilities_ < prob_threshold)
    cost = below_threshold / total_num
    return label_count, cost

# Objective function for optimization
def objective(params, embeddings, label_lower, label_upper, penalty_val=0.05):
    """
    Objective function for hyperopt to minimize, which incorporates constraints
    on the number of clusters we want to identify.
    Args:
        params (dict): Dictionary containing hyperparameters to optimize
        embeddings (array-like): Message embeddings
        label_lower (int): Lower bound on the number of clusters
        label_upper (int): Upper bound on the number of clusters
        penalty_val (float): Penalty to be applied
    Returns:
        dict: Dictionary containing the loss value, label count, and status
    """
    # Generate clusters using the specified hyperparameters
    clusters = generate_clusters(embeddings, 
	n_neighbors=params['n_neighbors'], 
	n_components=params['n_components'], 
	min_cluster_size=params['min_cluster_size'], 
	min_samples=params['min_samples'], 
	cluster_selection_method=params['cluster_selection_method'], 
	metric=params['metric'], 
	random_state=params['random_state'])
    # Score the clusters
    label_count, cost = score_clusters(clusters, prob_threshold=0.05)
    # Apply penalty if the number of clusters is outside the desired range
    if (label_count < label_lower) or (label_count > label_upper):
        penalty = penalty_val
    else:
        penalty = 0
    # Compute the loss function
    loss = cost + penalty
    # Return results to Hyperopt
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}

# Bayesian search
def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100, penalty_val=0.05):
    """
    Perform Bayesian search on hyperopt hyperparameter space to minimize objective function.
    Args:
        embeddings (array-like): Message embeddings
        space (dict): Dictionary containing hyperparameter search space
        label_lower (int): Lower bound on the number of clusters
        label_upper (int): Upper bound on the number of clusters
        max_evals (int): Maximum number of evaluations
        penalty_val (float): Penalty to be applied
    Returns:
        tuple: Tuple containing the best parameters, best clusters, and search trials
    """
    # Initialize trials
    trials = Trials()
    # Define objective function with fixed parameters
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    # Perform Bayesian optimization
    best = fmin(fmin_objective,space=space,algo=tpe.suggest,max_evals=max_evals,trials=trials)
    # Retrieve best parameters
    best_params = space_eval(space, best)
    print('Best parameters:')
    print(best_params)
    print(f"Label count: {trials.best_trial['result']['label_count']}")    
    # Generate clusters using best parameters
    best_clusters = generate_clusters(embeddings, 
	n_neighbors=best_params['n_neighbors'], 
	n_components=best_params['n_components'], 
	min_cluster_size=best_params['min_cluster_size'], 
	min_samples=best_params['min_samples'], 
	cluster_selection_method=best_params['cluster_selection_method'], 
	metric=best_params['metric'], 
	random_state=best_params['random_state'])
    return best_params, best_clusters, trials

# Topic creation
def most_common(lst, n):
    """
    Return the n most common elements from the list.
    Args:
        lst (list): List of elements
        n (int): Number of most common elements to return
    Returns:
        list: List of tuples containing the most common elements and their counts
    """
    counter = Counter(lst)
    return counter.most_common(n)

def extract_labels(category_docs, nlp):
    """
    Extract labels from documents in the same cluster by concatenating
    most common verbs, objects, nouns, and adjectives.
    Args:
        category_docs (list): List of documents in the same cluster
        nlp (spacy object): Spacy corpus object 
    Returns:
        str: Extracted label
    """
    verbs = []
    dobjs = []
    nouns = []
    adjs = []
    # Extract verbs, objects, nouns, and adjectives from each document
    for doc in category_docs:
        for token in nlp(doc):
            if not token.is_stop:
                if token.dep_ == 'ROOT':
                    verbs.append(token.text.lower())

                elif token.dep_ == 'dobj':
                    dobjs.append(token.lemma_.lower())

                elif token.pos_ == 'NOUN':
                    nouns.append(token.lemma_.lower())

                elif token.pos_ == 'ADJ':
                    adjs.append(token.lemma_.lower())
    # Extract most common words of each form
    verb = most_common(verbs, 1)[0][0] if verbs else ''
    dobj = most_common(dobjs, 1)[0][0] if dobjs else ''
    noun1 = most_common(nouns, 1)[0][0] if nouns else ''
    noun2 = most_common(nouns, 2)[1][0] if len(set(nouns)) > 1 else ''
    # Concatenate the most common verb-dobj-noun1-noun2 (if they exist)
    label_words = [word for word in (verb, dobj, noun1, noun2) if word]
    label = '_'.join(label_words)
    return label


# Term Frequency-Inverse Document Frequency
def c_tf_idf(documents, m, ngram_range=(1, 1), stop_words='english'):
    """
    Calculate class-based TF-IDF (Term Frequency-Inverse Document Frequency) for a given set of documents.
    Args:
        documents (list): List of documents.
        m (int): Total, unjoined, number of documents.
        ngram_range (tuple): Range for n-grams.
        stop_words (str or list): Stop words to be removed.
    Returns:
        tf_idf (numpy.ndarray): Class-based TF-IDF matrix.
        count (CountVectorizer): CountVectorizer object.
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count