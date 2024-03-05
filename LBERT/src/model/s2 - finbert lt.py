import os
import re
import json
import time
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from collections import Counter
from torch.nn import DataParallel, Module, Softmax
from typing import List, Optional, Tuple, Any, Union
from src.utils.splitting import transform_list_of_texts
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from src.utils.bert import collate_fn_pooled_tokens, TokenizedDataset, error_metrics
from transformers import BertForSequenceClassification, BertTokenizer, BatchEncoding

# -----------------
# Load data
dataDF = pd.read_excel('data/test/citywire_test.xlsx')

# -----------------
# Get the text variables and their classifications
texts = dataDF['content'].to_list()
labels = dataDF['label'].apply(lambda x: int(x.split()[0].replace('.',''))).to_list()

# Load pre trained models
num_labels = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain',num_labels=num_labels) 
model.load_state_dict(torch.load('models/finbert_lt_epoch_4.model'))
model.to(device)

# Tokenize the articles
chunk_size = 510
stride = 510
min_chunk = 1
max_length = None
batch_size = 24
tokens = transform_list_of_texts(texts,tokenizer,chunk_size,stride,min_chunk,max_length)
dataset = TokenizedDataset(tokens)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=collate_fn_pooled_tokens)

# Eval the nn
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

classes = []
model.eval()
for step, batch in enumerate(dataloader):    
    # get predictions
    input_ids = batch[0]
    attention_mask = batch[1]
    number_of_chunks = [len(x) for x in input_ids]
    # concatenate all input_ids into one batch
    input_ids_combined, attention_mask_combined = [], []        
    for x in input_ids:
        input_ids_combined.extend(x.tolist())
    for x in attention_mask:
        attention_mask_combined.extend(x.tolist())
    input_ids_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in input_ids_combined])
    attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in attention_mask_combined])
    # get model predictions for the combined batch
    with torch.no_grad():
        preds = model(input_ids_combined_tensors, attention_mask_combined_tensors)
        preds = preds[0].cpu()
        # split result preds into chunks
        preds_split = preds.split(number_of_chunks)
        # Pooled the prediction usin mean for items within the same article
        pooled_preds = torch.stack([x.mean(dim=0).reshape(num_labels) if len(x) > 1 else x.reshape(num_labels) for x in preds_split])
    # get classes 
    probs = torch.nn.functional.softmax(pooled_preds,dim=-1)
    pred_classes = [np.argmax(prob.tolist()) for prob in probs]
    classes.append(pred_classes)
classes = sum(classes, [])

# Append classes to articles
labels_l = ['1. Change Inv. Team', '2. Closure', '3. Underperformance','4. Not Disp. Opp.', '5. Others Opps.']
dataDF['predict'] = classes
dataDF['predict'] = [labels_l[i] for i in dataDF['predict']]

dataDF.to_excel('docs/test log/test_eval.xlsx',sheet_name='test data',index=False)




