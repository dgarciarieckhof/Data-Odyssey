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
from datetime import datetime
from collections import Counter
from torch.optim import AdamW, Optimizer
from torch.nn.functional import cross_entropy
from torch.nn import DataParallel, Module, Softmax
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Any, Union
from src.utils.splitting import transform_list_of_texts
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from src.utils.bert import collate_fn_pooled_tokens, TokenizedDataset, error_metrics
from torch.utils.data import Dataset, WeightedRandomSampler, RandomSampler, SequentialSampler, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, BatchEncoding

# -----------------
# Load data
dataDF = pd.read_csv('data/processed/s1_traindata.csv')

# -----------------
# Get the text variables and their classifications
texts = dataDF['content'].to_list()
labels = dataDF['label_f'].apply(lambda x: int(int(x.split()[0].replace('.',''))-1)).to_list()
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# Load pre trained models
num_labels = len(np.unique(labels))
tokenizer_ = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
model_ = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain',num_labels=num_labels)

# -----------------
# Fit the neural network on a down-stream task
# Tokenized and pass to the dataloader
chunk_size = 510
stride = 510
min_chunk = 1
max_length = None
batch_size = 24
epochs = 10
device = 'cpu'
optimizer = AdamW(model_.parameters(), lr=5e-5, eps=1e-8)

# Tokenize train set
class_counts = Counter(y_train)
class_weights = {item: 1/count for item, count in class_counts.items()}
sample_weights = [class_weights[i] for i in y_train]
tokens = transform_list_of_texts(X_train,tokenizer_,chunk_size,stride,min_chunk,max_length)
dataset = TokenizedDataset(tokens, y_train)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn_pooled_tokens)

# Tokenize test set
test_tokens = transform_list_of_texts(X_test,tokenizer_,chunk_size,stride,min_chunk,max_length)
test_dataset = TokenizedDataset(test_tokens, y_test)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, collate_fn=collate_fn_pooled_tokens)

# Train the nn
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Training outputs
results = pd.DataFrame(columns=['epoch','train_loss','val_loss','train_ac','val_ac','train_f1','val_f1','train_bac','val_bac'])

model_.to(device)
progress_bar = tqdm(total=len(dataloader), desc=f'Epoch 0/{epochs}', unit='batch', leave=False, disable=False)

for epoch in range(epochs):
    model_.train()
    trn_loss, trn_ac, trn_f1, trn_bac, val_loss, val_ac, val_f1, val_bac = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    # ------
    # Init training
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        labels_train = batch[-1].float().cpu()
        # Get predictions
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        # Concatenate all input_ids into one batch
        input_ids_combined, attention_mask_combined = [], []        
        for x in input_ids:
            input_ids_combined.extend(x.tolist())
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())
        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in input_ids_combined])
        attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in attention_mask_combined])                
        # Get model predictions for the combined batch                
        preds = model_(input_ids_combined_tensors, attention_mask_combined_tensors)
        preds = preds[0].cpu()
        # Split result preds into chunks
        preds_split = preds.split(number_of_chunks)
        # Pooled the prediction usin mean for items within the same article
        pooled_preds = torch.stack([x.mean(dim=0).reshape(num_labels) if len(x) > 1 else x.reshape(num_labels) for x in preds_split])
        # Calculate error metrics
        loss = cross_entropy(pooled_preds, labels_train.long())
        probs = torch.nn.functional.softmax(pooled_preds,dim=-1)
        pred_classes = [np.argmax(prob.tolist()) for prob in probs]
        true_classes = labels_train.long().tolist()
        ac, f1score, balac  = error_metrics(true_classes, pred_classes)  
        # Optimize the model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)
        optimizer.step()
        # Update progress bar description
        trn_loss += loss.item()
        trn_ac += ac 
        trn_f1 += f1score
        trn_bac += balac
        # Print training metrics
        progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
        progress_bar.set_postfix(loss=trn_loss / (step + 1), f1_score=f1score, ac = ac, bal_ac=balac)
        progress_bar.update()
    # Average error metrics for each epoch
    averageT_loss = trn_loss / len(dataloader)
    averageT_ac = trn_ac / len(dataloader)    
    averageT_f1 = trn_f1 / len(dataloader)
    averageT_bac = trn_bac / len(dataloader)
    # Save model parameters
    torch.save(model_.state_dict(), f'models/finbert_lt_epoch_{epoch}.model')
    
    # ------
    # Validation error metrics
    model_.eval()
    for step_, batch_ in enumerate(test_dataloader):    
        labels_test = batch_[-1].float().cpu()
        # Get predictions
        input_ids = batch_[0]
        attention_mask = batch_[1]
        number_of_chunks = [len(x) for x in input_ids]
        # Concatenate all input_ids into one batch
        input_ids_combined, attention_mask_combined = [], []        
        for x in input_ids:
            input_ids_combined.extend(x.tolist())
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())
        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in input_ids_combined])
        attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in attention_mask_combined])
        # Get model predictions for the combined batch
        with torch.no_grad():
            preds = model_(input_ids_combined_tensors, attention_mask_combined_tensors)
            preds = preds[0].cpu()
            # split result preds into chunks
            preds_split = preds.split(number_of_chunks)
            # Pooled the prediction usin mean for items within the same article
            pooled_preds = torch.stack([x.mean(dim=0).reshape(num_labels) if len(x) > 1 else x.reshape(num_labels) for x in preds_split])
        # Error metrics
        loss = cross_entropy(pooled_preds, labels_test.long())
        probs = torch.nn.functional.softmax(pooled_preds,dim=-1)
        pred_classes = [np.argmax(prob.tolist()) for prob in probs]
        true_classes = labels_test.long().tolist()
        ac, f1score, balac  = error_metrics(true_classes, pred_classes)
        # Summary
        val_loss += loss.item() 
        val_ac += ac
        val_f1 += f1score
        val_bac += balac
    # Summary of each batch
    averageV_loss = val_loss / len(test_dataloader)
    averageV_ac = val_ac / len(test_dataloader)    
    averageV_f1 = val_f1 / len(test_dataloader)
    averageV_bac = val_bac / len(test_dataloader)
    # Store results
    epoch_res = {'epoch':epoch+1,'train_loss':averageT_loss,'val_loss':averageV_loss,'train_ac':averageT_ac,'val_ac':averageV_ac,'train_f1':averageT_f1,'val_f1':averageV_f1,'train_bac':averageT_bac,'val_bac':averageV_bac}
    results = pd.concat([results, pd.DataFrame(epoch_res, index=[0])], ignore_index=True)
    # Reset progress bar
    progress_bar.reset()

    # ------
    # Early stopping
    val = averageV_loss
    consecutive_count = 0
    previous_value = float('inf')
    if val >= previous_value:
        consecutive_count += 1
    else:
        consecutive_count = 0
    if consecutive_count >= 3:
        print("Break statement triggered: No decrease in loss for 3 consecutive epochs.")
        break
    previous_value = val    

# Close the progress bar
progress_bar.close()    

# Save results to dataframe
today = datetime.today()
formatted_date = today.strftime("%Y%m%d")
results.to_excel(f'docs/train log/{formatted_date}_training_log.xlsx')