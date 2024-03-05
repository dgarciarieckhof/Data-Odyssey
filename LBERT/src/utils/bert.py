from __future__ import annotations
import json
import torch
from pathlib import Path
from torch import Tensor
from abc import ABC, abstractmethod
from torch.optim import AdamW, Optimizer
from typing import List, Optional, Tuple, Any, Union
from torch.nn import BCELoss, DataParallel, Module, Linear, Sigmoid, Softmax
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertModel, PreTrainedTokenizerBase, RobertaModel

class TokenizedDataset(Dataset):
    """
    Dataset for tokens with optional labels.
    Args:
        tokens (BatchEncoding): The batch of input_ids and attention_mask tensors.
        labels (Optional[List]): Optional list of labels.
    Attributes:
        input_ids (Tensor): The input_ids tensor.
        attention_mask (Tensor): The attention_mask tensor.
        labels (Optional[List]): Optional list of labels.
    """
    def __init__(self, tokens: BatchEncoding, labels: Optional[List] = None):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = labels
    def __len__(self) -> int:
        return len(self.input_ids)
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Any], Tuple[Tensor, Tensor]]:
        if self.labels:
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx]
    
def collate_fn_pooled_tokens(data):
    """
    Dataset for tokens with optional labels.
    Args:
        data (List[Tuple[Tensor]]): The list of tuples containing input_ids and attention_mask tensors.
            Each tuple represents a single sample.
            - data[i][0]: input_ids (Tensor): The input_ids tensor for sample i.
            - data[i][1]: attention_mask (Tensor): The attention_mask tensor for sample i.
            - data[i][2]: labels (Optional): Optional labels tensor for sample i.

    Returns:
        List: The collated list containing input_ids, attention_mask, and labels (if available).
    """
    input_ids = [data[i][0] for i in range(len(data))]
    attention_mask = [data[i][1] for i in range(len(data))]
    if len(data[0]) == 2:
        collated = [input_ids, attention_mask]
    else:
        labels = Tensor([data[i][2] for i in range(len(data))])
        collated = [input_ids, attention_mask, labels]
    return collated    

def error_metrics(true_classes, pred_classes):
    erm1 = accuracy_score(true_classes, pred_classes)
    erm2 = f1_score(true_classes,pred_classes,average='weighted',zero_division=True)
    erm3 = balanced_accuracy_score(true_classes,pred_classes)
    return erm1, erm2, erm3 