import torch
from torch import Tensor
from typing import List, Optional, Tuple, Any, Union
from transformers import BatchEncoding, PreTrainedTokenizerBase

# Functions for preparing input for longer texts - based on
# https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html

def tokenize_whole_text(text: str, tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
    """
    Tokenizes the entire text without truncation and without special tokens.
    Args:
        text (str): The text to tokenize.
        tokenizer (PreTrainedTokenizerBase): The tokenizer object.
    Returns:
        BatchEncoding: The batch encoding containing the tokens.
    """
    # Tokenize the text without truncation and special tokens
    tokens = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors="pt")
    return tokens

def tokenize_text_with_truncation(text: str, tokenizer: PreTrainedTokenizerBase, maximal_text_length: int) -> BatchEncoding:
    """
    Tokenizes the text with truncation to maximal_text_length and without special tokens.
    Args:
        text (str): The text to tokenize.
        tokenizer (PreTrainedTokenizerBase): The tokenizer object.
        maximal_text_length (int): The maximum length of the text after truncation.
    Returns:
        BatchEncoding: The batch encoding containing the tokens.
    """
    # Tokenize the text with truncation and without special tokens
    tokens = tokenizer(text, add_special_tokens=False, max_length=maximal_text_length, truncation=True, return_tensors="pt")
    return tokens

def check_split_parameters_consistency(chunk_size: int, stride: int, minimal_chunk_length: int) -> None:
    """
    Checks the consistency of split parameters.
    Args:
        chunk_size (int): The size of each chunk.
        stride (int): The stride value.
        minimal_chunk_length (int): The minimum length of each chunk.
    Raises:
        Exception: If the chunk_size is greater than 510.
        Exception: If the minimal_chunk_length is greater than chunk_size.
        Exception: If the stride is greater than chunk_size.
    """
    if chunk_size > 510:
        raise Exception("Size of each chunk cannot be bigger than 510!")
    if minimal_chunk_length > chunk_size:
        raise Exception("Minimal length cannot be bigger than size!")
    if stride > chunk_size:
        raise Exception("Stride cannot be bigger than size! Chunks must overlap or be near each other!")

def split_overlapping(tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int) -> list[Tensor]:
    """
    Helper function for dividing 1-dimensional tensors into overlapping chunks.
    Args:
        tensor (Tensor): The input tensor to split.
        chunk_size (int): The size of each chunk.
        stride (int): The stride value.
        minimal_chunk_length (int): The minimum length of each chunk.
    Returns:
        list[Tensor]: The list of overlapping chunks.
    """
    check_split_parameters_consistency(chunk_size, stride, minimal_chunk_length)
    # Split the tensor into overlapping chunks
    result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
    if len(result) > 1:
        # Ignore chunks with less than minimal_length number of tokens
        result = [x for x in result if len(x) >= minimal_chunk_length]
    return result

def split_tokens_into_smaller_chunks(tokens: BatchEncoding, chunk_size: int, stride: int, minimal_chunk_length: int) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Splits tokens into overlapping chunks with given size and stride.
    Args:
        tokens (BatchEncoding): The input tokens to split.
        chunk_size (int): The size of each chunk.
        stride (int): The stride value.
        minimal_chunk_length (int): The minimum length of each chunk.
    Returns:
        Tuple[List[Tensor], List[Tensor]]: The tuple containing input_id_chunks and mask_chunks.
    """
    input_id_chunks = split_overlapping(tokens["input_ids"][0], chunk_size, stride, minimal_chunk_length)
    mask_chunks = split_overlapping(tokens["attention_mask"][0], chunk_size, stride, minimal_chunk_length)
    return input_id_chunks, mask_chunks

def add_special_tokens_at_beginning_and_end(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> None:
    """
    Adds special CLS token (token id = 101) at the beginning.
    Adds SEP token (token id = 102) at the end of each chunk.
    Adds corresponding attention masks equal to 1 (attention mask is boolean).
    Args:
        input_id_chunks (List[Tensor]): List of input_id chunks.
        mask_chunks (List[Tensor]): List of attention mask chunks.
    Returns:
        None
    """
    for i in range(len(input_id_chunks)):
        # Adding CLS (token id 101) and SEP (token id 102) tokens
        input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
        # Adding attention masks corresponding to special tokens
        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])

def add_padding_tokens(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> None:
    """
    Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens.
    Args:
        input_id_chunks (List[Tensor]): List of input_id chunks.
        mask_chunks (List[Tensor]): List of attention mask chunks.
    Returns:
        None
    """
    for i in range(len(input_id_chunks)):
        # Get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # Check if tensor length satisfies required chunk size
        if pad_len > 0:
            # If padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])

def stack_tokens_from_all_chunks(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Reshapes data to a form compatible with BERT model input.
    Args:
        input_id_chunks (List[Tensor]): List of input_id chunks.
        mask_chunks (List[Tensor]): List of attention mask chunks.
    Returns:
        Tuple[Tensor, Tensor]: The stacked input_ids and attention_mask tensors.
    """
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)
    return input_ids.long(), attention_mask.int()

def transform_single_text(text: str, tokenizer: PreTrainedTokenizerBase, chunk_size: int, stride: int, minimal_chunk_length: int, maximal_text_length: Optional[int]) -> Tuple[Tensor, Tensor]:
    """
    Transforms the entire text to model input of BERT model.
    Args:
        text (str): The input text to transform.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        chunk_size (int): The size of each chunk.
        stride (int): The stride for overlapping chunks.
        minimal_chunk_length (int): The minimal length of a chunk.
        maximal_text_length (Optional[int]): The maximal length of the text. If provided, the text will be truncated.
    Returns:
        Tuple[Tensor, Tensor]: The input_ids and attention_mask tensors.
    """
    if maximal_text_length:
        tokens = tokenize_text_with_truncation(text, tokenizer, maximal_text_length)
    else:
        tokens = tokenize_whole_text(text, tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens, chunk_size, stride, minimal_chunk_length)
    add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
    return input_ids, attention_mask

def transform_list_of_texts(texts: List[str], tokenizer: PreTrainedTokenizerBase, chunk_size: int, stride: int, minimal_chunk_length: int, maximal_text_length: Optional[int] = None) -> BatchEncoding:
    """
    Transforms a list of texts to model inputs of BERT model.
    Args:
        texts (List[str]): The list of input texts to transform.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        chunk_size (int): The size of each chunk.
        stride (int): The stride for overlapping chunks.
        minimal_chunk_length (int): The minimal length of a chunk.
        maximal_text_length (Optional[int]): The maximal length of the text. If provided, the text will be truncated.
    Returns:
        BatchEncoding: The batch of input_ids and attention_mask tensors.
    """
    model_inputs = [transform_single_text(text, tokenizer, chunk_size, stride, minimal_chunk_length, maximal_text_length) for text in texts]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return BatchEncoding(tokens)