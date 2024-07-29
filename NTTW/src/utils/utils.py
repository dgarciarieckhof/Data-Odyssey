import os
import re
import gc
import json
import spacy
import torch
import GPUtil
import warnings
import chromadb
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from typing import List
from datetime import datetime
from chromadb.config import Settings
from googleapiclient.discovery import build
from sentence_transformers.util import pytorch_cos_sim
from deepmultilingualpunctuation import PunctuationModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# -------------------
# interact with youtube videos
def get_video_metadata(video_id: str, api_key: str) -> dict:
    """
    Retrieves metadata for a YouTube video specified by video_id using the YouTube Data API.
    Args:
        video_id (str): The ID of the YouTube video.
        api_key (str): Your API key for accessing the YouTube Data API.
    Returns:
        dict: A dictionary containing the following video metadata:
            - 'channel': The title of the channel that uploaded the video.
            - 'published': The date string representing the video's publish date.
            - 'title': The title of the video.
            - 'description': The first line of the video's description, cleaned of extra spaces.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    channel = response['items'][0]['snippet']['channelTitle']
    published = response['items'][0]['snippet']['publishedAt']
    published = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').date().strftime('%Y-%m-%d')
    title = response['items'][0]['snippet']['title']
    description = response['items'][0]['snippet']['description']
    description = description.split('\n')[0].strip()
    description = re.sub(r'\s\s+', ' ', description)
    return {'channel': channel, 'published': published, 'title': title, 'description': description}

def get_video_transcripts(video_id: str, api_key: str, languages: List[str]) -> dict:
    """
    Retrieves video metadata and transcripts for a YouTube video.
    Args:
        video_id (str): The ID of the YouTube video.
        api_key (str): Your API key for accessing the YouTube Data API.
        languages (list[str]): A list of language codes for the transcripts to retrieve.
    Returns:
        dict: A dictionary containing video metadata and transcript data:
            - 'channel': The title of the channel that uploaded the video.
            - 'published': The date string representing the video's publish date.
            - 'title': The title of the video.
            - 'description': The cleaned first line of the video's description.
            - 'language': The language code of the retrieved transcript.
            - 'transcript': The transcript text.
    """
    video_data = get_video_metadata(video_id=video_id, api_key=api_key)
    transcripts_available = YouTubeTranscriptApi.list_transcripts(video_id)
    transcripts = transcripts_available.find_transcript(languages)
    video_data['language'] = transcripts.language
    video_data['transcript'] = transcripts.fetch()
    return video_data

def parse_transcript(transcript: dict) -> str:
    """
    Parses a transcript text to clean and normalize it.
    Args:
        transcript (dict): The raw transcript text.
    Returns:
        str: The cleaned and normalized transcript text.
    """
    text = ""
    for segment in transcript:
        text += ' ' + segment['text']
    text = text.strip()
    text = text.replace('[Music]', '')
    text = text.strip()
    text = re.sub(r'\s\s+', ' ', text)
    return text

def get_video_data(video_ids: List[str], api_key: str, languages: List[str] = ['en']) -> List[dict]:
    """
    Retrieves video data including metadata and transcripts for a list of video IDs.
    Args:
        video_ids (List[str]): List of YouTube video IDs.
        api_key (str): Your API key for accessing the YouTube Data API.
        languages (List[str]): List of language codes for the transcripts to retrieve. Default is ['en'].
    Returns:
        List[dict]: A list of dictionaries containing video metadata and transcripts.
    """
    video_data_list = []
    for video_id in tqdm(video_ids):
        video_data = get_video_transcripts(video_id=video_id, api_key=api_key, languages=languages)
        metadata = {k: v for k, v in video_data.items() if k != 'transcript'}
        transcript = parse_transcript(transcript=video_data['transcript'])
        video_data_list.append({'video_id': video_id, 'metadata': metadata, 'transcript': transcript})
    return video_data_list

# -------------------
# Vector DB
def setup_vector_db(model_name: str, path_name: str, collection_name: str, device: torch.device) -> (SentenceTransformer, int, object, object):
    """
    Sets up a vector database using a specified SentenceTransformer model.
    Args:
        model_name (str): The name of the SentenceTransformer model to use.
        path_name (str): The name of the folder where data will be stored.
        collection_name (str): The name of the collection.
        device (torch.device): The device to load the model on (CPU or GPU).
    Returns:
        Tuple[SentenceTransformer, int, object, object]: A tuple containing the model, 
        maximum sequence length, database collection, and database client.
    """
    model = SentenceTransformer(model_name, device=device)
    max_seq_length = model.max_seq_length - 2
    client = chromadb.PersistentClient(
        path=path_name,
        settings=Settings(allow_reset=True)
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=model_name),
        metadata={'hnsw:space': 'cosine'}
    )
    return model, max_seq_length, collection, client

def populate_database(collection, chunks: List[str], embeddings: List[List[float]], 
                      metadata: dict, video_id: str) -> None:
    """
    Populates a vector database with text chunks, embeddings, and metadata.
    Args:
        collection: The database collection to populate.
        chunks (List[str]): List of text chunks.
        embeddings (List[List[float]]): List of embeddings corresponding to the text chunks.
        metadata (dict): Metadata to associate with each chunk.
        video_id (str): The ID of the video being processed.
    Returns:
        None
    """
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_id"] = i
        collection.add(
            ids=[f"video_{video_id}_chunk_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[chunk_metadata]
        )


# -------------------
# Chunking and embeddings
def split_text(text: str, min_chunk_size: int, overlap: int) -> List[str]:
    """
    Split a string into chunks of words with at least a specified minimum size and overlap.

    Args:
        text (str): The input string to split.
        min_chunk_size (int): The minimum size of each chunk (in words).
        overlap (int): The number of words to overlap between consecutive chunks.

    Returns:
        List[str]: A list of chunks where each chunk is a string of words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + min_chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start + min_chunk_size > len(words):
            final_chunk = ' '.join(words[start:])
            if final_chunk not in chunks:
                chunks.append(final_chunk)
            break
    return chunks

def semantic_chunking(transcript: str, model: SentenceTransformer, nlp: spacy.language.Language, 
                      max_group: int = 5, sim_threshold: float = 0.25) -> List[str]:
    """
    Performs semantic chunking on a transcript using a SentenceTransformer model and spaCy NLP model.

    Args:
        transcript (str): The transcript text.
        model (SentenceTransformer): The SentenceTransformer model for encoding sentences.
        nlp (spacy.language.Language): The spaCy NLP model for sentence segmentation.
        max_group (int): Maximum number of sentences in a chunk. Default is 5.
        sim_threshold (float): Similarity threshold for grouping sentences. Default is 0.25.

    Returns:
        List[str]: A list of semantically chunked text segments.
    """
    restorer = PunctuationModel()
    transcript = restorer.restore_punctuation(transcript)
    doc = nlp(transcript)
    sentences = [str(sent).strip() for sent in doc.sents]
    print(f'{len(sentences)} sentences extracted from transcript')

    embeddings = model.encode(sentences)
    n = len(embeddings)
    similarities = [pytorch_cos_sim(embeddings[i-1], embeddings[i]).item() for i in range(1, n)]

    groups = [[sentences[0]]]
    for i in range(1, n):
        if len(groups[-1]) >= max_group:
            groups.append([sentences[i]])
        elif similarities[i-1] > sim_threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    chunks = [' '.join(g) for g in groups]
    return chunks

def process_chunks(chunks: List[str], max_seq_length: int, model: SentenceTransformer) -> (List[str], List[List[float]]):
    """
    Processes text chunks to ensure they fit within a specified maximum sequence length and generates embeddings.

    Args:
        chunks (List[str]): List of text chunks to process.
        max_seq_length (int): The maximum sequence length for each chunk.
        model (SentenceTransformer): The SentenceTransformer model for generating embeddings.

    Returns:
        Tuple[List[str], List[List[float]]]: A tuple containing the list of processed text chunks and their corresponding embeddings.
    """
    df = pd.DataFrame({'chunks': chunks})
    df['len'] = df['chunks'].str.len()
    df['flag'] = df['len'] > max_seq_length
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_seq_length, chunk_overlap=50)
    df['chunks_'] = df.apply(lambda row: text_splitter.split_text(row['chunks']) if row['flag'] else [row['chunks']], axis=1)
    df = df.explode('chunks_')

    embeddings = model.encode(df['chunks_'].values)
    embeddings = [embedding.tolist() for embedding in embeddings]
    temp = pd.DataFrame(embeddings)
    temp.index = df.index
    temp = temp.groupby(temp.index.get_level_values(0)).mean()
    embeddings = temp.values.tolist()

    return df['chunks_'].tolist(), embeddings

# -------------------
# gpu utilization
def get_device() -> torch.device:
    """
    Returns the appropriate device (GPU if available, otherwise CPU).
    Returns:
        torch.device: The device to use (CPU or GPU).
    """
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        print("GPU is not available, using CPU")
        return torch.device('cpu')

def print_gpu_stats() -> None:
    """
    Prints the statistics of all available GPUs.
    Returns:
        None
    """
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Load: {gpu.load * 100:.2f}%")
        print(f"GPU Free Memory: {gpu.memoryFree / 1024:.2f}GB")
        print(f"GPU Used Memory: {gpu.memoryUsed / 1024:.2f}GB")
        print(f"GPU Total Memory: {gpu.memoryTotal / 1024:.2f}GB")
        print(f"GPU Temperature: {gpu.temperature:.2f} °C")
        print(f"GPU UUID: {gpu.uuid}")

def clear_gpu(vars: List[object]) -> None:
    """
    Clears the GPU memory by deleting specified variables, emptying the cache, and running garbage collection.
    Args:
        vars (List[object]): List of variables to delete.
    Returns:
        None
    """
    for var in vars:
        del var
    torch.cuda.empty_cache()
    gc.collect()
