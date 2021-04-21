"""General utilities for training.

Author:
    Shrey Desai
"""

import os
import json
import gzip
import pickle

import torch
from tqdm import tqdm

import spacy

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
nlp_large = spacy.load("en_core_web_lg")

def cuda(args, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).

    Args:
        tensor: PyTorch tensor.

    Returns:
        Tensor on CUDA device.
    """
    if args.use_gpu and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def unpack(tensor):
    """
    Unpacks tensor into Python list.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Python list with tensor contents.
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()


def load_dataset(path):
    """
    Loads MRQA-formatted dataset from path.

    Args:
        path: Dataset path, e.g. "datasets/squad_train.jsonl.gz"

    Returns:
        Dataset metadata and samples.
    """
    with gzip.open(path, 'rb') as f:
        elems = [
            json.loads(l.rstrip())
            for l in tqdm(f, desc=f'loading \'{path}\'', leave=False)
        ]
    meta, samples = elems[0], elems[1:]
    return (meta, samples)


def load_cached_embeddings(path):
    """
    Loads embedding from pickle cache, if it exists, otherwise embeddings
    are loaded into memory and cached for future accesses.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    bare_path = os.path.splitext(path)[0]
    cached_path = f'{bare_path}.pkl'
    if os.path.exists(cached_path):
        return pickle.load(open(cached_path, 'rb'))
    embedding_map = load_embeddings(path)
    pickle.dump(embedding_map, open(cached_path, 'wb'))
    return embedding_map


def load_embeddings(path):
    """
    Loads GloVe-style embeddings into memory. This is *extremely slow* if used
    standalone -- `load_cached_embeddings` is almost always preferable.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    embedding_map = {}
    with open(path) as f:
        next(f)  # Skip header.
        for line in f:
            try:
                pieces = line.rstrip().split()
                embedding_map[pieces[0]] = [float(weight) for weight in pieces[1:]]
            except:
                pass
    return embedding_map

def search_span_endpoints(start_probs, end_probs, passage, question, window=15):
    """
    Finds an optimal answer span given start and end probabilities.
    Specifically, this algorithm finds the optimal start probability p_s, then
    searches for the end probability p_e such that p_s * p_e (joint probability
    of the answer span) is maximized. Finally, the search is locally constrained
    to tokens lying `window` away from the optimal starting point.

    Args:
        start_probs: Distribution over start positions.
        end_probs: Distribution over end positions.
        window: Specifies a context sizefrom which the optimal span endpoint
            is chosen from. This hyperparameter follows directly from the
            DrQA paper (https://arxiv.org/abs/1704.00051).

    Returns:
        Optimal starting and ending indices for the answer span. Note that the
        chosen end index is *inclusive*.
    """
    # for idx in range(len(passage)):
    #     print(idx, passage[idx])
    passage_ents = nlp_large(' '.join(passage)).ents
    passage_text = []
    for x in passage_ents:
        words = x.text.split()
        for w in words:
            if w not in stop_words:
                passage_text.append(w)
    # print('passage', passage_text)
    question_text = []
    for w in question:
        if w not in stop_words and w not in string.punctuation:
            question_text.append(w)
    # print('question ->', question_text)
    begin = -1
    end = len(passage)-1
    # for ent in passage_text:
    #     if ent in question_text:
    #         print(ent)
    #         if begin == -1:
    #             begin = passage.index(ent)
    #         else:
    #             end = passage[begin:].index(ent)
    # if end > begin + window and begin != -1:
    #     begin = max(0, begin-5)
    #     end = min(len(passage)-1, end+5)
    #     start_probs = start_probs[begin:end+1]
    #     end_probs = end_probs[begin:end+1]
    #     print(begin)
    #     print(end)
    begin = -1
    end = -1
    for idx in range(len(passage)):
        if passage[idx] in passage_text and passage[idx] in question_text:
            begin = idx
            break
    for idx in range(len(passage)-1, -1, -1):
        if passage[idx] in passage_text and passage[idx] in question_text:
            end = idx
            break
    if end > begin + window and begin != -1 and end != -1:
        begin = max(0, begin-20)
        end = min(len(passage)-1, end+20)
        start_probs = start_probs[begin:end+1]
        end_probs = end_probs[begin:end+1]
        print(begin, end)

    max_start_index = start_probs.index(max(start_probs))
    max_end_index = -1
    max_joint_prob = 0.

    for end_index in range(len(end_probs)):
        if max_start_index <= end_index <= max_start_index + window:
            joint_prob = start_probs[max_start_index] * end_probs[end_index]
            if joint_prob > max_joint_prob:
                max_joint_prob = joint_prob
                max_end_index = end_index

    # if end > begin + window and begin != -1:
    #     max_start_index += begin
    #     max_end_index += begin

    if end > begin + window and begin != -1 and end != -1:
        max_start_index += begin
        max_end_index += begin

    return (max_start_index, max_end_index)