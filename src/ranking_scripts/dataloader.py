import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import gmean

#imports for phrase level
import re
import numpy as np
from scipy.stats import gmean
import networkx as nx
import torch
from torch.nn.functional import cosine_similarity
import networkx as nx
from collections import defaultdict
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import logging


# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
processed_datasets_path = os.path.join(dir_path, "..", "..", "data", "processed_datasets")

class newReviewDataset(Dataset):
    def __init__(self, data_path, initial_train_size=1, return_embedding='specter', use_pseudo_for_scibert=False, start_idx=0,device='cpu',seed=42, shuffle = False):
        logging.info("Initializing newReviewDataset")
        self.data_path = data_path
        self.texts, self.labels = self._load_data()
        
        #shuffle data according to seed
        if shuffle:
            np.random.seed(seed)
            self.shuffle_indices = np.random.permutation(len(self.texts))
            self.texts = [self.texts[i] for i in self.shuffle_indices]
            self.labels = [self.labels[i] for i in self.shuffle_indices]
        
        self.return_embedding = return_embedding
        self.use_pseudo_for_scibert = use_pseudo_for_scibert
        self.device = device
        
        #print count of 0 and 1 labels sum(labels),len(labels)-sum(labels)
        print("====Data Summary====")
        print("0 = not relevant, 1 = relevant")
        print("Number of 0 labels:",len(self.labels)-sum(self.labels))
        print("Number of 1 labels:",sum(self.labels))
        
        print("Intializing embeddings")

        # Embedding initialization
        self.embeddings = self._initialize_embeddings()
        

        # Partition indices
        self.known_indices, self.unknown_indices = self._get_init_idx(self.labels)
        #self.known_indices = list(range(initial_train_size))
        #self.unknown_indices = list(range(initial_train_size, len(self.texts)))
        self.labels = np.array(self.labels)
        
        # Train and pseudo train data
        # Train embeddings: Use known_indices to slice self.embeddings
        self.train_embeddings = self.embeddings[self.known_indices]
        self.train_labels = self.labels[self.known_indices]
        self.pseudo_train_labels = np.array([])
        self.pseudo_train_embeddings = []
        
        # Unknown data
        self.unknown_embeddings = self.embeddings[self.unknown_indices]
        self.unknown_labels = self.labels[self.unknown_indices]
        
    def get_shuffle_indices(self):
        return self.shuffle_indices
        
    

    def _load_data(self):
        data = pd.read_csv(self.data_path)
        texts = data['title'] + ' ' + data['abstract']
        labels = data['label'].values
        return texts.tolist(), labels.tolist()
    
    def _get_init_idx(y):
        idx_unknown = list(range(len(y)))
        idx_known = []
        n_rel = 0
        n_nrel = 0
        i = 0
        idx = 0
        while n_rel + n_nrel < 10:

            if y[idx] == 1 and n_rel < 5:
                val = idx_unknown.pop(i)
                idx_known.append(val)
                n_rel += 1
                idx += 1
            elif y[idx] == 0 and n_nrel < 5:
                val = idx_unknown.pop(i)
                idx_known.append(val)
                n_nrel += 1
                idx += 1
            else:
                i += 1
                idx += 1
    
        return idx_known, idx_unknown

    def _initialize_embeddings(self):
        if self.return_embedding == 'specter':
            tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            model = AutoModel.from_pretrained('allenai/specter')
            model.to(self.device)
            return self._get_transformer_embeddings(self.texts, tokenizer, model)
        elif self.return_embedding == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=768)
            tfidf_matrix = vectorizer.fit_transform(self.texts).toarray()
            #return [torch.tensor(vec, dtype=torch.float32) for vec in tfidf_matrix]
            return torch.tensor(tfidf_matrix, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Invalid embedding type")

    def _get_transformer_embeddings(self, texts, tokenizer, model):
        print("Getting transformer embeddings")
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move all tensors to the specified device
        with torch.no_grad():
            result = model(**inputs)
        print("Got embeddings")
        return result.last_hidden_state[:, 0, :]

    def update_train_set(self, index):
        
        # print("Before update:")
        # print("known_indices:", self.known_indices)
        # print("unknown_indices:", self.unknown_indices)
        # print("train_embeddings shape:", self.train_embeddings.shape if self.create_tensors else len(self.train_embeddings))
        # print("unknown_embeddings shape:", self.unknown_embeddings.shape if self.create_tensors else len(self.unknown_embeddings))
        # print("train_labels shape:", self.train_labels.shape)
        # print("unknown_labels shape:", self.unknown_labels.shape)
        
        # Get the index from unknown_indices at position 'index'
        idx = self.unknown_indices.pop(index)
        self.known_indices.append(idx)
        
        self.train_embeddings = self.embeddings[self.known_indices]
        self.unknown_embeddings = self.embeddings[self.unknown_indices]
        
        self.train_labels = self.labels[self.known_indices]
        self.unknown_labels = self.labels[self.unknown_indices]
         # Debug prints after update
        # print("After update:")
        # print("known_indices:", self.known_indices)
        # print("unknown_indices:", self.unknown_indices)
        # print("train_embeddings shape:", self.train_embeddings.shape if self.create_tensors else len(self.train_embeddings))
        # print("unknown_embeddings shape:", self.unknown_embeddings.shape if self.create_tensors else len(self.unknown_embeddings))
        # print("train_labels shape:", self.train_labels.shape)
        # print("unknown_labels shape:", self.unknown_labels.shape)

    def add_pseudo_labels(self, pseudo_labels):
        self.pseudo_train_labels = torch.tensor(pseudo_labels, dtype=torch.float32, device=self.device)
        self.pseudo_train_embeddings = self.embeddings[self.unknown_indices[:len(pseudo_labels)]]

    def __len__(self):
        return len(self.train_embeddings) + (len(self.pseudo_train_embeddings) if self.use_pseudo_for_scibert else 0)

    def __getitem__(self, idx):
        if idx < len(self.train_embeddings):
            return self.train_embeddings[idx], self.train_labels[idx]
        pseudo_idx = idx - len(self.train_embeddings)
        return self.pseudo_train_embeddings[pseudo_idx], self.pseudo_train_labels[pseudo_idx]

    def get_unknown_data(self):
        return self.unknown_embeddings, self.unknown_labels

    def get_known_data(self):
        return self.train_embeddings, self.train_labels
    
