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

# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
processed_datasets_path = os.path.join(dir_path, "..", "..", "data", "processed_datasets")

class ReviewDataset(Dataset):
    def __init__(self, data_path, initial_train_size=1, return_embedding='specter',use_pseudo_for_scibert = False,start_idx=0):
        self.data_path = data_path
        self.texts, self.labels = self._load_data()
        self.return_embedding = return_embedding

        # Initialize embeddings based on return_embedding
        if self.return_embedding == 'specter' or self.return_embedding == 'both':
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            self.model = AutoModel.from_pretrained('allenai/specter')
            self.embeddings = self._get_specter_embeddings(self.texts)

        if self.return_embedding == 'tfidf':
            self.embedding_tfidf = self._get_tfidf_embeddings(self.texts)

        if self.return_embedding == 'scibert' or self.return_embedding == 'both':
            self.autophrase_file = os.path.join(processed_datasets_path, "example_data_embedded_for_scibert.txt")
            self.autophrase_data = self._read_autophrase_output_for_scibert()
            self.scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            self.embeddings_scibert = self._create_scibert_embeddings(len(self.texts))

        self.known_indices = list(range(initial_train_size))
        self.unknown_indices = list(range(initial_train_size, len(self.texts)))
        
        self.labels = np.array(self.labels)

        # Initialize train and unknown embeddings based on return_embedding
        if self.return_embedding == 'specter':
            self.train_embeddings = self.embeddings[self.known_indices]
            self.unknown_embeddings = self.embeddings[self.unknown_indices]
        elif self.return_embedding == 'tfidf':
            self.train_embeddings_tfidf = [self.embedding_tfidf[i] for i in self.known_indices]
            self.unknown_embeddings_tfidf = [self.embedding_tfidf[i] for i in self.unknown_indices]
        elif self.return_embedding == 'scibert':
            self.train_embeddings_scibert = [self.embeddings_scibert[i] for i in self.known_indices]
            self.unknown_embeddings_scibert = [self.embeddings_scibert[i] for i in self.unknown_indices]
        elif self.return_embedding == 'both':
            self.train_embeddings = self.embeddings[self.known_indices]
            self.train_embeddings_scibert = [self.embeddings_scibert[i] for i in self.known_indices]
            self.unknown_embeddings = self.embeddings[self.unknown_indices]
            self.unknown_embeddings_scibert = [self.embeddings_scibert[i] for i in self.unknown_indices]
        
        self.train_labels = self.labels[self.known_indices]
        self.unknown_labels = self.labels[self.unknown_indices]
        
        self.current_idx = start_idx
        
        self.pseudo_train_labels = np.array([])
        self.pseudo_train_embeddings_scibert = []
        
        self.use_pseudo_for_scibert = use_pseudo_for_scibert

        
        print(self.known_indices)
        print(self.unknown_indices)
        

    def _load_data(self):
        data = pd.read_csv(self.data_path)
        texts = data['title'] + ' ' + data['abstract']
        labels = data['label'].values
        return texts.tolist(), labels.tolist()

    def _get_specter_embeddings(self, texts):
         inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt",max_length=512)
         with torch.no_grad():
              result = self.model(**inputs)
         embedding = result.last_hidden_state[:, 0, :]
         return embedding

    def _get_tfidf_embeddings(self, texts):
        vectorizer = TfidfVectorizer(max_features=768)
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        return [torch.tensor(vec, dtype=torch.float32) for vec in tfidf_matrix]
    
    def _read_autophrase_output_for_scibert(self):
        """
        Reads each line of the AutoPhrase segmentation output (one doc per line),
        extracts every phrase enclosed in <phrase_Q=...>...</phrase>.
        Returns a list of lists: phrases_per_doc[i] is all phrases from doc i.
        """
        pattern = re.compile(r'<phrase_Q=[^>]*>([^<]+)</phrase>')
        phrases_per_doc = []
        
        with open(self.autophrase_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    phrases_per_doc.append([])
                    continue
                
                # Extract all phrases from the document
                matches = pattern.findall(line)
                phrases_per_doc.append(matches)
        
        return phrases_per_doc
    
    def _create_scibert_embeddings(self, num_texts):
        """
        Creates phrase-level SciBERT embeddings following the paper's methodology
        while maintaining compatibility with the existing codebase structure.
        Returns a list of torch.Tensor with shape [768] for each document.
        """
        all_phrases_per_doc = self._read_autophrase_output_for_scibert()
        scibert_embeddings = []

        for i in range(num_texts):
            phrases = all_phrases_per_doc[i]
            doc_text = self.texts[i]
            
            if not phrases:
                scibert_embeddings.append(torch.zeros(768))  # Match original size
                continue
                
            phrase_embeddings = []
            for phrase in phrases:
                # Content feature (x_l_p)
                content_inputs = self.scibert_tokenizer(
                    phrase,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    content_outputs = self.scibert_model(**content_inputs)
                content_emb = content_outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Average token embeddings

                # Context feature (y_l_p)
                masked_text = doc_text.replace(phrase, self.scibert_tokenizer.mask_token)
                context_inputs = self.scibert_tokenizer(
                    masked_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    context_outputs = self.scibert_model(**context_inputs)
                
                # Get [MASK] token embedding
                mask_positions = (context_inputs["input_ids"] == self.scibert_tokenizer.mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) > 0:
                    context_emb = context_outputs.last_hidden_state[0, mask_positions[1][0]]
                else:
                    context_emb = context_outputs.last_hidden_state.mean(dim=1).squeeze(0)

                # Final phrase embedding: average of content and context
                phrase_emb = (content_emb + context_emb) / 2
                phrase_embeddings.append(phrase_emb)

            # Average all phrase embeddings for the document
            if phrase_embeddings:
                doc_embedding = torch.stack(phrase_embeddings).mean(dim=0)
                scibert_embeddings.append(doc_embedding)
            else:
                scibert_embeddings.append(torch.zeros(768))

        return scibert_embeddings
    

    def update_train_set(self, index):
        """
        Adds the selected document (using the index from unknown documents) to the
        training set, moves its index from unknown to known
        """
        
        # extract paper from unknown
        new_train_embedding = self.unknown_embeddings[index]
        new_train_label = self.unknown_labels[index]
        new_train_embedding_scibert = self.unknown_embeddings_scibert[index]
        new_train_embedding_tfidf = self.unknown_embeddings_tfidf[index]
        
        #add paper to train
        if len(self.train_embeddings.shape) == 1:
             self.train_embeddings = new_train_embedding.unsqueeze(0)
        else:
            self.train_embeddings = torch.cat((self.train_embeddings, new_train_embedding.unsqueeze(0)), dim=0)
        self.train_labels = np.append(self.train_labels, new_train_label)
        self.train_embeddings_scibert.append(new_train_embedding_scibert)
        self.train_embeddings_tfidf.append(new_train_embedding_tfidf)
        
        #update known and unknown indices
        
        selected_index = self.unknown_indices[index]
        
        self.known_indices.append(selected_index)
        
        del self.unknown_indices[index]
        
        #update embeddings
        self.unknown_embeddings = self.embeddings[self.unknown_indices]
        self.unknown_labels = np.array(self.labels)[self.unknown_indices]
        self.unknown_embeddings_scibert = [self.embeddings_scibert[i] for i in self.unknown_indices]
        self.unknown_embeddings_tfidf = [self.embedding_tfidf[i] for i in self.unknown_indices]
        
    
    def add_pseudo_labels(self, pseudo_labels):
        """
        Adds the selected document (using the index from unknown documents) to the
        pseudo training set
        """
        
        #Add all papers to the pseudo train set
        self.pseudo_train_labels = np.array([])
        self.pseudo_train_embeddings_scibert = []
        for i,label in enumerate(pseudo_labels):
            
            new_pseudo_train_label = label
            new_pseudo_train_embedding_scibert = self.unknown_embeddings_scibert[i]
            self.pseudo_train_labels = np.append(self.pseudo_train_labels, new_pseudo_train_label)
            self.pseudo_train_embeddings_scibert.append(new_pseudo_train_embedding_scibert)
    
    def __len__(self):
        if self.return_embedding == 'scibert' and self.use_pseudo_for_scibert:
            return len(self.train_embeddings) + len(self.pseudo_train_embeddings)
        else:
            return len(self.train_embeddings)

    def __getitem__(self, idx):
        if self.return_embedding == 'specter':
           return self.train_embeddings[idx], self.train_labels[idx]
        elif self.return_embedding == 'scibert':
            if self.use_pseudo_for_scibert:
                if idx < len(self.train_embeddings_scibert):
                    return self.train_embeddings_scibert[idx], self.train_labels[idx]
                else:
                    pseudo_idx = idx - len(self.train_embeddings_scibert)
                    return self.pseudo_train_embeddings_scibert[pseudo_idx], self.pseudo_train_labels[pseudo_idx]
            else:
                return self.train_embeddings_scibert[idx], self.train_labels[idx]
        elif self.return_embedding == 'both':
            return self.train_embeddings[idx], self.train_embeddings_scibert[idx], self.train_labels[idx]
        elif self.return_embedding == 'tfidf':
            return self.train_embeddings_tfidf[idx], self.train_labels[idx]
        
    def get_unknown_data(self):
        if self.return_embedding == 'specter':
             return self.unknown_embeddings, self.unknown_labels
        elif self.return_embedding == 'scibert':
            return self.unknown_embeddings_scibert, self.unknown_labels
        elif self.return_embedding == 'both':
             return self.unknown_embeddings, self.unknown_embeddings_scibert, self.unknown_labels
        elif self.return_embedding == 'tfidf':
            return self.unknown_embeddings_tfidf, self.unknown_labels
    
    def get_known_data(self):
        if self.return_embedding == 'specter':
             return self.train_embeddings, self.train_labels
        elif self.return_embedding == 'scibert':
            return self.train_embeddings_scibert, self.train_labels
        elif self.return_embedding == 'both':
             return self.train_embeddings, self.train_embeddings_scibert, self.train_labels
        elif self.return_embedding == 'tfidf':
            return self.train_embeddings_tfidf, self.train_labels




class newReviewDataset(Dataset):
    def __init__(self, data_path, initial_train_size=1, return_embedding='specter', use_pseudo_for_scibert=False, start_idx=0,device='cpu'):
        self.data_path = data_path
        self.texts, self.labels = self._load_data()
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
        
        # Create semantic phrase graph, scibert embeddings and select relevant phrases
        self.phrase_level_input = self._initialize_phrase_level()

        # Partition indices
        self.known_indices = list(range(initial_train_size))
        self.unknown_indices = list(range(initial_train_size, len(self.texts)))
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

        
    def _load_data(self):
        data = pd.read_csv(self.data_path)
        texts = data['title'] + ' ' + data['abstract']
        labels = data['label'].values
        return texts.tolist(), labels.tolist()

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
            raise ValueError("Invalid embedding type. Choose 'specter', 'tfidf', or 'scibert'.")

    def _get_transformer_embeddings(self, texts, tokenizer, model):
        print("Getting transformer embeddings")
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move all tensors to the specified device
        with torch.no_grad():
            result = model(**inputs)
        print("Got embeddings")
        return result.last_hidden_state[:, 0, :]

    def _initialize_phrase_level(self):
        if self.return_embedding == 'scibert':
            self.autophrase_file = os.path.join(processed_datasets_path, "example_data_embedded_for_scibert.txt")
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            model.to(self.device)
            self.phrase_graph, self.phrase_embeddings, self.selected_phrases = self.create_phrase_similarity_graph(
                self.texts, 
                self.labels, 
                len(self.texts), 
                tokenizer, 
                model
            )
            return [self.phrase_graph, self.phrase_embeddings, self.selected_phrases]

    # PHRASE LEVEL START
    def _read_autophrase_output_for_scibert(self):
        pattern = re.compile(r'<phrase_Q=[^>]*>([^<]+)</phrase>')
        phrases_per_doc = []
        with open(self.autophrase_file, "r", encoding="utf-8") as f:
            for line in f:
                matches = pattern.findall(line.strip())
                phrases_per_doc.append(matches if matches else [])
        return phrases_per_doc
    
    def select_relevant_phrases(texts, labels, selection_percentage=0.30):
        """
        Select relevant phrases based on indicative and unusual scores.
        
        Args:
            texts: List of texts with AutoPhrase markup
            labels: List of binary labels
            selection_percentage: Percentage of top phrases to select
        
        Returns:
            set: Selected phrases
        """
        def extract_phrases_from_autophrase(text):
            pattern = re.compile(r'<phrase_Q=([^>]*)>([^<]+)</phrase>')
            matches = pattern.findall(text)
            return [(phrase, float(quality)) for quality, phrase in matches]
        
        def calculate_indicative_score(phrase, documents, labels):
            """Calculate ID(p) = np,1 / |{Dl ∩ R}|"""
            relevant_docs = [doc for doc, label in zip(documents, labels) if label == 1]
            np1 = sum(1 for doc in relevant_docs if phrase in doc)
            total_relevant = len(relevant_docs)
            return np1 / total_relevant if total_relevant > 0 else 0
        
        def calculate_unusual_score(phrase, documents):
            """Calculate UN(p) = log(|Dl| / |np,1 ∪ np,0|)"""
            epsilon = 1e-10
            total_docs = len(documents)
            docs_with_phrase = sum(1 for doc in documents if phrase in doc) + epsilon
            return np.log(total_docs / docs_with_phrase)
        
        # Extract all phrases from the corpus
        all_phrases = set()
        for text in texts:
            phrases = extract_phrases_from_autophrase(text)
            all_phrases.update(phrase for phrase, quality in phrases if quality >= 0.0)
        
        # Calculate scores for each phrase
        epsilon = 1e-10
        phrase_scores = {}
        for phrase in all_phrases:
            id_score = calculate_indicative_score(phrase, texts, labels) + epsilon
            un_score = calculate_unusual_score(phrase, texts) + epsilon
            if id_score > 0 and un_score > 0:
                combined_score = gmean([id_score, un_score])
                phrase_scores[phrase] = combined_score
        
        # Select top phrases
        sorted_phrases = sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)
        top_n = int(len(sorted_phrases) * selection_percentage)
        selected_phrases = {phrase for phrase, score in sorted_phrases[:top_n]}
        
        return selected_phrases

    def create_selected_phrase_embeddings(texts, num_texts, tokenizer, model, selected_phrases):
        """
        Create embeddings only for the selected phrases.
        """
        phrase_embeddings = {}
        
        # Collect all mentions of selected phrases
        phrase_mentions = {phrase: [] for phrase in selected_phrases}
        
        for doc_idx in range(num_texts):
            doc_text = texts[doc_idx]
            sentences = doc_text.split('.')
            
            for phrase in selected_phrases:
                for sentence in sentences:
                    if phrase in sentence:
                        phrase_mentions[phrase].append((doc_idx, sentence.strip()))
        
        # Create embeddings for selected phrases
        for phrase in selected_phrases:
            mentions = phrase_mentions[phrase]
            if not mentions:
                continue
                
            mention_embeddings = []
            for _, sentence in mentions:
                # Content feature (x_l_p)
                content_inputs = tokenizer(
                    sentence,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    content_outputs = model(**content_inputs)
                
                input_ids = content_inputs["input_ids"][0]
                phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
                phrase_positions = []
                
                for i in range(len(input_ids) - len(phrase_tokens) + 1):
                    if input_ids[i:i+len(phrase_tokens)].tolist() == phrase_tokens:
                        phrase_positions.extend(range(i, i + len(phrase_tokens)))
                
                if phrase_positions:
                    content_emb = content_outputs.last_hidden_state[0, phrase_positions].mean(dim=0)
                else:
                    content_emb = content_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                # Context feature (y_l_p)
                masked_text = sentence.replace(phrase, tokenizer.mask_token)
                context_inputs = tokenizer(
                    masked_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    context_outputs = model(**context_inputs)
                
                mask_positions = (context_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) > 0:
                    context_emb = context_outputs.last_hidden_state[0, mask_positions[1][0]]
                else:
                    context_emb = context_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                mention_emb = torch.cat([content_emb, context_emb], dim=0)
                mention_embeddings.append(mention_emb)
            
            if mention_embeddings:
                phrase_embeddings[phrase] = torch.stack(mention_embeddings).mean(dim=0)
        
        return phrase_embeddings

    def construct_phrase_graph(phrase_embeddings):
        """
        Construct graph only for selected phrases.
        """
        G = nx.Graph()
        phrases = list(phrase_embeddings.keys())
        G.add_nodes_from(phrases)
        
        for i, phrase_i in enumerate(phrases):
            emb_i = phrase_embeddings[phrase_i]
            
            for j, phrase_j in enumerate(phrases[i:], start=i):
                emb_j = phrase_embeddings[phrase_j]
                
                similarity = cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                weight = torch.sqrt(torch.maximum(torch.tensor(similarity), torch.tensor(0.0))).item()
                
                if weight > 0:
                    G.add_edge(phrase_i, phrase_j, weight=weight)
        
        return G

    def create_phrase_similarity_graph(self, texts, labels, num_texts, tokenizer, model, selection_percentage=0.30):
        """
        Main function to create phrase similarity graph:
        1. Select relevant phrases
        2. Create embeddings for selected phrases
        3. Build graph with similarity edges
        """
        # Select relevant phrases
        selected_phrases = self.select_relevant_phrases(texts, labels, selection_percentage)
        
        # Create embeddings only for selected phrases
        phrase_embeddings = self.create_selected_phrase_embeddings(
            texts, num_texts, tokenizer, model, selected_phrases
        )
        
        # Build graph
        phrase_graph = self.construct_phrase_graph(phrase_embeddings)
        
        return phrase_graph, phrase_embeddings, selected_phrases


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

# <-----------------------------------------------PHRASE LEVEL RANKING----------------------------------------------------->
import torch
import networkx as nx
from scipy.spatial.distance import cosine
from math import sqrt
from itertools import combinations
import numpy as np
import re

class PhraseFeatureSelector:
    def __init__(self, selection_percentage=0.30):
        self.selection_percentage = selection_percentage
        self.selected_phrases = None
        
    def calculate_indicative_score(self, phrase, documents, labels):
        """
        Calculate ID(p) = np,1 / |{Dl ∩ R}|
        """
        relevant_docs = [doc for doc, label in zip(documents, labels) if label == 1]
        np1 = sum(1 for doc in relevant_docs if phrase in doc)
        total_relevant = len(relevant_docs)
        return np1 / total_relevant if total_relevant > 0 else 0
    
    def calculate_unusual_score(self, phrase, documents):
        """
        Calculate UN(p) = log(|Dl| / |np,1 ∪ np,0|)
        """
        epsilon = 1e-10  # Small value to prevent division by zero
        total_docs = len(documents)
        docs_with_phrase = sum(1 for doc in documents if phrase in doc) + epsilon
        return np.log(total_docs / (docs_with_phrase))
    
    def extract_phrases_from_autophrase(self, text):
        """Extract phrases and their quality scores from AutoPhrase output"""
        pattern = re.compile(r'<phrase_Q=([^>]*)>([^<]+)</phrase>')
        matches = pattern.findall(text)
        return [(phrase, float(quality)) for quality, phrase in matches]
    
    def select_features(self, texts, labels):
        """
        Select top phrases based on the geometric mean of indicative and unusual scores.
        """
        epsilon = 1e-10  # Small value to prevent division by zero

        # Extract all phrases from the corpus
        all_phrases = set()
        for text in texts:
            phrases = self.extract_phrases_from_autophrase(text)
            all_phrases.update(phrase for phrase, quality in phrases if quality >= 0.0)
        
        # Calculate scores for each phrase
        phrase_scores = {}
        for phrase in all_phrases:
            id_score = self.calculate_indicative_score(phrase, texts, labels) + epsilon
            un_score = self.calculate_unusual_score(phrase, texts) + epsilon
            if id_score > 0 and un_score > 0:  # Ensure positive scores for geometric mean
                combined_score = gmean([id_score, un_score])
                phrase_scores[phrase] = combined_score
        
        # Rank phrases by combined score and select top 30%
        sorted_phrases = sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)
        top_n = int(len(sorted_phrases) * self.selection_percentage)
        self.selected_phrases = {phrase for phrase, score in sorted_phrases[:top_n]}
        
        return self.selected_phrases

class PhraseGraphConstructor:
    def __init__(self, selected_phrases, embeddings):
        """
        Initializes the graph constructor.

        Args:
            selected_phrases (set): Set of selected phrases.
            embeddings (torch.Tensor): Tensor of shape (num_phrases, 768) containing SciBERT embeddings.
        """
        self.selected_phrases = list(selected_phrases)
        self.embeddings = embeddings
        assert len(self.selected_phrases) == self.embeddings.size(0), "Mismatch between phrases and embeddings"

    def cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.

        Args:
            vec1 (torch.Tensor): First vector.
            vec2 (torch.Tensor): Second vector.

        Returns:
            float: Cosine similarity.
        """
        cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return cos_sim

    def construct_graph(self):
        """
        Constructs the phrase graph based on semantic similarity.

        Returns:
            networkx.Graph: The constructed graph with phrases as nodes and weighted edges.
        """
        G = nx.Graph()
        num_phrases = len(self.selected_phrases)

        # Add all phrases as nodes
        for phrase in self.selected_phrases:
            G.add_node(phrase)

        # Compute pairwise similarities and add edges
        for i, j in combinations(range(num_phrases), 2):
            vec_i = self.embeddings[i]
            vec_j = self.embeddings[j]
            cos_sim = self.cosine_similarity(vec_i, vec_j)
            w_ij = sqrt(max(cos_sim, 0))
            if w_ij > 0:
                G.add_edge(self.selected_phrases[i], self.selected_phrases[j], weight=w_ij)

        return G

# Assuming MainClass has methods to get selected phrases and embeddings
class MainClass:
    def __init__(self, autophrase_file, texts):
        self.autophrase_file = autophrase_file
        self.texts = texts
        # Initialize other necessary components here

    def _read_autophrase_output_for_scibert(self):
        pattern = re.compile(r'<phrase_Q=[^>]*>([^<]+)</phrase>')
        phrases_per_doc = []
        with open(self.autophrase_file, "r", encoding="utf-8") as f:
            for line in f:
                matches = pattern.findall(line.strip())
                phrases_per_doc.append(matches if matches else [])
        return phrases_per_doc
    
    def _create_scibert_embeddings(self, num_texts, tokenizer, model):
        all_phrases_per_doc = self._read_autophrase_output_for_scibert()
        scibert_embeddings = []
        
        for i in range(num_texts):
            phrases = all_phrases_per_doc[i]
            doc_text = self.texts[i]
            
            if not phrases:
                scibert_embeddings.append(torch.zeros(768))
                continue
                
            phrase_embeddings = []
            for phrase in phrases:
                # Content feature (x_l_p)
                content_inputs = tokenizer(
                    phrase,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    content_outputs = model(**content_inputs)
                content_emb = content_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                # Context feature (y_l_p)
                masked_text = doc_text.replace(phrase, tokenizer.mask_token)
                context_inputs = tokenizer(
                    masked_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    context_outputs = model(**context_inputs)
                    
                # Get [MASK] token embedding
                mask_positions = (context_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) > 0:
                    context_emb = context_outputs.last_hidden_state[0, mask_positions[1][0]]
                else:
                    context_emb = context_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                # Final phrase embedding: average of content and context
                phrase_emb = (content_emb + context_emb) / 2
                phrase_embeddings.append(phrase_emb)
                
            # Average all phrase embeddings for the document
            if phrase_embeddings:
                doc_embedding = torch.stack(phrase_embeddings).mean(dim=0)
                scibert_embeddings.append(doc_embedding)
            else:
                scibert_embeddings.append(torch.zeros(768))
            
        return torch.stack(scibert_embeddings)

    def select_phrases(self, texts, labels):
        selector = PhraseFeatureSelector()
        selected_phrases = selector.select_features(texts, labels)
        return selected_phrases

    def construct_phrase_graph(self, selected_phrases, embeddings):
        graph_constructor = PhraseGraphConstructor(selected_phrases, embeddings)
        graph = graph_constructor.construct_graph()
        return graph

    def process(self, tokenizer, model, labels):
        num_texts = len(self.texts)
        embeddings = self._create_scibert_embeddings(num_texts, tokenizer, model)
        selector = PhraseFeatureSelector()
        selected_phrases = selector.select_features(self.texts, labels)
        
        # Filter embeddings to only include selected phrases
        selected_indices = [i for i, phrase in enumerate(selector.selected_phrases)]
        selected_embeddings = embeddings[selected_indices]

        # Construct the graph
        graph = self.construct_phrase_graph(selected_phrases, selected_embeddings)
        
        return graph




    def _create_scibert_embeddings(self, num_texts, tokenizer, model):
        all_phrases_per_doc = self._read_autophrase_output_for_scibert()
        scibert_embeddings = []
        
        for i in range(num_texts):
            phrases = all_phrases_per_doc[i]
            doc_text = self.texts[i]
            
            if not phrases:
                scibert_embeddings.append(torch.zeros(768))
                continue
                
            phrase_embeddings = []
            for phrase in phrases:
                # Content feature (x_l_p)
                content_inputs = tokenizer(
                    phrase,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    content_outputs = model(**content_inputs)
                content_emb = content_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                # Context feature (y_l_p)
                masked_text = doc_text.replace(phrase, tokenizer.mask_token)
                context_inputs = tokenizer(
                    masked_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    context_outputs = model(**context_inputs)
                    
                # Get [MASK] token embedding
                mask_positions = (context_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) > 0:
                    context_emb = context_outputs.last_hidden_state[0, mask_positions[1][0]]
                else:
                    context_emb = context_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                # Final phrase embedding: average of content and context
                phrase_emb = (content_emb + context_emb) / 2
                phrase_embeddings.append(phrase_emb)
                
            # Average all phrase embeddings for the document
            if phrase_embeddings:
                doc_embedding = torch.stack(phrase_embeddings).mean(dim=0)
                scibert_embeddings.append(doc_embedding)
            else:
                scibert_embeddings.append(torch.zeros(768))
            
        return torch.stack(scibert_embeddings)