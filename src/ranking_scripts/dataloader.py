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
        elif self.return_embedding == 'scibert':
            # Initialize SciBERT components
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            model.to(self.device)
            # Load AutoPhrase results
            self.autophrase_file = os.path.join(processed_datasets_path, "example_data_embedded_for_scibert.txt")
            phrases_per_doc = self._read_autophrase_output_for_scibert()
            # Create embeddings
            embeddings = self._create_scibert_embeddings(self.texts, len(self.texts), tokenizer, model, phrases_per_doc)
            
            # For scibert, return both embeddings and phrases in a list
            return [embeddings, phrases_per_doc]
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

    def _read_autophrase_output_for_scibert(self):
        """Read and parse AutoPhrase output file"""
        pattern = re.compile(r'<phrase_Q=[^>]*>([^<]+)</phrase>')
        phrases_per_doc = []
        with open(self.autophrase_file, "r", encoding="utf-8") as f:
            for line in f:
                matches = pattern.findall(line.strip())
                phrases_per_doc.append(matches if matches else [])
        return phrases_per_doc

    def _create_scibert_embeddings(self, texts, num_texts, tokenizer, model, phrases_per_doc):
        """Creates phrase-level SciBERT embeddings following paper methodology exactly"""
        print("Creating SciBERT embeddings")
        scibert_embeddings = []

        for doc_idx in range(num_texts):
            phrases = phrases_per_doc[doc_idx]
            doc_text = texts[doc_idx]
            
            if not phrases:
                scibert_embeddings.append(torch.zeros(768*2, device=self.device))
                continue
                
            doc_phrase_embeddings = []
            
            for phrase in phrases:
                phrase_mention_embeddings = []
                
                # Get all occurrences (mentions) of the phrase in doc
                phrase_mentions = []
                start_idx = 0
                while True:
                    idx = doc_text.find(phrase, start_idx)
                    if idx == -1:
                        break
                    phrase_mentions.append((idx, idx + len(phrase)))
                    start_idx = idx + 1
                    
                if not phrase_mentions:
                    continue
                    
                # For each mention
                for mention_start, mention_end in phrase_mentions:
                    # Content feature (x_l_p)
                    content_inputs = tokenizer(
                        phrase,
                        padding=True, 
                        truncation=True,
                        max_length=64,
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        content_outputs = model(**content_inputs)
                    content_emb = content_outputs.last_hidden_state.mean(dim=1).squeeze(0)

                    # Context feature (y_l_p) 
                    masked_text = (
                        doc_text[:mention_start] + 
                        tokenizer.mask_token + 
                        doc_text[mention_end:]
                    )
                    context_inputs = tokenizer(
                        masked_text,
                        padding=True,
                        truncation=True, 
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        context_outputs = model(**context_inputs)

                    # Get [MASK] token embedding
                    mask_positions = (context_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                    if len(mask_positions[0]) > 0:
                        context_emb = context_outputs.last_hidden_state[0, mask_positions[1][0]]
                    else:
                        context_emb = content_emb  # Fallback if mask token not found

                    # Concatenate content and context features for this mention
                    mention_emb = torch.cat([content_emb, context_emb], dim=0)
                    phrase_mention_embeddings.append(mention_emb)

                if phrase_mention_embeddings:
                    # Average across all mentions of this phrase
                    avg_phrase_emb = torch.stack(phrase_mention_embeddings).mean(dim=0)
                    doc_phrase_embeddings.append(avg_phrase_emb)

            if doc_phrase_embeddings:
                # Average across all phrases in document
                doc_embedding = torch.stack(doc_phrase_embeddings).mean(dim=0)
                scibert_embeddings.append(doc_embedding)
            else:
                scibert_embeddings.append(torch.zeros(768*2, device=self.device))

        return torch.stack(scibert_embeddings)

#TODO: - Check if embedding needs to be once or in every iteration
#      - Check if phrase classifier is correct
#      - Check if combined ranking is correct + figure out where to move the code
#      - Figure out the whole pipeline
#      - Get autophrase outputs for all datasets
class PhraseLevelProcessor:
    def __init__(self, review_dataset):
        """
        Initialize with dataset containing precomputed data
        
        Args:
            review_dataset: newReviewDataset instance with precomputed embeddings and phrases
        """
        self.dataset = review_dataset
        # Access precomputed data
        self.embeddings = review_dataset.embeddings[1] # The precomputed embeddings
        self.phrases_per_doc = review_dataset.embeddings[0] # The precomputed autophrase output

    def process_iteration(self, known_indices):
        """Process phrase level features for current iteration"""
        # Select relevant phrases based on current known docs
        selected_phrases = self.select_relevant_phrases(known_indices)
        
        # Build graph using PRECOMPUTED embeddings
        phrase_graph = self.construct_phrase_graph(selected_phrases)
        
        # Apply Louvain clustering
        communities, selected_features, feature_values = self.apply_louvain_to_phrase_graph(
            phrase_graph, selected_phrases, known_indices
        )
        
        return communities, selected_features, feature_values

    def select_relevant_phrases(self, known_indices, selection_percentage=0.30):
        """Select relevant phrases using ID and UN scores"""
        # Get labeled documents
        labeled_docs = [self.dataset.texts[i] for i in known_indices]
        labeled_labels = [self.dataset.labels[i] for i in known_indices]
        
        # Get phrases with quality scores from autophrase output
        relevant_docs = [doc for doc, label in zip(labeled_docs, labeled_labels) if label == 1]
        
        phrase_scores = {}
        for doc_idx, doc_phrases in enumerate(self.phrases_per_doc):
            if doc_idx in known_indices:
                for phrase in doc_phrases:
                    # Calculate ID score (eq 8)
                    np1 = sum(1 for doc in relevant_docs if phrase in doc)
                    id_score = np1 / len(relevant_docs) if len(relevant_docs) > 0 else 0
                    
                    # Calculate UN score (eq 9)
                    total_docs = len(labeled_docs)
                    docs_with_phrase = sum(1 for doc in labeled_docs if phrase in doc)
                    un_score = np.log(total_docs / (docs_with_phrase + 1e-10))
                    
                    if id_score > 0 and un_score > 0:
                        phrase_scores[phrase] = gmean([id_score, un_score])

        # Select top phrases
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(sorted_phrases) * selection_percentage)
        return {phrase for phrase, _ in sorted_phrases[:top_n]}

    def construct_phrase_graph(self, selected_phrases):
        """Construct graph using precomputed embeddings"""
        G = nx.Graph()
        phrases_to_use = [p for p in selected_phrases if p in self.embeddings]
        G.add_nodes_from(phrases_to_use)
        
        for i, phrase_i in enumerate(phrases_to_use):
            emb_i = self.embeddings[phrase_i]
            for j, phrase_j in enumerate(phrases_to_use[i:], start=i):
                emb_j = self.embeddings[phrase_j]
                sim = cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                weight = float(torch.sqrt(torch.clamp(torch.tensor(sim), min=0.0)))
                if weight > 0:
                    G.add_edge(phrase_i, phrase_j, weight=weight)
        
        return G

    def apply_louvain_to_phrase_graph(self, graph, selected_phrases, known_indices):
        """Apply Louvain clustering and get features"""
        communities = self._louvain_clustering(graph)
        
        # Calculate feature values using PRECOMPUTED embeddings
        selected_features = self._select_phrase_features(communities, selected_phrases, known_indices)
        feature_values = self._calculate_feature_values(selected_features, communities)
        
        return communities, selected_features, feature_values

    def _louvain_clustering(self, graph):
        """Apply Louvain clustering as per paper"""
        from sknetwork.clustering import Louvain
        
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        rows, cols, weights = [], [], []
        for u, v, data in graph.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            w = data.get('weight', 1.0)
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([w, w])
        
        adj_matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
        louvain = Louvain(resolution=1.0, random_state=None, tol=1e-9, max_iter=100)
        labels = louvain.fit_transform(adj_matrix)
        
        return {node: int(label) for node, label in zip(nodes, labels)}

    def _select_phrase_features(self, communities, selected_phrases, known_indices, alpha=0.5):
        """Select features using equations 11-13"""
        community_to_phrases = defaultdict(list)
        for phrase, comm in communities.items():
            community_to_phrases[comm].append(phrase)
            
        relevant_labeled = [i for i in known_indices if self.dataset.labels[i] == 1]
        
        cluster_to_docs = defaultdict(set)
        for cluster_id, phrases in community_to_phrases.items():
            for phrase in phrases:
                for doc_idx in known_indices:
                    if phrase in self.dataset.texts[doc_idx]:
                        cluster_to_docs[cluster_id].add(doc_idx)

        threshold = alpha * len(known_indices)
        return {cluster_id for cluster_id, docs in cluster_to_docs.items() 
                if len(docs) > threshold}

    def _calculate_feature_values(self, selected_features, communities):
        """Calculate feature values using precomputed embeddings"""
        feature_values = {}
        
        # Group phrases by community
        community_to_phrases = defaultdict(list)
        for phrase, comm in communities.items():
            if comm in selected_features:
                community_to_phrases[comm].append(phrase)
        
        # Calculate centroids using precomputed embeddings
        community_centroids = {}
        for comm_id, phrases in community_to_phrases.items():
            comm_embeddings = [self.embeddings[p] for p in phrases if p in self.embeddings]
            if comm_embeddings:
                community_centroids[comm_id] = torch.stack(comm_embeddings).mean(dim=0)
        
        # Calculate features for all docs
        for doc_idx, doc_text in enumerate(self.dataset.texts):
            doc_features = []
            for cluster_id in sorted(selected_features):
                cluster_phrases = community_to_phrases[cluster_id]
                max_sim = 0.0
                
                for phrase in cluster_phrases:
                    if phrase in doc_text and phrase in self.embeddings:
                        phrase_emb = self.embeddings[phrase]
                        centroid = community_centroids[cluster_id]
                        sim = cosine_similarity(
                            phrase_emb.unsqueeze(0),
                            centroid.unsqueeze(0)
                        ).item()
                        max_sim = max(max_sim, sim)
                        
                doc_features.append(max_sim)
            
            feature_values[doc_idx] = doc_features
            
        return feature_values

    def train_phrase_classifier(self):
        """Train classifier and get phrase-level ranking"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Get features for training
        X_train = [self.feature_values[idx] for idx in self.dataset.known_indices]
        y_train = [self.dataset.labels[idx] for idx in self.dataset.known_indices]
        X_unknown = [self.feature_values[idx] for idx in self.dataset.unknown_indices]
        
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        predictions = rf.predict_proba(X_unknown)[:, 1]
        
        return np.argsort(-predictions)