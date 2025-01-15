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




#TODO: - Check if embedding needs to be once or in every iteration
#      - Check if phrase classifier is correct
#      - Check if combined ranking is correct + figure out where to move the code
#      - Figure out the whole pipeline
#      - Get autophrase outputs for all datasets
class PhraseLevelProcessor:
    def __init__(self, tokenizer, model, autophrase_filename, device='cpu'):
        """
        Initialize the phrase level processor
        
        Args:
            tokenizer: SciBERT tokenizer
            model: SciBERT model
            device: torch device
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.autophrase_file = os.path.join(processed_datasets_path, autophrase_filename)

    def process_iteration(self, texts, labels, known_indices):
        """
        Process phrase level features for current iteration
        
        Args:
            texts: List of all document texts
            labels: Array of document labels
            known_indices: List of indices of labeled documents
            
        Returns:
            tuple: (communities, selected_features, feature_values)
        """
        # Read phrases from AutoPhrase
        all_phrases = self._read_autophrase_output_for_scibert()
        
        # Select relevant phrases based on current labeled documents
        selected_phrases = self.select_relevant_phrases(
            [texts[i] for i in known_indices],
            [labels[i] for i in known_indices]
        )
        
        # Create embeddings for selected phrases
        phrase_embeddings = self.create_selected_phrase_embeddings(
            texts, 
            len(texts),
            self.tokenizer,
            self.model,
            selected_phrases
        )
        
        # Build phrase graph
        phrase_graph = self.construct_phrase_graph(phrase_embeddings)
        
        # Apply Louvain clustering and get features
        communities, selected_features, feature_values = self.apply_louvain_to_phrase_graph(
            [phrase_graph, phrase_embeddings, selected_phrases]
        )
        
        return communities, selected_features, feature_values
    
    def _read_autophrase_output_for_scibert(self):
        """Read and parse AutoPhrase output file"""
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

    def louvain_clustering(self, graph):
        """
        Implementation of Louvain clustering using scikit-network while following SciMine specifications.
        The algorithm finds communities by optimizing modularity in a hierarchical way.
        
        Args:
            graph: NetworkX graph with weighted edges representing phrase similarities
            
        Returns:
            dict: Node to community mapping
        """
        from sknetwork.clustering import Louvain
        import numpy as np
        from scipy import sparse
        
        # Convert NetworkX graph to sparse adjacency matrix
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        # Create sparse matrix with weights
        rows, cols, weights = [], [], []
        for u, v, data in graph.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            w = data.get('weight', 1.0)
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([w, w])
        
        adj_matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
        
        # Initialize Louvain with parameters matching SciMine's approach
        louvain = Louvain(
            resolution=1.0,  # Standard resolution for modularity optimization
            random_state=None,  # Deterministic results
            tol=1e-9,  # High precision for convergence
            max_iter=100  # Allow sufficient iterations for convergence
        )
        
        # Run Louvain clustering
        labels = louvain.fit_transform(adj_matrix)
        
        # Convert numeric labels back to a dictionary mapping nodes to their communities
        communities = {}
        unique_labels = np.unique(labels)
        
        # Ensure community labels are consecutive integers starting from 0
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        for node, label in zip(nodes, labels):
            communities[node] = label_map[label]
        
        return communities

    def apply_louvain_to_phrase_graph(self, phrase_level_input):
        """
        Apply Louvain clustering to the phrase graph and store results.
        This function applies community detection and selects phrase-level features
        according to SciMine paper specifications.
        
        Args:
            phrase_level_input: List containing [phrase_graph, phrase_embeddings, selected_phrases]
            
        Returns:
            tuple: (community_assignments, selected_features, feature_values)
                - community_assignments: Dict mapping phrases to their communities
                - selected_features: Set of cluster IDs selected as features
                - feature_values: Dict mapping documents to their feature vectors
                
        Note:
            This follows section 4.3 of SciMine paper for identifying 
            non-overlapping communities and selecting phrase-level features.
        """
        try:
            # Extract components from input
            phrase_graph = phrase_level_input[0]
            phrase_embeddings = phrase_level_input[1]
            self.selected_phrases = phrase_level_input[2]
            
            if len(phrase_graph.nodes()) == 0:
                print("Warning: Empty phrase graph")
                return {}, set(), {}
                
            # Apply Louvain clustering
            communities = self.louvain_clustering(phrase_graph)
            
            # Store the community assignments
            self.phrase_communities = communities
            
            # Create reverse mapping of community to phrases
            community_to_phrases = defaultdict(list)
            for phrase, comm in communities.items():
                community_to_phrases[comm].append(phrase)
            self.community_to_phrases = dict(community_to_phrases)
            
            # Calculate and store community centroids
            community_centroids = {}
            for comm_id, phrases in community_to_phrases.items():
                comm_embeddings = [phrase_embeddings[p] for p in phrases if p in phrase_embeddings]
                if comm_embeddings:
                    centroid = torch.stack(comm_embeddings).mean(dim=0)
                    community_centroids[comm_id] = centroid
            self.community_centroids = community_centroids
            
            # Select phrase-level features using equations 11-13 from the paper
            selected_features = self.select_phrase_level_features()
            
            # Calculate feature values for each document
            feature_values = {}
            for doc_idx, doc in enumerate(self.texts):
                doc_features = []
                for cluster_id in sorted(selected_features):  # Sort for consistent order
                    # Get max similarity between doc phrases and cluster phrases
                    cluster_phrases = self.community_to_phrases[cluster_id]
                    max_sim = 0.0
                    for phrase in cluster_phrases:
                        if phrase in doc:
                            # Get phrase embedding and cluster centroid
                            if phrase in phrase_embeddings and cluster_id in community_centroids:
                                phrase_emb = phrase_embeddings[phrase]
                                centroid = community_centroids[cluster_id]
                                # Calculate cosine similarity
                                sim = torch.cosine_similarity(phrase_emb.unsqueeze(0), 
                                                           centroid.unsqueeze(0)).item()
                                max_sim = max(max_sim, sim)
                    doc_features.append(max_sim)
                feature_values[doc_idx] = doc_features
            
            # Store results
            self.feature_values = feature_values
            
            # Print statistics
            num_communities = len(set(communities.values()))
            num_features = len(selected_features)
            avg_community_size = len(self.selected_phrases) / num_communities if num_communities > 0 else 0
            print(f"Number of communities detected: {num_communities}")
            print(f"Number of selected features: {num_features}")
            print(f"Average community size: {avg_community_size:.2f}")
            
            return communities, selected_features, feature_values
            
        except ImportError:
            print("Please install scikit-network: pip install scikit-network")
            raise
        except Exception as e:
            print(f"Error in Louvain clustering: {str(e)}")
            raise
    
    def select_phrase_level_features(self, alpha=0.5):
        """
        Select phrase-level features from clusters based on correlation with relevant documents.
        Following equations 11-13 from SciMine paper.
        
        Args:
            alpha: Threshold percentage for feature selection (default: 0.5)
            
        Returns:
            set: Selected cluster IDs that serve as phrase-level features
        """
        # Get labeled documents and their indices
        labeled_docs = {i: doc for i, doc in enumerate(self.texts) 
                    if i in self.known_indices}
        
        # Get relevant labeled documents (D_l ∩ R) - equation 13 reference set
        relevant_labeled_docs = {
            idx: doc for idx, doc in labeled_docs.items() 
            if self.labels[idx] == 1
        }
        
        # For each phrase, find documents containing it among relevant labeled docs (D_p)
        phrase_to_docs = defaultdict(set)
        for phrase in self.selected_phrases:
            for doc_idx, doc in relevant_labeled_docs.items():
                if phrase in doc:
                    phrase_to_docs[phrase].add(doc_idx)

        # For each cluster, find related documents (D_ci) - equation 12
        cluster_to_docs = defaultdict(set)
        for cluster_id, phrases in self.community_to_phrases.items():
            for phrase in phrases:
                cluster_to_docs[cluster_id].update(phrase_to_docs[phrase])

        # Select clusters based on threshold (C_s) - equation 11
        threshold = alpha * len(labeled_docs) 
        selected_clusters = {
            cluster_id for cluster_id, docs in cluster_to_docs.items()
            if len(docs) > threshold
        }
        
        # Store results
        self.selected_cluster_features = selected_clusters
        self.cluster_to_docs = dict(cluster_to_docs)
        
        # Print statistics
        print(f"Number of clusters selected as features: {len(selected_clusters)}")
        print(f"Total number of clusters: {len(self.community_to_phrases)}")
        
        return selected_clusters

    def train_phrase_classifier(self):
        """
        Train Random Forest classifier on phrase-level features and get predictions.
        Returns ranked indices of unknown documents.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Get feature vectors and labels for training
        X_train = [self.feature_values[idx] for idx in self.known_indices]
        y_train = [self.labels[idx] for idx in self.known_indices]
        
        # Get feature vectors for unknown documents
        X_unknown = [self.feature_values[idx] for idx in self.unknown_indices]
        
        # Train classifier and get predictions
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        predictions = rf.predict_proba(X_unknown)[:, 1]  # Get probability of positive class
        
        # Return sorted indices (highest probability first)
        return np.argsort(-predictions)
    
    def get_combined_ranking(self, vae_ranking):
        """
        Combine VAE and phrase-level rankings using mean reciprocal rank.
        Returns: Final sorted indices.
        """
        # Get phrase-level ranking
        phrase_ranking = self.train_phrase_classifier()
        
        # Calculate MRR scores
        scores = np.zeros(len(self.unknown_indices))
        for idx in range(len(self.unknown_indices)):
            # Add reciprocal ranks (1-based ranking)
            scores[idx] = (1.0/(vae_ranking[idx] + 1) + 
                        1.0/(phrase_ranking[idx] + 1))
        
        # Return indices sorted by combined score
        return np.argsort(-scores)
