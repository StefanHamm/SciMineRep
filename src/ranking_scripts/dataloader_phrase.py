import math
import pickle
from typing import Dict, List, Set, Tuple, TypeVar
import nltk
import torch
from torch.utils.data import Dataset
import pandas as pd
import tqdm
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
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from tqdm.auto import tqdm
from community import community_louvain
from community import best_partition
import networkx as nx
from typing import List, Dict, Set, Tuple
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch

# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
processed_datasets_path = os.path.join(dir_path, "..", "..", "data", "processed_datasets")

class PhraselevelDataloader(Dataset):
    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.texts, self.labels = self._load_data()
        self.device = device
        
        #print count of 0 and 1 labels sum(labels),len(labels)-sum(labels)
        print("====Data Summary====")
        print("0 = not relevant, 1 = relevant")
        print("Number of 0 labels:",len(self.labels)-sum(self.labels))
        print("Number of 1 labels:",sum(self.labels))
        
        print("Intializing embeddings")

        # Embedding initialization
        # Check if the embeddings and phrases are saved in files
        embedding_file = os.path.join(processed_datasets_path, "embedding_phrase_based_test.pkl")
        phrase_file = os.path.join(processed_datasets_path, "unique_phrases_test.pkl")

        if os.path.exists(embedding_file) and os.path.exists(phrase_file):
            # Load the embeddings and phrases from files
            with open(embedding_file, "rb") as f:
                self.embeddings = pickle.load(f)
            with open(phrase_file, "rb") as f:
                self.unique_phrases = pickle.load(f)
        else:
            # Initialize the embeddings and phrases using the function
            self.embeddings, self.unique_phrases = self._initialize_embeddings()

            # Save the embeddings and phrases to files
            with open(embedding_file, "wb") as f:
                pickle.dump(self.embeddings, f)
            with open(phrase_file, "wb") as f:
                pickle.dump(self.unique_phrases, f)
        
    def _load_data(self):
        data = pd.read_csv(self.data_path)
        texts = data['title'] + ' ' + data['abstract']
        labels = data['label'].values
        return texts.tolist(), labels.tolist()

    def _initialize_embeddings(self):
        # Initialize SciBERT components
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        model.to(self.device)

        # Load AutoPhrase results
        processed_datasets_path = os.path.join(dir_path, "..", "..", "data", "processed_datasets")
        autophrase_file = os.path.join(processed_datasets_path, "example_data_embedded_for_scibert.txt")
        unique_phrases = self._read_autophrase_output_for_scibert(autophrase_file)

        # Create embeddings
        phrase_embeddings = self._create_scibert_embeddings(self.texts, unique_phrases, tokenizer, model)
        
        return phrase_embeddings, unique_phrases  # Dictionary {phrase: embedding}

    def _read_autophrase_output_for_scibert(self, file_path):
        try:
            # First try UTF-8
            with open(file_path, 'r', encoding='utf-8') as file:
                autophrase_output = file.read()
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try UTF-16
                with open(file_path, 'r', encoding='utf-16') as file:
                    autophrase_output = file.read()
            except UnicodeDecodeError:
                # If both fail, try with error handling
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    autophrase_output = file.read()

        phrase_pattern = r'<phrase_Q=1\.000>(.+?)</phrase>'
        phrases = re.findall(phrase_pattern, autophrase_output)

        unique_phrases = list(set(phrases))

        return unique_phrases

    def _create_scibert_embeddings(self, texts: List[str], unique_phrases: List[str], 
                                tokenizer, model) -> List[np.ndarray]:
        """
        Create MLM-based embeddings following the paper's specifications exactly.
        Now with NLTK sentence tokenization and caching for improved accuracy and performance.
        """
        from nltk.tokenize import sent_tokenize
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt_tab')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Initialize storage for all mentions of each phrase
        phrase_mentions = {phrase: [] for phrase in unique_phrases}
        
        # Create cache filename based on input file
        cache_filename = os.path.join(processed_datasets_path, "data_phrase_based.txt")
        
        # Try to load cached sentences first
        all_sentences = []
        try:
            if os.path.exists(cache_filename):
                print("Loading cached sentence splits...")
                with open(cache_filename, 'r', encoding='utf-8') as f:
                    all_sentences = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(all_sentences)} sentences from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}")
            all_sentences = []
        
        # If no cached sentences, process texts and save to cache
        if not all_sentences:
            print("Processing documents and creating sentence cache...")
            all_sentences = []
            for text in tqdm(texts, desc="Processing documents"):
                # Split text into title and abstract
                parts = text.split(',', 1)  # Split on first comma to separate title and abstract
                if len(parts) == 2:
                    title, abstract = parts
                    # Remove quotes if present
                    abstract = abstract.strip('"')
                    sentences = [title]  # Title is first sentence
                    # Use NLTK's sent_tokenize for more accurate sentence splitting
                    sentences.extend(sent_tokenize(abstract))
                else:
                    # If no comma found, treat entire text as one sentence
                    sentences = [text]
                all_sentences.extend(sentences)
            
            # Save processed sentences to cache
            try:
                with open(cache_filename, 'w', encoding='utf-8') as f:
                    for sentence in all_sentences:
                        f.write(f"{sentence}\n")
                print(f"Saved {len(all_sentences)} sentences to cache.")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        # Process each sentence for phrases
        for sentence in tqdm(all_sentences, desc="Processing sentences"):
            for phrase in unique_phrases:
                if phrase in sentence:
                    # 1. Get content feature (x_l_p)
                    inputs = tokenizer(sentence, return_tensors="pt", 
                                    truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get token positions for the phrase
                    phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
                    phrase_positions = []
                    
                    for i in range(len(inputs['input_ids'][0])):
                        if inputs['input_ids'][0][i:i+len(phrase_tokens)].tolist() == phrase_tokens:
                            phrase_positions.extend(range(i, i+len(phrase_tokens)))
                            break
                    
                    if not phrase_positions:  # Skip if phrase tokens not found
                        continue
                        
                    # Average phrase token embeddings for content feature
                    content_feature = outputs.last_hidden_state[0, phrase_positions].mean(dim=0)
                    
                    # 2. Get context feature (y_l_p)
                    # Replace phrase with single [MASK] token
                    masked_sentence = sentence.replace(phrase, tokenizer.mask_token)
                    masked_inputs = tokenizer(masked_sentence, return_tensors="pt",
                                        truncation=True, max_length=512)
                    masked_inputs = {k: v.to(device) for k, v in masked_inputs.items()}
                    
                    with torch.no_grad():
                        masked_outputs = model(**masked_inputs)
                    
                    # Get embedding of [MASK] token
                    mask_positions = (masked_inputs['input_ids'] == tokenizer.mask_token_id).nonzero()
                    if len(mask_positions) == 0:  # Skip if mask token not found
                        continue
                        
                    context_feature = masked_outputs.last_hidden_state[0, mask_positions[0, 1]]
                    
                    # 3. Concatenate content and context features [x_l_p; y_l_p]
                    mention_embedding = torch.cat([content_feature, context_feature])
                    phrase_mentions[phrase].append(mention_embedding.cpu().numpy())
        
        # 4. Calculate final embeddings by averaging mention vectors
        ordered_embeddings = []
        for phrase in unique_phrases:  # Maintain original order
            mentions = phrase_mentions[phrase]
            if mentions:
                # e_p = (1/N_p) * sum([x_l_p; y_l_p])
                ordered_embeddings.append(np.mean(mentions, axis=0))
            else:
                # For phrases not found, use zero vector of same size
                if len(ordered_embeddings) > 0:
                    ordered_embeddings.append(np.zeros_like(ordered_embeddings[0]))
                else:
                    # For first phrase if not found, determine size from model
                    embedding_size = model.config.hidden_size * 2  # *2 for concatenated features
                    ordered_embeddings.append(np.zeros(embedding_size))
        
        return ordered_embeddings


# TODO: 
#      - Check why selected_communities is empty
#      - create autophrase if the preprocessed documents are done
#      - Precompute embeddings when we have the autophrase and set checks in the dataloades so it loads the correct file based on an input string to the dataloader
#      - Clean up print statements when bugs are fixed
class PhraseLevelPipeline(Dataset):
    def __init__(self, unknown_indices, pseudonegative_unknown_indices, phrase_dataset: PhraselevelDataloader, device='cpu'):
        self.unknown_indices = unknown_indices
        self.pseudonegative_unknown_indices = pseudonegative_unknown_indices
        self.ds = phrase_dataset
        self.device = device
        print(f"Phrase Level Pipeline Initialized - Number of unknown_indices: {len(self.unknown_indices)}")

    def pipeline(self):
        #Load data from ds and split into unknown and known
        unique_phrases = self.ds.unique_phrases
        embeddings = self.ds.embeddings
        unknown_texts = [text for idx, text in enumerate(self.ds.texts) if idx in self.unknown_indices]
        train_texts = [text for idx, text in enumerate(self.ds.texts) if idx not in self.unknown_indices]
        train_labels = [text for idx, text in enumerate(self.ds.labels) if idx not in self.unknown_indices]
        pseudonegative_texts = [text for idx, text in enumerate(unknown_texts) if idx in self.pseudonegative_unknown_indices]

        selected_phrases, selected_embeddings = self._select_relevant_phrases(unique_phrases, train_texts, train_labels, embeddings, 0.3)
        # print(selected_phrases)
        # print(selected_embeddings)
        graph = self._construct_phrase_graph(selected_phrases, selected_embeddings)
        # print(graph)
        relevant_texts = self._get_relevant_texts(train_texts, train_labels)
        community_phrases, selected_communities = self._detect_phrase_communities(train_texts, train_labels, graph, 1, relevant_texts)
        # print(f'community_phrases {community_phrases}')
        # print(f'selected_communities {selected_communities}')
        selected_phrases = self._get_selected_phrases(community_phrases, selected_communities)
        # print(f'selected_phrases {selected_phrases}')
        print("Input data:")
        print("  embeddings shape:", len(embeddings))
        print("  unique_phrases length:", len(unique_phrases))
        print("  community_phrases keys:", community_phrases.keys())
        print("  selected_communities:", selected_communities)
        print("  train_texts length:", len(train_texts))
        print("  train_labels length:", len(train_labels))
        print("  pseudonegative_texts length:", len(pseudonegative_texts))
        RFC = self._train_random_forest(embeddings, unique_phrases, community_phrases, selected_communities, train_texts, train_labels, pseudonegative_texts)
        if RFC is None:
            print("Random Forest classifier not trained due to no selected communities.")
            # Handle the case when the classifier is not trained, e.g., skip subsequent steps that rely on the classifier
            return None
        else:
            ranking = self._get_phrase_rankings(RFC, embeddings, unique_phrases, community_phrases, selected_communities, unknown_texts, pseudonegative_texts)

        return ranking

    def _get_relevant_texts(self, texts, labels):
        relevant_texts = []
        # Iterate over texts and labels simultaneously
        for text, label in zip(texts, labels):
            if label == 1:
                relevant_texts.append(text)

        return relevant_texts

    def _select_relevant_phrases(self, unique_phrases, texts, labels, scibert_embeddings, selection_percentage=0.30):
        """
        Selects the most relevant phrases based on indicative and unusual measures.
        
        Args:
            unique_phrases (list): List of unique phrases.
            texts (list): List of documents, each document is a string with title and abstract concatenated.
            labels (list): List of corresponding relevance labels for each document. 
            scibert_embeddings (list): List of embeddings for the unique phrases.
            selection_percentage (float, optional): Percentage of top phrases to select. Defaults to 0.30.
        
        Returns:
            tuple: selected_phrases (list), selected_embeddings (list)
        """
        assert len(texts) == len(labels), "Number of documents and labels must match"
        assert len(unique_phrases) == len(scibert_embeddings), "Number of unique phrases and embeddings must match"

        num_relevant_docs = sum(labels)
        num_labeled_docs = len(labels)
        assert num_labeled_docs > 0

        phrase_scores = []
        
        for phrase, embedding in zip(unique_phrases, scibert_embeddings):
            np1 = 0  # Number of relevant docs phrase appears in
            np0 = 0  # Number of irrelevant docs phrase appears in
            
            for text, label in zip(texts, labels):
                if phrase in text:
                    if label == 1:
                        np1 += 1
                    else:
                        np0 += 1
            
            if num_relevant_docs == 0:
                indicative = 0
            else:
                indicative = np1 / num_relevant_docs
            if np1 + np0 == 0:
                unusual = 0  # Or assign a default value
            else:
                unusual = math.log(num_labeled_docs / (np1 + np0))
            
            score = math.sqrt(indicative * unusual)  # Geometric mean
            
            phrase_scores.append((phrase, score, embedding))
        
        # Sort phrases by score in descending order
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top phrases based on selection percentage
        num_selected = int(len(phrase_scores) * selection_percentage)
        selected_phrases_scores = phrase_scores[:num_selected]
        
        selected_phrases = [phrase for phrase, _, _ in selected_phrases_scores]  
        selected_embeddings = [embedding for _, _, embedding in selected_phrases_scores]
        
        return selected_phrases, selected_embeddings

    def _construct_phrase_graph(self, phrases, embeddings):
        """Eq 10: Build phrase graph using embedding similarities"""
        G = nx.Graph()
        G.add_nodes_from(phrases)
        
        # Calculate pairwise similarities (eq 10)
        for i, (phrase_i, emb_i) in enumerate(zip(phrases, embeddings)):
            for phrase_j, emb_j in zip(phrases[i+1:], embeddings[i+1:]):
                emb_i_tensor = torch.tensor(emb_i)
                emb_j_tensor = torch.tensor(emb_j)
                sim = cosine_similarity(emb_i_tensor.unsqueeze(0), emb_j_tensor.unsqueeze(0)).item()
                weight = float(torch.sqrt(torch.tensor(max(sim, 0))))  # Clamp and sqrt as per paper
                G.add_edge(phrase_i, phrase_j, weight=weight)
        
        return G

    def _detect_phrase_communities(self, documents, labels, graph, threshold = 0.1, relevant_docs = None):
        """
        Implements community detection and feature selection as described in the paper.
        """
        # Step 1: Apply Louvain clustering to detect communities
        # Using the Louvain method for community detection as specified in the paper
        # Implementation from python-louvain package which follows the original algorithm
        communities = best_partition(graph, random_state=42)  # Set random_state for reproducibility
        
        # Organize phrases by community
        community_phrases: Dict[int, Set[str]] = {}
        for phrase, community_id in communities.items():
            if community_id not in community_phrases:
                community_phrases[community_id] = set()
            community_phrases[community_id].add(phrase)
        
        # Get positive documents (D_l in equation 13)
        D_l = {i for i, label in enumerate(labels) if label == 1}
        
        # If relevant_docs (R) not provided, assume all documents are potentially relevant
        if relevant_docs is None:
            relevant_docs = set(range(len(documents)))
        
        # D_l ∩ R as specified in equation (13)
        D_l_intersection_R = D_l.intersection(relevant_docs)
        
        def phrase_in_document(phrase: str, document: str) -> bool:
            """Check if a phrase appears as a complete word/phrase in the document."""
            pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
            return bool(re.search(pattern, document.lower()))
        
        # Step 2: Implement equations (11), (12), and (13)
        community_documents: Dict[int, Set[int]] = {}
        
        for community_id, phrases in community_phrases.items():
            # Initialize set for D_ci (equation 12)
            D_ci = set()
            
            # For each phrase in community
            for phrase in phrases:
                # Find documents containing this phrase (D_p in equation 13)
                D_p = set()
                for doc_idx in D_l_intersection_R:
                    if phrase_in_document(phrase, documents[doc_idx]):
                        D_p.add(doc_idx)
                
                # Update D_ci with documents containing this phrase
                D_ci.update(D_p)
            
            community_documents[community_id] = D_ci
        
        # Apply threshold criterion (equation 11)
        threshold_count = threshold * len(D_l)
        selected_communities = {
            community_id 
            for community_id, docs in community_documents.items() 
            if len(docs) > threshold_count
        }
        
        return community_phrases, selected_communities

    def _get_selected_phrases(self, community_phrases, selected_communities):
        """
        Helper function to get all phrases from selected communities.
        """
        selected_phrases = set()
        for community_id in selected_communities:
            selected_phrases.update(community_phrases[community_id])
        return selected_phrases

    def _train_random_forest(self, embeddings, unique_phrases, community_phrases, selected_communities, train_texts, train_labels, pseudonegative_texts):
        """
        Train a random forest classifier based on phrase-level features.
        Implements equations 14-16 from the paper exactly.
        """
        if not selected_communities:
            print("No communities explicitly selected. Using all available communities.")
            return None
        
        # Create mapping from phrases to their embeddings
        phrase_to_embedding = {phrase: emb for phrase, emb in zip(unique_phrases, embeddings)}
        
        # Calculate centroids for each community (equation 15)
        community_centroids = {}
        for community_id in selected_communities:
            phrases_in_community = community_phrases[community_id]
            
            # Get embeddings for phrases in this community
            total_weighted_embedding = None
            total_weight = 0.0
            
            # First calculate |D_ci| - number of documents the community is related to
            D_ci = set()
            for phrase in phrases_in_community:
                for text in train_texts:
                    if phrase in text:
                        D_ci.add(text)
            D_ci_size = len(D_ci)
            
            # Now calculate weighted centroid
            for phrase in phrases_in_community:
                if phrase in phrase_to_embedding:
                    # Calculate |D_p| - number of documents containing this phrase
                    D_p = sum(1 for text in train_texts if phrase in text)
                    
                    # Calculate weight w_p = |D_p| / |D_ci| (equation 15)
                    if D_ci_size > 0:  # Avoid division by zero
                        weight = D_p / D_ci_size
                        phrase_emb = torch.tensor(phrase_to_embedding[phrase])
                        
                        # Accumulate weighted embeddings
                        if total_weighted_embedding is None:
                            total_weighted_embedding = weight * phrase_emb
                        else:
                            total_weighted_embedding += weight * phrase_emb
                        total_weight += weight
            
            # Normalize to get final centroid
            if total_weighted_embedding is not None and total_weight > 0:
                community_centroids[community_id] = total_weighted_embedding / total_weight
        
        # Calculate feature values for each document (equations 14 and 16)
        feature_values = []
        all_texts = train_texts + pseudonegative_texts
        all_labels = train_labels + [0] * len(pseudonegative_texts)  # Pseudo-negative documents are labeled 0
        
        for text in all_texts:
            document_features = []
            
            for community_id in sorted(selected_communities):  # Sort for consistent ordering
                community_centroid = community_centroids[community_id]
                
                # Find max f_p,ci for all phrases p in document d that belong to community c_j
                # This implements equation 16: F_d,j = max({f_p,cj} | p ∈ d, p ∈ c_j, c_j ∈ C_s)
                max_similarity = 0.0
                for phrase in phrases_in_community:
                    if phrase in text and phrase in phrase_to_embedding:
                        phrase_emb = torch.tensor(phrase_to_embedding[phrase])
                        
                        # Calculate cosine similarity f_p,ci (equation 14)
                        similarity = torch.nn.functional.cosine_similarity(
                            phrase_emb.unsqueeze(0),
                            community_centroid.unsqueeze(0)
                        ).item()
                        max_similarity = max(max_similarity, similarity)
                
                document_features.append(max_similarity)
            
            feature_values.append(document_features)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(feature_values, all_labels)
        
        return clf, feature_values

    def _get_phrase_rankings(self, rfc, embeddings, unique_phrases, community_phrases, selected_communities, unknown_texts, pseudonegative_texts):
        """
        Generate phrase-level feature-based rankings using the trained Random Forest classifier for non-pseudonegative unknown documents.

        Args:
            rfc: Trained Random Forest classifier
            embeddings: List of phrase embeddings
            unique_phrases: List of all phrases
            community_phrases: Dict mapping community IDs to sets of phrases
            selected_communities: Set of selected community IDs
            unknown_texts: List of unknown documents
            pseudonegative_texts: List of pseudonegative documents

        Returns:
            numpy.ndarray: Array of indices giving the ranking order [r^2_d] for non-pseudonegative unknown documents
        """
        # Determine the non-pseudonegative unknown texts
        non_pseudonegative_unknown_texts = [text for text in unknown_texts if text not in pseudonegative_texts]

        # Create mapping from phrases to their embeddings
        phrase_to_embedding = {phrase: emb for phrase, emb in zip(unique_phrases, embeddings)}

        # Calculate community centroids
        community_centroids = {}
        for community_id in selected_communities:
            phrases_in_community = community_phrases[community_id]

            # Calculate |D_ci| and weighted embeddings
            D_ci = set()
            for phrase in phrases_in_community:
                for text in non_pseudonegative_unknown_texts:
                    if phrase in text:
                        D_ci.add(text)
            D_ci_size = len(D_ci)

            total_weighted_embedding = None
            total_weight = 0.0

            for phrase in phrases_in_community:
                if phrase in phrase_to_embedding:
                    D_p = sum(1 for text in non_pseudonegative_unknown_texts if phrase in text)

                    if D_ci_size > 0:
                        weight = D_p / D_ci_size
                        phrase_emb = torch.tensor(phrase_to_embedding[phrase])

                        if total_weighted_embedding is None:
                            total_weighted_embedding = weight * phrase_emb
                        else:
                            total_weighted_embedding += weight * phrase_emb
                        total_weight += weight

            if total_weighted_embedding is not None and total_weight > 0:
                community_centroids[community_id] = total_weighted_embedding / total_weight

        # Calculate feature values for prediction
        feature_values = []
        for text in non_pseudonegative_unknown_texts:
            document_features = []

            for community_id in sorted(selected_communities):
                community_centroid = community_centroids[community_id]
                max_similarity = 0.0

                for phrase in phrases_in_community:
                    if phrase in text and phrase in phrase_to_embedding:
                        phrase_emb = torch.tensor(phrase_to_embedding[phrase])
                        similarity = torch.nn.functional.cosine_similarity(
                            phrase_emb.unsqueeze(0),
                            community_centroid.unsqueeze(0)
                        ).item()
                        max_similarity = max(max_similarity, similarity)

                document_features.append(max_similarity)

            feature_values.append(document_features)

        # Get predictions and rankings
        predictions = rfc.predict_proba(feature_values)[:, 1]  # Get probability of positive class
        ranked_indices = np.argsort(-predictions)  # Sort in descending order

        return ranked_indices  # Return rankings for non-pseudonegative unknown documents