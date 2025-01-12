import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))

class ReviewDataset(Dataset):
    def __init__(self, data_path, initial_train_size=1, return_embedding='specter',use_pseudo_for_scibert = False,start_idx=0):
        self.data_path = data_path
        self.texts, self.labels = self._load_data()
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter')
        self.embeddings = self._get_specter_embeddings(self.texts)
        
        # Placeholder for SciBERT - we generate random embeddings for now
        self.embeddings_scibert = self._create_placeholder_scibert_embeddings(len(self.texts))
        
        self.known_indices = list(range(initial_train_size))
        self.unknown_indices = list(range(initial_train_size, len(self.texts)))
        
        self.labels = np.array(self.labels)
       
        self.train_embeddings = self.embeddings[self.known_indices]
        self.train_labels = self.labels[self.known_indices]
        self.train_embeddings_scibert = [self.embeddings_scibert[i] for i in self.known_indices]

        self.unknown_embeddings = self.embeddings[self.unknown_indices]
        self.unknown_labels = self.labels[self.unknown_indices]
        self.unknown_embeddings_scibert = [self.embeddings_scibert[i] for i in self.unknown_indices]
        
        self.return_embedding = return_embedding
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
     
    def _create_placeholder_scibert_embeddings(self, num_texts):
    # Create random embeddings as a placeholder for SciBERT
        placeholder_embeddings = []
        for i in range(num_texts):
            num_phrases = np.random.randint(3,10) #Generate a random number of phrases between 3 and 10 for each paper
            phrase_embeddings = []
            for j in range(num_phrases):
                phrase_embeddings.append(torch.randn(768)) #768 is the scibert embedding size
            # Average the phrase embeddings for each document
            averaged_embedding = torch.mean(torch.stack(phrase_embeddings), dim=0)
            placeholder_embeddings.append(averaged_embedding)
        return placeholder_embeddings
    

    def update_train_set(self, index):
        """
        Adds the selected document (using the index from unknown documents) to the
        training set, moves its index from unknown to known
        """
        
        # extract paper from unknown
        new_train_embedding = self.unknown_embeddings[index]
        new_train_label = self.unknown_labels[index]
        new_train_embedding_scibert = self.unknown_embeddings_scibert[index]
        
        #add paper to train
        if len(self.train_embeddings.shape) == 1:
             self.train_embeddings = new_train_embedding.unsqueeze(0)
        else:
            self.train_embeddings = torch.cat((self.train_embeddings, new_train_embedding.unsqueeze(0)), dim=0)
        self.train_labels = np.append(self.train_labels, new_train_label)
        self.train_embeddings_scibert.append(new_train_embedding_scibert)
        
        #update known and unknown indices
        
        selected_index = self.unknown_indices[index]
        
        self.known_indices.append(selected_index)
        
        del self.unknown_indices[index]
        
        #update embeddings
        self.unknown_embeddings = self.embeddings[self.unknown_indices]
        self.unknown_labels = np.array(self.labels)[self.unknown_indices]
        self.unknown_embeddings_scibert = [self.embeddings_scibert[i] for i in self.unknown_indices]
        
    
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
        
    def get_unknown_data(self):
        if self.return_embedding == 'specter':
             return self.unknown_embeddings, self.unknown_labels
        elif self.return_embedding == 'scibert':
            return self.unknown_embeddings_scibert, self.unknown_labels
        elif self.return_embedding == 'both':
             return self.unknown_embeddings, self.unknown_embeddings_scibert, self.unknown_labels
    
    def get_known_data(self):
        if self.return_embedding == 'specter':
             return self.train_embeddings, self.train_labels
        elif self.return_embedding == 'scibert':
            return self.train_embeddings_scibert, self.train_labels
        elif self.return_embedding == 'both':
             return self.train_embeddings, self.train_embeddings_scibert, self.train_labels