import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from dataloader import newReviewDataset,ReviewDataset
from dataloader_phrase import PhraseLevelPipeline, PhraselevelDataloader
from vae_model import VAE
from utils import set_seed, set_device
import os
import random
import math

#check if cuda is available
deviceType = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(deviceType)


# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))

def train_vae(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for embeddings,labels in dataloader:
            x = embeddings.to(device)
            y = labels.to(device)
           
            ranking_score, recon_x, mu, log_var = model(x)
            loss = model.loss_function(ranking_score, recon_x, x, mu, log_var, y, beta=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

def evaluate_vae(model, embeddings, device):
     model.eval()
     with torch.no_grad():
          x = embeddings.to(device)
          ranking_score, _,_,_ = model(x)
          predictions = ranking_score.cpu().numpy().flatten()
     return predictions
     

def rank_papers(model, unknown_embeddings, device):
    predictions = evaluate_vae(model, unknown_embeddings, device)
    print(predictions)
    indices = np.argsort(-predictions)
    return indices,predictions

def calculate_metrics(predictions, labels, threshold=0.5):
    """
    Calculates precision, recall, and f1-score for given predictions and labels.
    """
    print(labels)
    predicted_labels = (predictions >= threshold).astype(int) #converting the ranking scores to binary values
    print(predicted_labels)
    true_positives = np.sum((predicted_labels == 1) & (labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (labels == 0))
    false_negatives = np.sum((predicted_labels == 0) & (labels == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def evaluate_per_iteration(model, review_dataset, device, threshold = 0.5):
    
    #Calculate metrics for the training set
    train_embeddings,train_labels = review_dataset.get_known_data()
    train_predictions = evaluate_vae(model, train_embeddings, device)
    train_precision, train_recall, train_f1 = calculate_metrics(train_predictions, train_labels, threshold)
    print(f"Train Precision: {train_precision}, Recall: {train_recall}, F1-Score: {train_f1}")

    #Calculate metrics for the unknown set
    unknown_embeddings,unknown_labels = review_dataset.get_unknown_data()
    unknown_predictions = evaluate_vae(model, unknown_embeddings, device)
    unknown_precision, unknown_recall, unknown_f1 = calculate_metrics(unknown_predictions,unknown_labels, threshold)
    print(f"Unknown Precision: {unknown_precision}, Recall: {unknown_recall}, F1-Score: {unknown_f1}")

def ranking_ensemble(r1_d, r2_d):
    """
    Combine two ranking lists using mean reciprocal rank scores.
    
    Args:
        r1_d: First ranking list [r^1_d] from VAE model (numpy array of indices)
        r2_d: Second ranking list [r^2_d] from phrase-level features (numpy array of indices)
        
    Returns:
        numpy.ndarray: Array of indices in their final ranking order
    """
    # Initialize score dictionary
    final_scores = {}
    
    # Calculate MRR scores for each document (equation 17)
    for doc_idx in r1_d:  # Iterate through all document indices
        # Get positions in each ranking (adding 1 since rankings are 1-based)
        rank1 = np.where(r1_d == doc_idx)[0][0] + 1
        rank2 = np.where(r2_d == doc_idx)[0][0] + 1
        
        # Calculate MRR score: sum of reciprocal ranks (equation 17)
        mrr = (1/rank1) + (1/rank2)
        final_scores[doc_idx] = mrr
    
    # Sort document indices by final score in descending order
    ranked_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    
    return np.array(ranked_indices)

#TODO:
#     - Finish the ranking function (add some print statements etc..)
#     - Implement evaulation
def main():
    # Settings
    data_path = os.path.join(dir_path, './../../data/processed_datasets/Bannach-Brown_2019_ids.csv_processed.csv')
    #data_path = os.path.join(dir_path, './../../data/example_data_processed/example_data.csv')
    set_seed(999)
    device = set_device()
    
    latent_dim = 128
    initial_train_size = 1
    batch_size = 32
    epochs = 200
    
    # Initialize Dataset
    #review_dataset = newReviewDataset(data_path, initial_train_size=initial_train_size, return_embedding='specter', create_tensors=True,device=deviceType)
    review_dataset = newReviewDataset(data_path, initial_train_size=initial_train_size, return_embedding='tfidf',device=deviceType)
    #initialize model
    input_dim = review_dataset.embeddings.shape[1]
    vae = VAE(input_dim, latent_dim, beta=0.1).to(deviceType)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    
    #Phrase level settings
    phrase_dataset = PhraselevelDataloader(data_path, device=deviceType)

    # Iterative training loop
    while len(review_dataset.unknown_indices) > 0:
        print("************************")
        print(f"Train size: {len(review_dataset.train_embeddings)}, Unknonwn Size: {len(review_dataset.unknown_indices)}")
    
        # DataLoaders
        train_loader = DataLoader(review_dataset, batch_size=batch_size, shuffle=True)
    
        # Train the VAE
        train_vae(vae, train_loader, optimizer, epochs=epochs, device=device)

        # Get predictions and rank documents
        unknown_embeddings, unknown_labels = review_dataset.get_unknown_data()
        ranked_indices, predictions = rank_papers(vae, unknown_embeddings, device)
        
        #info
        print("Highest rankd index: ",ranked_indices[0])
        print("Value of highest ranked index: ",predictions[ranked_indices[0]])
        print("Label of highest ranked index: ",unknown_labels[ranked_indices[0]])
        # Select the highest ranked document
        selected_index = ranked_indices[0]
        
        # Get pseudo-negative documents and unknown_indices and run phrase level pipeline
        threshold_index = int(len(ranked_indices) * 0.7)  #plp needs indices from the worst 30% of unknow_texts as 0 labeled documents (pseudonegative docs)
        pseudonegative_unknown_indices = ranked_indices[threshold_index:]
        unknown_indices = review_dataset.unknown_indices
        plp = PhraseLevelPipeline(unknown_indices, pseudonegative_unknown_indices, phrase_dataset, device)
        rfc_ranking = plp.pipeline()

        if rfc_ranking is None:
            print("No positive documents in the training set, RFC was not trained.")
        else:        
            final_ranking = ranking_ensemble(ranked_indices, rfc_ranking)
            print("Final ranking: ",final_ranking)

        # Update the training set with human feedback (simulate by using ground truth label)
        review_dataset.update_train_set(selected_index)

        # Evaluate (optional, do not use for training)
        evaluate_per_iteration(vae, review_dataset, device)
        

if __name__ == '__main__':
    main()