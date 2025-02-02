import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from dataloader import newReviewDataset
from dataloader_phrase import PhraseLevelPipeline, PhraselevelDataloader
from vae_model import VAE
from utils import set_seed, set_device
import os
import random
import math
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import itertools

#Configure logger
logging.basicConfig(
    format='%(asctime)s - %(message)s',  # Include timestamp in logs
    level=logging.INFO  # Set the log level to INFO or DEBUG as needed
)

# Example usage
#logging.info("This is an info log with a timestamp")


#check if cuda is available
deviceType = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(deviceType)


# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))

def train_vae(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
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

            #logging.info(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

def evaluate_vae(model, embeddings, device):
     model.eval()
     with torch.no_grad():
          x = embeddings.to(device)
          ranking_score, _,_,_ = model(x)
          predictions = ranking_score.cpu().numpy().flatten()
     return predictions
     

def rank_papers(model, unknown_embeddings, device):
    predictions = evaluate_vae(model, unknown_embeddings, device)
    logging.info(predictions)
    indices = np.argsort(-predictions)
    return indices,predictions

def calculate_metrics(predictions, labels, threshold=0.5):
    """
    Calculates precision, recall, and f1-score for given predictions and labels.
    """
    logging.info(labels)
    predicted_labels = (predictions >= threshold).astype(int) #converting the ranking scores to binary values
    logging.info(predicted_labels)
    true_positives = np.sum((predicted_labels == 1) & (labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (labels == 0))
    false_negatives = np.sum((predicted_labels == 0) & (labels == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_metrics_top_k(predictions, labels):
    """
    Calculates precision, recall, and f1-score for given predictions and labels,
    assuming the top-k predictions are positive, where k is the number of true
    relevant documents.
    """
    k = np.sum(labels == 1)  # Number of true relevant documents
    if k == 0:
        return 0, 0, 0  # Handle case where there are no relevant documents left

    # Get indices of top-k predictions
    top_k_indices = np.argsort(-predictions)[:k]

    # Create predicted labels based on top-k indices
    predicted_labels = np.zeros_like(predictions)
    predicted_labels[top_k_indices] = 1

    true_positives = np.sum((predicted_labels == 1) & (labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (labels == 0))
    false_negatives = np.sum((predicted_labels == 0) & (labels == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


reached_10 = False
reached_wss95 = False
reached_wss85 = False

def evaluate_per_iteration(model, review_dataset, device, threshold = 0.5):
    
    #Calculate metrics for the training set
    train_embeddings,train_labels = review_dataset.get_known_data()
    train_predictions = evaluate_vae(model, train_embeddings, device)
    train_precision, train_recall, train_f1 = calculate_metrics(train_predictions, train_labels, threshold)
    logging.info(f"Train Precision: {train_precision}, Recall: {train_recall}, F1-Score: {train_f1}")
    #logging.info(f"Found {np.sum(train_labels)} valid papers")
    #Calculate metrics for the unknown set
    unknown_embeddings,unknown_labels = review_dataset.get_unknown_data()
    unknown_predictions = evaluate_vae(model, unknown_embeddings, device)
    #unknown_precision, unknown_recall, unknown_f1 = calculate_metrics(unknown_predictions,unknown_labels, threshold)
    unknown_precision, unknown_recall, unknown_f1 = calculate_metrics_top_k(unknown_predictions,unknown_labels)
    logging.info(f"Found {np.sum(train_labels)/(np.sum(train_labels)+np.sum(unknown_labels))} % of valid papers({np.sum(train_labels)}/{np.sum(train_labels)+np.sum(unknown_labels)}) in the training set in number of iterations ({len(train_labels)}/{len(train_labels)+len(unknown_labels)})")

    logging.info(f"Unknown Precision: {unknown_precision}, Recall: {unknown_recall}, F1-Score: {unknown_f1}")
    
    
    global reached_10
    global reached_wss95
    global reached_wss85
    number_of_relevant_docs = sum(train_labels)+sum(unknown_labels)
    #calculate rrf @10
    # if training len is 10 % of combined length of training and unknown set
    if len(train_embeddings) >= int(0.1*(len(train_embeddings)+len(unknown_embeddings))) and not reached_10:
        reached_10 = True
        
        rrf_score = sum(train_labels)/number_of_relevant_docs
        logging.info(f"RRF @10: {rrf_score}")
        #write the score to a file
        with open("scores.txt", "a") as f:
            f.write(f"RRF@10 = {rrf_score}\n")
            
    
    
    if sum(train_labels)>= 0.95 * number_of_relevant_docs and not reached_wss95:
        reached_wss95 = True
        wss_score = 1 - (len(train_labels)/number_of_relevant_docs)
        logging.info(f"WSS @95: {wss_score}")
        with open("scores.txt", "a") as f:
            f.write(f"WSS@95 = {wss_score}\n")
    
    if sum(train_labels)>= 0.85 * number_of_relevant_docs and not reached_wss85:
        reached_wss85 = True
        wss_score = 1 - (len(train_labels)/number_of_relevant_docs)
        logging.info(f"WSS @85: {wss_score}")
        with open("scores.txt", "a") as f:
            f.write(f"WSS@85 = {wss_score}\n")
        
    
    #calculate combined prcisions and recall
    # cobined_predicitons = np.concatenate((train_predictions,unknown_predictions))
    # combined_labels = np.concatenate((train_labels,unknown_labels))
    # combined_precision, combined_recall, combined_f1 = calculate_metrics(cobined_predicitons,combined_labels, threshold)
    # logging.info(f"Combined Precision: {combined_precision}, Recall: {combined_recall}, F1-Score: {combined_f1}")

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
    
    #calculate recall at k for both rankings
    
    
    
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
#     - Finish the ranking function (add some logging.info statements etc..)
#     - Implement evaulation
def main():
    # Settings
    #data_path = os.path.join(dir_path, './../../data/processed_datasets/Bannach-Brown_2019_ids.csv_processed.csv')
    data_path = os.path.join(dir_path, './../../data/processed_datasets/Cohen_2006_CalciumChannelBlockers_ids.csv_processed.csv')
    #data_path = os.path.join(dir_path, './../../data/processed_datasets/Kwok_2020_ids.csv_processed.csv')
    #data_path = os.path.join(dir_path, './../../data/example_data_processed/example_data.csv')
    
    paths = [os.path.join(dir_path, './../../data/processed_datasets/Bannach-Brown_2019_ids.csv_processed.csv'),os.path.join(dir_path, './../../data/processed_datasets/Cohen_2006_CalciumChannelBlockers_ids.csv_processed.csv')
             ,os.path.join(dir_path, './../../data/processed_datasets/Kwok_2020_ids.csv_processed.csv')]
    seeds =[42,999,49681]
    seed = 49681
    set_seed(seed)
    
    device = set_device()
    
    
    enable_phrase_level = True
    latent_dim = 128
    initial_train_size = 1
    batch_size = 64
    epochs = 200
    
    # Initialize Dataset
    #review_dataset = newReviewDataset(data_path, initial_train_size=initial_train_size, return_embedding='specter', create_tensors=True,device=deviceType)
    for data_path,seed in itertools.product(paths,seeds):
    
        with open("scores.txt", "a") as f:
            f.write("************************\n")
            f.write(f"Scores for {data_path}\n")
            f.write(f"Seed = {seed}\n")
            f.write(f"************************\n")
    
        review_dataset = newReviewDataset(data_path, initial_train_size=initial_train_size, return_embedding='tfidf',device=deviceType,shuffle=True,seed=seed)
        
        
        #initialize model
        input_dim = review_dataset.embeddings.shape[1]
        vae = VAE(input_dim, latent_dim, beta=0.1).to(deviceType)
        #vae = torch.compile(vae)
        optimizer = optim.Adam(vae.parameters(), lr=1e-4)
        
        #Phrase level settings
        
        indices = review_dataset.get_shuffle_indices()
        phrase_dataset = PhraselevelDataloader(data_path,deviceType,indices)
        
        iteration = 0
        # Iterative training loop
        while len(review_dataset.unknown_indices) > 0 and not reached_wss95:
            logging.info("************************")
            logging.info(f"Train size: {len(review_dataset.train_embeddings)}, Unknonwn Size: {len(review_dataset.unknown_indices)}")
        
            # DataLoaders
            train_loader = DataLoader(review_dataset, batch_size=batch_size)
            
            # Train the VAE
            logging.info("Training VAE...")
            train_vae(vae, train_loader, optimizer, epochs=epochs, device=device)
            logging.info("Training complete.")

            # Get predictions and rank documents
            unknown_embeddings, unknown_labels = review_dataset.get_unknown_data()
            ranked_indices, predictions = rank_papers(vae, unknown_embeddings, device)
            
            #info
            logging.info(f"Highest ranked index: {ranked_indices[0]}")
            logging.info(f"Value of highest ranked index: {predictions[ranked_indices[0]]}")
            logging.info(f"Label of highest ranked index: {unknown_labels[ranked_indices[0]]}")

            # Select the highest ranked document
            selected_index = ranked_indices[0]
            
            if enable_phrase_level and iteration >= 30:
                # Get pseudo-negative documents and unknown_indices and run phrase level pipeline
                threshold_index = int(len(ranked_indices) * 0.7)  #plp needs indices from the worst 30% of unknow_texts as 0 labeled documents (pseudonegative docs)
                pseudonegative_unknown_indices = ranked_indices[threshold_index:]
                unknown_indices = review_dataset.unknown_indices
                plp = PhraseLevelPipeline(unknown_indices, pseudonegative_unknown_indices, phrase_dataset, device)
                logging.info("Running phrase level pipeline...")
                rfc_ranking = plp.pipeline()
                
              

                if rfc_ranking is None :
                    logging.info("No positive documents in the training set, RFC was not trained.")
                else:        
                    final_ranking = ranking_ensemble(ranked_indices, rfc_ranking)
                    logging.info(f"Final ranking: {final_ranking}")
                    selected_index = final_ranking[0]

    
            review_dataset.update_train_set(selected_index)

            # Evaluate (optional, do not use for training)
            evaluate_per_iteration(vae, review_dataset, device)
            
            # Calculate WSS and RRF every 20 iterations
            
            y_true = np.array([review_dataset.labels[i] for i in ranked_indices])
            y_pred = (predictions[ranked_indices] >= 0.5).astype(int)  # Convert to binary predictions
            y_scores = predictions[ranked_indices]
            
            
            
            
            iteration += 1
        

if __name__ == '__main__':
    main()