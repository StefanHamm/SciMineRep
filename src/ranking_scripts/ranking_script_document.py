import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

def load_data(path):
    data = pd.read_csv(path)
    #texts are the columns title and abstract in a dataframe
    texts = data[['title', 'abstract']]
    texts = data['title'] + ' ' + data['abstract']
    labels = data['label'].values
    return texts, labels

def get_specter_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    
    title_abs = [d["title"] + tokenizer.sep_token + d["abstract"] for d in texts]
    
    inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt",max_length=512)
    with torch.no_grad():
        result = model(**inputs)
    embedding = result.last_hidden_state[:, 0, :]
    return embedding.numpy()

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)
        self.beta = beta
        self.classifier = nn.Linear(latent_dim, 1)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc2(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        y_pred = self.classifier(z)
        return recon_x, mu, log_var, y_pred

def loss_function(recon_x, x, mu, log_var, y_pred, y_true, beta):
    reconstruction_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    classification_loss = nn.BCEWithLogitsLoss()(y_pred, y_true.unsqueeze(1))
    return reconstruction_loss + beta * kl_divergence + classification_loss

def train_vae(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            recon_x, mu, log_var, y_pred = model(x)
            loss = loss_function(recon_x, x, mu, log_var, y_pred, y, model.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')
        
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            mu, _ = model.encode(x)
            predictions.append(mu.cpu().numpy())
    return np.concatenate(predictions)


def train_and_rank(train_path, test_path):
    # Load training data
    train_texts, train_labels = load_data(train_path)
    train_embeddings = get_specter_embeddings(train_texts)
    
    # Load test data
    test_texts, test_labels = load_data(test_path)
    test_embeddings = get_specter_embeddings(test_texts)
    
    # Define the VAE model
    input_dim = train_embeddings.shape[1]
    latent_dim = 128
    beta = 0.1
    vae = VAE(input_dim, latent_dim, beta)
    
    # Prepare DataLoader for training
    train_dataset = TensorDataset(torch.tensor(train_embeddings, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train the VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    train_vae(vae, train_loader, optimizer, epochs=200, device=device)
    
    # Get relevance scores for test set
    vae.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
        _, _, _, y_pred = vae(test_tensor)
        relevance_scores = y_pred.squeeze().cpu().numpy()
    
    # Create ranking based on relevance scores
    test_indices = np.argsort(-relevance_scores)  # descending order
    
    # Return the ranking as indices
    return test_indices

def ranking_algorithm2(path):
    texts, labels = load_data(path)
    embeddings = get_specter_embeddings(texts)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    latent_dim = 128
    beta = 0.1
    vae = VAE(input_dim, latent_dim, beta).to(device)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    train_vae(vae, train_loader, optimizer, epochs=200, device=device)
    
    latent_rep = evaluate(vae, DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=32), device)
    return latent_rep

def main():
    texts, labels = load_data()
    embeddings = get_specter_embeddings(texts)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    latent_dim = 128
    beta = 0.1
    vae = VAE(input_dim, latent_dim, beta).to(device)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    train_vae(vae, train_loader, optimizer, epochs=200, device=device)
    
    latent_rep = evaluate(vae, DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=32), device)
    
    # Classification using the latent representation
    # Implement your classification logic here
