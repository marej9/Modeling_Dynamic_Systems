## Standard libraries
import os
import numpy as np
import random
import math
import json
from functools import partial
import pandas as pd
from torch.utils.data import Dataset, DataLoader

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb
import matplotlib
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from optuna.pruners import MedianPruner
import optuna
import time


# Define the scaled dot product function
def scaled_dot_product(q, k, v, mask=None):
    # Retrieves the dimension of the query vectors.
    d_k = q.size()[-1]
    # Computes the dot product between the query (q) and the transpose of the key (k), resulting in the attention logits.
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    # Scales the attention logits by the square root of the dimension of the query vectors to stabilize gradients.
    attn_logits = attn_logits / math.sqrt(d_k)
    # Applies a mask to the attention logits, setting positions where the mask is zero to a very large negative value to prevent attention to those positions.
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    # Computes the weighted sum of the value vectors (v) using the attention weights.
    values = torch.matmul(attention, v)
    return values, attention

# Helper function to expand mask
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # creates a linear layer that projects the input tensor into three separate tensors: query (Q), key (K), and value (V) vectors.
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        # creates another linear layer that projects the concatenated output of all attention heads back to the original embedding dimension.
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # initialize the parameters of the linear layers.
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        # Projects the input tensor x into a combined query, key, and value tensor using the qkv_proj linear layer.
        qkv = self.qkv_proj(x)
        
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        
        # Permutes the dimensions to [batch_size, num_heads, seq_length, 3*head_dim] to facilitate splitting into Q, K, and V.
        qkv = qkv.permute(0, 2, 1, 3)
        # Splits the combined tensor into three separate tensors: query (q), key (k), and value (v), each with shape [batch_size, num_heads, seq_length, head_dim].
        q, k, v = qkv.chunk(3, dim=-1)
        
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        # Permutes the dimensions of the values tensor to [batch_size, seq_length, num_heads, head_dim]
        values = values.permute(0, 2, 1, 3)
        # Reshapes the values tensor to [batch_size, seq_length, embed_dim] by concatenating the heads
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        # Projects the concatenated values tensor back to the original embedding dimension using the o_proj linear layer.
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, embed_dim ,num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, embed_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x
    
        
class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
    
    

# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handling odd d_model
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
       
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
# Define the TransformerPredictor class
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, dim_feedforward, num_layers, dropout=0.1, input_dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim),
            nn.ReLU(inplace=True),  # Activation function
            nn.Linear(self.model_dim, self.model_dim)  # Additional layer
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            embed_dim=self.model_dim,
            num_heads=self.num_heads,
            dim_feedforward=2*self.model_dim,
            dropout=self.dropout
        )
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),  # Additional layer
            nn.ReLU(inplace=True),  # Activation function
            nn.Linear(self.model_dim, self.input_dim)
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x
    

# Custom Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, data_set, sequence_length):
        """
        Prepares data for sequence-based training with future steps targets
        """
        
        """        # Drop the time column (assume it's not needed for the model)
        data = pd.read_csv(file_path).iloc[1:, 1:].values  # Load X, Y, Z
        
        #convert to Pytorch tensor
        data = torch.tensor(data,dtype=torch.float32)

        #compute mean and standard deviation over features (columns)
        mean = data.mean(dim=0)
        std = data.std(dim=0)

         # Apply normalization
        data = (data - mean) / std"""
        

        """   
        self.sequence_length = sequence_length
        self.future_steps = future_steps
        """
        

        self.X, self.Y = [], []
       

        for i in range(len(data_set) - sequence_length - sequence_length + 1):
            self.X.append(data_set[i:i+sequence_length])
            self.Y.append(data_set[i+sequence_length: i+sequence_length+sequence_length])     

        # Create inputs (X) and targets (Y)
        self.X = torch.stack(self.X)  # All rows except the last
        self.Y = torch.stack(self.Y)   # All rows except the first (shifted)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return a single sample (input, target)
        return self.X[index], self.Y[index]
        

def create_dataloader(file_path, batch_size, sequence_length, shuffle=False):
    """
    Creates a DataLoader from the given file path.
    """
    dataset = TimeSeriesDataset(file_path, sequence_length)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def split_and_normalize_dataset(file_path, train_ratio=0.7, val_ratio = 0.15):

    data = pd.read_csv(file_path).iloc[1:, 1:].values
    data = torch.tensor(data, dtype=torch.float32)

    dataset_size = len(data) 
    train_size = int(train_ratio * dataset_size)
    val_size =  int(val_ratio * dataset_size)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size+val_size:]

    #compute mean and standard deviation over features (columns)
    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)

    # Apply normalization
    normalized_train_data = (train_data - train_mean) / train_std
    normalized_val_data = (val_data - train_mean) / train_std
    normalized_test_data = (test_data - train_mean) / train_std

    return normalized_train_data, normalized_val_data, normalized_test_data

# Train Function
def train(model, data, optimizer, loss_fn):

    if torch.cuda.is_available():
        device = "cuda"
    elif  torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    device = torch.device(device)
    model.to(device)
    model.train()

    batch_losses=list()

   
    for X, Y in data:  # X, Y are batches
        
        # Send tensors to the device
        X, Y = X.to(device), Y.to(device)

        # Clear gradients
        optimizer.zero_grad()

        prediction = model(X)
        loss = loss_fn(prediction, Y)
        # prediction of shape (batch_size, future_steps, output_size=features)
        # Y of shape (batch_size, future_steps, output_size=features)        
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
    
    # Store epoch loss
    train_loss = np.mean(batch_losses) # mean value for all batches in one epoch
    return train_loss


def validation(model, dataloader, loss_fn):
    """
    Evaluates the model on the given dataloader and calculates the average loss.
    
    Parameters:
        model: Trained RNN model
        dataloader: DataLoader containing the evaluation dataset
        loss_fn: Loss function (e.g., MSELoss)
        future_steps: Number of future steps to predict
        
    Returns:
        avg_loss: Average loss over the entire dataset
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif  torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    device = torch.device(device)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    all_losses = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in dataloader:  # Iterate over batches
            X, Y = X.to(device), Y.to(device)
            
            predictions = model(X)
            loss = loss_fn(predictions, Y)
            all_losses.append(loss.item())

    val_loss = np.mean(all_losses)  # Calculate the average loss
    #print(f"==> Average Loss: {val_loss:.6f}")

    return  val_loss

def test(model, dataloader, loss_fn):
    """
    Test the model on the given dataloader and calculates the average loss.
    
    Parameters:
        model: Trained RNN model
        dataloader: DataLoader containing the evaluation dataset
        loss_fn: Loss function (e.g., MSELoss)
        future_steps: Number of future steps to predict
        
    Returns:
        avg_loss: Average loss over the entire dataset
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif  torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    all_losses = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in dataloader:  # Iterate over batches
            X, Y = X.to(device), Y.to(device)
            
            predictions = model(X)
            loss = loss_fn(predictions, Y)
            all_losses.append(loss.item())

    test_loss = np.mean(all_losses)  # Calculate the average loss
    #print(f"==> Average Loss: {val_loss:.6f}")

    return  test_loss

def plot_predictions(model, data, sequence_length, future_steps):
    if torch.cuda.is_available():
        device = "cuda"
    elif  torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    model.to(device)
    
    model.eval()  # Set model to evaluation mode

    # Select a random starting point for the sequence
    start_idx = 100
    input_sequence = data[start_idx:start_idx + sequence_length].unsqueeze(0).to(device)  # Add batch dimension


    # Get the actual future steps
    actual_future = data[start_idx + sequence_length:start_idx + sequence_length + future_steps].to(device)

    # Predict future steps
    with torch.no_grad():
        predicted_future = model(input_sequence).squeeze(0)  # Remove batch dimension

    
    # Convert tensors to lists for plotting
    actual_future = actual_future.cpu().tolist()
    predicted_future = predicted_future.cpu().tolist()
    
    # Plot the results in 2D
    time_steps = list(range(future_steps))
    plt.figure(figsize=(12, 6))

    # Plot x
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, [x[0] for x in actual_future], label='Actual Future')
    plt.plot(time_steps, [x[0] for x in predicted_future], label='Predicted Future')
    plt.title('X')
    plt.legend()

    # Plot y
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, [x[1] for x in actual_future], label='Actual Future')
    plt.plot(time_steps, [x[1] for x in predicted_future], label='Predicted Future')
    plt.title('Y')
    plt.legend()

    # Plot z
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, [x[2] for x in actual_future], label='Actual Future')
    plt.plot(time_steps, [x[2] for x in predicted_future], label='Predicted Future')
    plt.title('Z')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the results in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot([x[0] for x in actual_future], [x[1] for x in actual_future], [x[2] for x in actual_future], label='Actual Future', color='b')
    ax.plot([x[0] for x in predicted_future], [x[1] for x in predicted_future], [x[2] for x in predicted_future], label='Predicted Future', color='r', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Actual vs Predicted Future Steps')
    ax.legend()

    plt.show()



def objective(trial, file_path, sequence_length):
    # Initialize test_loss
    test_loss = None
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 80, 300, step = 20)
    # epochs = trial.suggest_int('epochs', 60, 400, step=20)
    input_dim = 3  # Number of features
    model_dim = trial.suggest_int('model_dim', 16, 128, step=2)
    embed_dim = model_dim
    num_heads = trial.suggest_categorical("num_heads", [1, 2])
    dim_feedforward = trial.suggest_int('dim_feedforward', 4, 256)
    num_layers = trial.suggest_int('num_layers', 2, 16)
    dropout = trial.suggest_float("dropout", 0.0, 0.1)
    input_dropout = trial.suggest_float("input_dropout", 0.0, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)

    # Load and preprocess data
    train_data, val_data, test_data = split_and_normalize_dataset(file_path)

    # Create dataloaders
    train_loader = create_dataloader(train_data, batch_size, sequence_length, shuffle=False)
    val_loader = create_dataloader(val_data, batch_size, sequence_length, shuffle=False)
    test_loader = create_dataloader(test_data, batch_size, sequence_length, shuffle=False)

    # Initialize the model
    model = TransformerPredictor(input_dim, model_dim, num_heads, dim_feedforward, num_layers, dropout, input_dropout)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the scheduler
    scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=epochs)
    # Loss function
    loss_fn = nn.MSELoss()
    
    training_losses = []
    val_losses = []
    test_losses = []
    patience_start = 50
    if patience_start > epochs:
        patience_start = epochs
    start_point = patience_start
    training_losses = []
    val_losses = []
    best_loss = float("inf")
    best_epoch = 0  # Variable to store the best epoch

    for epoch in range(epochs):
        # Train the model
        train_results = train(model, train_loader, optimizer, loss_fn)
        training_losses.append(train_results)

        # Evaluate the model
        val_results = validation(model, val_loader, loss_fn)
        val_losses.append(val_results)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1} Train Loss: {train_results}, Validation Loss: {val_results}")

        if val_results < best_loss:
            best_loss = val_results
            patience = patience_start  # Reset patience counter
            best_epoch = epoch + 1  # Store the best epoch
        else:
            patience -= 1
        if patience == 0:
            best_val_loss = val_losses[-start_point]
            break 

        elif epoch == epochs - 1:
            best_val_loss = val_losses[-patience]
            break 
        # Step the scheduler
        scheduler.step()
    
    return best_val_loss, best_epoch

def optimize_for_datasets(dataset_paths, sequence_lengths):
    results = {}
    
    for file_path in dataset_paths:
        for sequence_length in sequence_lengths:
            start_time = time.time()  # Record the start time
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, file_path, sequence_length), n_trials=1)
            end_time = time.time()  # Record the end time

            best_params = study.best_params
            best_val_loss, best_epoch = study.best_trial.values  # Get the best validation loss and epoch
            best_params["best_val_loss"] = best_val_loss  # Add the best validation loss to the parameters
            best_params["best_epoch"] = best_epoch  # Add the best epoch to the parameters
            best_params["sequence_length"] = sequence_length  # Add the sequence length to the parameters
            best_params["optimization_time"] = end_time - start_time  # Add the optimization time to the parameters
            results[f"{file_path}_seq_len_{sequence_length}"] = best_params

    with open('/Users/Aleksandar/Documents/Uni/FP/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_system/all_best_hyperparams.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    dataset_paths = [
        "/Users/Aleksandar/Documents/Uni/FP/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_system/lorenz_system_data.csv"
    ]
    sequence_lengths = [0.5, 1.0, 2.0, 5.0]
    sequence_lengths = [int(x * 50) for x in sequence_lengths]
    optimize_for_datasets(dataset_paths, sequence_lengths)