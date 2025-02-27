"""
hier wird rnn mit class torch.nn.RNN implementiert
"""
import os
from pathlib import Path
import torch  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first= True)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.relu1=nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size/2), num_classes)


    def forward(self, x):
        """
        Forward Pass
        x: (batch_size, sequence_length, input_size)
        future_steps: Number of future steps to predict
        out: batch_size, sequence_length, hidden_size
        hidden_state : num_layers, batch_size, hidden_size
        """ 
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("mps")
        out, _ = self.rnn(x, h0)
        out = self.fc(out)

        return out

    


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

def split_and_normalize_dataset(file_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    data = pd.read_csv(file_path).iloc[1:, 1:].values
    data = torch.tensor(data, dtype=torch.float32)

    dataset_size = len(data)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Define the 6 different combinations of slices
    combinations = [
        (slice(0, train_size), slice(train_size, train_size + val_size), slice(train_size + val_size, dataset_size)),

        (slice(val_size, val_size + train_size), slice(0, val_size),  slice(val_size + train_size, dataset_size)),

        (slice(test_size, test_size + train_size), slice(test_size + train_size, dataset_size), slice(0, test_size)),

        (slice(0, train_size), slice(train_size, train_size + test_size), slice(train_size + test_size, dataset_size)),

        (slice(val_size + test_size, dataset_size), slice(0, val_size), slice(val_size, val_size + test_size)),

        (slice(test_size + val_size, dataset_size), slice(test_size, test_size + val_size), slice(0, test_size))
    ]

    results = []
    for train_slice, val_slice, test_slice in combinations:
        train_data = data[train_slice]
        val_data = data[val_slice]
        test_data = data[test_slice]

        # Compute mean and standard deviation over features (columns)
        train_mean = train_data.mean(dim=0)
        train_std = train_data.std(dim=0)

        # Apply normalization
        normalized_train_data = (train_data - train_mean) / train_std
        normalized_val_data = (val_data - train_mean) / train_std
        normalized_test_data = (test_data - train_mean) / train_std

        results.append([normalized_train_data, normalized_val_data, normalized_test_data])

    return results



"""def split_and_normalize_dataset(file_path, train_ratio=0.7, val_ratio = 0.15):

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
"""
# Train Function
def train(model, data, optimizer, future_steps, loss_fn):

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


def validation(model, dataloader, future_steps, loss_fn):
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

def test(model, dataloader, future_steps,loss_fn):
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

def plot_predictions(model, data, sequence_length):
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
    
    start_idx = random.randint(0, len(data) - sequence_length - 1)
    input_sequence = data[start_idx:start_idx + sequence_length].unsqueeze(0).to(device)  # Add batch dimension


    # Get the actual future steps
    actual_future = data[start_idx + sequence_length:start_idx + sequence_length + future_steps].to(device)

    # Predict future steps
    with torch.no_grad():
        predicted_future = model(input_sequence, sequence_length).squeeze(0)  # Remove batch dimension

    
    # Convert tensors to lists for plotting
    actual_future = actual_future.cpu().tolist()
    predicted_future = predicted_future.cpu().tolist()
    
    # Plot the results in 2D
    time_steps = list(range(future_steps))
    plt.figure(figsize=(12, 6))

    # Plot x
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, [x[0] for x in actual_future], label='Actual Future')
    plt.plot(time_steps, [x[0] for x in predicted_future], label='Predicted Future')
    plt.title('X')
    plt.legend()

    # Plot y
    plt.subplot(2, 1, 2)
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

    #ax.plot([x[0] for x in actual_future], [x[1] for x in actual_future], label='Actual Future', color='b')
    #ax.plot([x[0] for x in predicted_future], [x[1] for x in predicted_future], label='Predicted Future', color='r', linestyle='--')
    ax.plot([x[0] for x in actual_future], [x[1] for x in actual_future], [x[2] for x in actual_future], label='Actual Future', color='b')
    ax.plot([x[0] for x in predicted_future], [x[1] for x in predicted_future], [x[2] for x in predicted_future], label='Predicted Future', color='r', linestyle='--')

    ax.set_xlabel('X')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Actual vs Predicted Future Steps')
    ax.legend()

    plt.show()





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
    
    
def seq_prediction(model, data, sequence_length, future_steps):
    # Geräteauswahl
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    model.eval()  # Modell in den Evaluierungsmodus versetzen

    # Zufälligen Startpunkt für die Sequenz auswählen
    start_idx = random.randint(0, len(data) - sequence_length - future_steps - 1)
    input_sequence = data[start_idx:start_idx + sequence_length].unsqueeze(0).to(device)  # Batch-Dimension hinzufügen

    # Tatsächliche zukünftige Schritte erhalten
    actual_future = data[start_idx + sequence_length:start_idx + sequence_length + future_steps].to(device)

    # Zukünftige Schritte vorhersagen
    with torch.no_grad():
        predicted_future = model(input_sequence).squeeze(0)  # Batch-Dimension entfernen

    # Tensoren in Listen für die Darstellung umwandeln
    actual_future = actual_future.cpu().tolist()
    predicted_future = predicted_future.cpu().tolist()
    
    return predicted_future, actual_future



def save_training_results(file_path, batch_size, epochs, input_dim, hidden_size, num_layers, dropout, learning_rate, sequence_lengths):
    # Load and preprocess data
    dataset_combination = split_and_normalize_dataset(file_path)
    
    # Initialize results dictionary
    results = {
        "hyperparameters": {
            "batch_size": batch_size,
            "sequence_length": sequence_lengths,
            "epochs": epochs,
            "input_dim": input_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate
        },
        "training_results": []
    }

    for i, (train_data, val_data, test_data) in enumerate(dataset_combination):
        for sequence in sequence_lengths:
            # Initialize the RNN model
            train_loader = create_dataloader(train_data, batch_size, sequence, shuffle=False)
            val_loader = create_dataloader(val_data, batch_size, sequence, shuffle=False)
            test_loader = create_dataloader(test_data, batch_size, sequence, shuffle=False)
            rnn_model = RNN(input_dim, hidden_size, num_layers, input_dim, sequence)

            # Initialize the optimizer
            optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

            # Initialize the scheduler
            scheduler = CosineWarmupScheduler(optimizer, warmup=20, max_iters=epochs)

            # Loss function
            loss_fn = nn.MSELoss()
            patience_start = 40
            if patience_start > epochs:
                patience_start = epochs
            else:
                patience_start = 40
            start_point = patience_start
            training_losses = []
            val_losses = []
            best_loss = float("inf")
            best_epoch = 0

            start_time = time.time()
            
            for epoch in range(epochs):
                # Train the model
                train_results = train(rnn_model, train_loader, optimizer, loss_fn)
                training_losses.append(train_results)

                # Evaluate the model
                val_results = validation(rnn_model, val_loader, loss_fn)
                val_losses.append(val_results)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch: {epoch + 1} Train Loss: {train_results}, Validation Loss: {val_results}")

                if val_results < best_loss:
                    best_loss = val_results
                    patience = patience_start  # Reset patience counter
                else:
                    patience -= 1
                if patience == 0:
                    best_training_loss = training_losses[-start_point]
                    best_val_loss = val_losses[-start_point]
                    test_loss = test(rnn_model, test_loader, loss_fn)
                    print("test_loss:", test_loss)
                    print(f"Best Train Loss: {best_training_loss}, Best Val Loss: {best_val_loss}")
                    best_epoch = epoch + 1
                    break 

                elif epoch == epochs - 1:
                    best_training_loss = training_losses[-patience]
                    best_val_loss = val_losses[-patience]
                    test_loss = test(rnn_model, test_loader, loss_fn)
                    print("test_loss:", test_loss)
                    print(f"Best Train Loss: {best_training_loss}, Best Val Loss: {best_val_loss}")
                    best_epoch = epoch + 1
                    break 
                # Step the scheduler
                scheduler.step()
            
            end_time = time.time()
            
            predicted_sequence, true_sequence = seq_prediction(rnn_model, test_data, sequence, sequence)
            
            training_result = {
                "combination_index": i + 1,
                "sequence_length": sequence,
                "training_time": end_time - start_time,
                "best_training_loss": best_training_loss,
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "best_epoch": best_epoch,
                "predicted_sequence": predicted_sequence,
                "true_sequence": true_sequence
            }
            
            results["training_results"].append(training_result)
    
    # Save results to JSON file in the same folder as the dataset file
    folder_path = os.path.dirname(file_path)
    json_file_path = os.path.join(folder_path, 'training_results.json')
    
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    # Example usage
    file_path = '/Users/Aleksandar/Documents/Uni/FP/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_system/lorenz_system_data.csv'
    batch_size = 64
    epochs = 200
    input_dim = 3  # Number of features
    hidden_size = 64
    num_layers = 1
    dropout = 0.0
    learning_rate = 0.0

    sequence_lengths = [0.5, 1.0, 2.0, 5.0]
    sequence_lengths = [int(x * 50) for x in sequence_lengths]

    save_training_results(file_path, batch_size, epochs, input_dim, hidden_size, num_layers, dropout, learning_rate, sequence_lengths)






    



    



    