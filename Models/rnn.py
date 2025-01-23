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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first= True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, future_steps = 1):
        """
        Forward Pass
        x: (batch_size, sequence_length, input_size)
        future_steps: Number of future steps to predict
        out: batch_size, sequence_length, hidden_size
        hidden_state : num_layers, batch_size, hidden_size
        """ 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        outputs = []
        out, h = self.rnn(x, h0) # pass trough RNN

        out = self.fc(out[:, -1, :]) # initial output from last time step, Extrahiert den Hidden State des letzten Zeitschritts der Sequenz für jeden Batch.
        outputs.append(out)
        
        for _ in range(future_steps - 1): # iterative prediction for future steps
            out, h = self.rnn(out.unsqueeze(1), h)
            out = self.fc(out[:, -1, :])
            
            outputs.append(out)

        return torch.stack(outputs, dim=1) # combine output : (batch_size, future_steps, num_classes)
     

    


# Custom Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, data_set, sequence_length, future_steps):
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
       

        for i in range(len(data_set) - sequence_length - future_steps + 1):
            self.X.append(data_set[i:i+sequence_length])
            self.Y.append(data_set[i+sequence_length: i+sequence_length+future_steps])     

        # Create inputs (X) and targets (Y)
        self.X = torch.stack(self.X)  # All rows except the last
        self.Y = torch.stack(self.Y)   # All rows except the first (shifted)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return a single sample (input, target)
        return self.X[index], self.Y[index]
        

def create_dataloader(file_path, batch_size, sequence_length, future_steps, shuffle=False):
    """
    Creates a DataLoader from the given file path.
    """
    dataset = TimeSeriesDataset(file_path, sequence_length, future_steps)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def split_and_normalize_dataset(file_path, train_ratio=0.7):

    data = pd.read_csv(file_path).iloc[1:, 1:].values
    data = torch.tensor(data, dtype=torch.float32)

    dataset_size = len(data) 
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_data = data[:train_size ]
    test_data = data[train_size:]

    #compute mean and standard deviation over features (columns)
    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)

    # Apply normalization
    normalized_train_data = (train_data - train_mean) / train_std
    normalized_test_data = (test_data - train_mean) / train_std


    return normalized_train_data, normalized_test_data


# Train Function
def train(model, data, epochs, optimizer, loss_fn, future_steps):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    batch_losses=list()

    print("=> Starting training")


    
    for X, Y in data:  # X, Y are batches
        

        # Send tensors to the device
        X, Y = X.to(device), Y.to(device)

        # Clear gradients
        optimizer.zero_grad()

        prediction = model(X, future_steps = future_steps)
        loss = loss_fn(prediction, Y)
        # prediction of shape (batch_size, future_steps, output_size=features)
        # Y of shape (batch_size, future_steps, output_size=features)
        

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

        
    # Store epoch loss
    train_loss = np.mean(batch_losses) # mean value for all batches in one epoch
    return train_loss
        




def predict_sequence(model, file_path, start_point, future_steps, sequence_length):
    """
    Plots predictions vs. true values for all features in a single graph.
    
    Parameters:
        model: Trained RNN model
        dataloader: DataLoader object containing the test dataset
        future_steps: Number of future steps to predict
    """
    
    model.eval()  # Set the model to evaluation mode
    

    # Drop the time column (assume it's not needed for the model)
    data = pd.read_csv(file_path).iloc[1:, 1:].values  # Load X, Y, Z
        
    #convert to Pytorch tensor
    data = torch.tensor(data,dtype=torch.float32)

    #compute mean and standard deviation over features (columns)
    mean = data.mean(dim=0)
    std = data.std(dim=0)

    # Apply normalization
    data = (data - mean) / std
    X, Y = [], []
       
    
    X = data[start_point: start_point +sequence_length].unsqueeze(0).to(device)
    Y = data[start_point + sequence_length: start_point +sequence_length+future_steps].unsqueeze(0).to(device)
    # X.shape = batch_size, future_steps, num_features
    
    print(f"X shape of {X.shape}")


    with torch.no_grad():
        predictions = model(X, future_steps=future_steps)  # Predict future steps

    # Shape: (sequence_length, num_features)
    Y = Y.squeeze(0).cpu().numpy()  # Shape: (future_steps, num_features)
    predictions = predictions.squeeze(0).cpu().numpy()  # Shape: (future_steps, num_features)
    
    print("True values shape:", Y.shape)
    print("Predictions shape:", predictions.shape)
    # Plot each feature
    num_features = Y.shape[1]
 

      # Plot predictions vs. true values for each feature
    time_steps = np.arange(future_steps)
    for feature_index in range(num_features):
        plt.figure(figsize=(8, 4))
        plt.plot(time_steps, Y[:, feature_index], label="True Values", marker="o")
        plt.plot(time_steps, predictions[:, feature_index], label="Predictions", marker="x", linestyle="--")
        plt.title(f"Feature {feature_index + 1}: Predictions vs True Values")
        plt.xlabel("Future Time Steps")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid()
    plt.show()





def evaluate(model, dataloader, loss_fn, future_steps):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    print("=> Starting evaluation")
    all_losses = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in dataloader:  # Iterate over batches
            X, Y = X.to(device), Y.to(device)
            
            predictions = model(X, future_steps=future_steps)
            loss = loss_fn(predictions, Y)
            all_losses.append(loss.item())

    avg_loss = np.mean(all_losses)  # Calculate the average loss
    print(f"==> Average Loss: {avg_loss:.4f}")

    return  avg_loss


def plot_prediction(model, dataloader, future_steps):
    """
    Plots the true values and predictions for one iteration.

    Parameters:
    model: Trained RNN model
    dataloader: DataLoader containing the evaluation dataset
    future_steps: Number of future steps to predict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in dataloader:  # Iterate over batches
            X, Y = X.to(device), Y.to(device)
            
            predictions = model(X, future_steps=future_steps)
            
            # Plot the first batch
            true_values = Y[0].cpu().numpy()  # Shape: (future_steps, num_features)
            predicted_values = predictions[0].cpu().numpy()  # Shape: (future_steps, num_features)
            
            time_steps = np.arange(future_steps)
            features = ['X', 'Y', 'Z']
            
            plt.figure(figsize=(12, 6))
            for i in range(true_values.shape[1]):
                plt.subplot(1, 3, i + 1)
                plt.plot(time_steps, true_values[:, i], label='True Values', marker='o')
                plt.plot(time_steps, predicted_values[:, i], label='Predictions', marker='x', linestyle='--')
                plt.title(f'Feature {features[i]}: Predictions vs True Values')
                plt.xlabel('Future Time Steps')
                plt.ylabel('Normalized Value')
                plt.legend()
                plt.grid()
            
            plt.tight_layout()
            plt.show()


def full_training(sys, dataset_path, sequence_length, future_steps):
    
    batch_size = sys["batch_size"]
    epochs = sys["epochs"]
    hidden_size = sys["hidden_size"]
    input_size = sys["input_size"]
    output_size = sys["output_size"]
    learning_rate = sys["learning_rate"]
    num_layers = sys["num_layers"]
    patience = system["patience"]
    val_losses=list()
    training_losses = list()
    best_loss = float("inf")
    patience_start = patience
    #split dataset 
    
    training_dataset, test_dataset = split_and_normalize_dataset(dataset_path, 0.7)

    # DataLoader mit Sequenzlänge und Zukunftsschritten
    dataloader_training = create_dataloader(training_dataset, batch_size, sequence_length, future_steps)
    dataloader_evaluate = create_dataloader(test_dataset, batch_size, sequence_length, future_steps)
    # Modell initialisieren
    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    start_training_time = datetime.now()

    for epoch in range(epochs):
        
        # Training
        training_loss = train(model, dataloader_training, optimizer, loss_fn, future_steps=future_steps) 
        training_losses.append(training_loss)
        #evaluation   
        val_loss = evaluate(model, dataloader_evaluate, loss_fn, future_steps=future_steps)   
        val_losses.append(val_loss)

        if (epoch +1) % 10 == 0: 
            print(f"=> Epoch: {epoch + 1}/{epochs}, Training_loss: {training_loss:.6f}, val_loss: {val_loss:.6f}") 

        if val_loss < best_loss:
            best_loss = val_loss    
            patience = patience_start  # Reset patience counter

        else:
            patience -= 1
            if patience == 0:   

                best_training_loss = training_losses[-start_point]
                best_val_loss = val_losses[-start_point]
                training_time = datetime.now() - start_training_time
                return best_training_loss, best_val_loss, training_time
            
    best_training_loss = training_losses[-1]
    best_val_loss = val_losses[-1]
    training_time = datetime.now() - start_training_time
    return  best_training_loss, best_val_loss, training_time

    


if __name__ == "__main__":

    
    # Hyperparameter
    sequence_length = 60  # Eingabelänge (Anzahl Zeitpunkte)
    future_steps = 10     # Anzahl der vorherzusagenden Schritte
    batch_size = 256
    epochs = 20
    hidden_size = 80
    input_size = 3
    output_size = 3
    learning_rate = 0.0001
    num_layers = 2
    start_point=230
   

    

    dataset_path = "D:\Master_EI\FP\Modeling_Dynamic_Systems\lorenz_attractor_dataset.csv"
    

    # lies confin ein
    config_path = os.path.join( os.getcwd(),"RNN_config.json")

    with open(config_path, 'r') as file:
        config = json.load(file)

    # Hyperparameter

    for system in config:


        dataset = Path(system["dataset"])
        
        
        result_file = os.path.join(os.getcwd(), system["system_name"])

        
        training_results.append(system)
        all_sets_results= list()

        for set in dataset:
            
            one_set_results = list()

            if not set.endswith('.csv'):
                continue
            
            for sequence_len, future_len in zip(system["sequence_length"],system["sequence_out"]):

                training_results = dict()

                trainig_res, val_res, train_time = full_training(system, set, sequence_len, future_len)

                training_results["sequence_len"] = sequence_len
                training_results["future_len"] = future_len
                training_results["trainig_res"] = trainig_res
                training_results["val_res"] = val_res
                training_results["train_time"] = train_time

                one_set_results.append(training_results)

            all_sets_results.append(one_set_results)

        training_results.append(all_sets_results)
            
        with open(result_file, 'w') as json_file:
            json.dump(training_results, json_file)
                
        





    



    



    