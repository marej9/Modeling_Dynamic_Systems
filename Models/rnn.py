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
    def __init__(self, file_path, sequence_length, future_steps):
        """
        Prepares data for sequence-based training with future steps targets
        """
        
        # Drop the time column (assume it's not needed for the model)
        data = pd.read_csv(file_path).iloc[1:, 1:].values  # Load X, Y, Z
        
        #convert to Pytorch tensor
        data = torch.tensor(data,dtype=torch.float32)

        #compute mean and standard deviation over features (columns)
        mean = data.mean(dim=0)
        std = data.std(dim=0)

         # Apply normalization
        data = (data - mean) / std
        

        self.sequence_length = sequence_length
        self.future_steps = future_steps

        self.X, self.Y = [], []
       

        for i in range(len(data) - sequence_length - future_steps + 1):
            self.X.append(data[i:i+sequence_length])
            self.Y.append(data[i+sequence_length: i+sequence_length+future_steps])     

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


# Train Function
def train(model, data, epochs, optimizer, loss_fn, future_steps):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_losses = {}
    losses_train = list()

    print("=> Starting training")

    for epoch in range(epochs):
        
        epoch_losses = list()
        
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

            epoch_losses.append(loss.item())

            
        # Store epoch loss
        train_losses[epoch] = np.mean(epoch_losses) # mean value for all batches in one epoch
        if (epoch +1) % 10 == 0: 
            print(f"=> Epoch: {epoch + 1}/{epochs}, Loss: {train_losses[epoch]:.4f}") 
            losses_train.append(train_losses[epoch])

    return losses_train

"""def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")"""

def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def plot_all_features(model, file_path, start_point, future_steps, sequence_length):
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



if __name__ == "__main__":

    
    # lies confin ein
    config_path = os.path.join( os.getcwd() ,"RNN_config.json")

    with open(config_path, 'r') as file:
        config = json.load(file)

    # Hyperparameter
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    hidden_size = config["hidden_size"]
    input_size = config["input_size"]
    output_size = config["output_size"]
    learning_rate = config["learning_rate"]
    num_layers = config["num_layers"]
    start_point = config["start_point"]
    end_point = config["end_point"]

    for system in config:
        training_dataset = Path(system["training_dataset"])
        validation_dataset = Path(system["validation_dataset"])
        result_file = os.path.join(os.getcwd(), system["system_name"])
        training_results = list()
        training_results.append(system)

        for sequence_length in system["sequence_length"]:
            
            for future_steps in system["sequence_out"]:
                    training = dict()
                    training["sequence_length"] = sequence_length
                    training["future_steps"] = future_steps


                    # DataLoader mit Sequenzlänge und Zukunftsschritten
                    dataloader = create_dataloader(training_dataset, batch_size, sequence_length, future_steps)

                    # Modell initialisieren
                    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, num_classes=output_size).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    loss_fn = nn.MSELoss()

                    # Training
                    start_training_time = datetime.now().time()
                    training_losses = train(model, dataloader, epochs, optimizer, loss_fn, future_steps=future_steps)  
                    training_time = datetime.now().time() - start_training_time      
                    print("training abgeschlossen")
                    training["training_losses"] = training_losses
                    training["training_time"] = training_time


                    # Evaluierung durchführen
                    dataloader = create_dataloader(validation_dataset, batch_size, sequence_length, future_steps)
                    evaluation_loss = evaluate(model, dataloader, loss_fn, future_steps=future_steps)

                    training["evaluation_loss"] = training_time

                    training_results.append(training)

        


 


    """
    # Hyperparameter
    sequence_length = 60  # Eingabelänge (Anzahl Zeitpunkte)
    future_steps = 10     # Anzahl der vorherzusagenden Schritte
    batch_size = 256
    epochs = 500
    hidden_size = 80
    input_size = 3
    output_size = 3
    learning_rate = 0.0001
    num_layers = 2
    start_point=230
    # DataLoader mit Sequenzlänge und Zukunftsschritten
    dataloader = create_dataloader(file_path, batch_size, sequence_length, future_steps)

    # Modell initialisieren
    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, num_classes=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    # Training
    train_losses = train(model, dataloader, epochs, optimizer, loss_fn, future_steps=future_steps)
    save_model(model, 'rnn_norm_model_future_steps.pth')
    print("training abgeschlossen")
    
    """
    """  
    model_path = "rnn_norm_model_future_steps.pth"  # Pfad zu deinem Modell
    file_path = "lorenz_attractor_dataset_test.csv"  # Pfad zu deinem Test-Dataset
    sequence_length = 60  # Eingabelänge
    future_steps = 10     # Anzahl der vorherzusagenden Schritte
    batch_size = 1        # Batchgröße (für Evaluation irrelevant, da nur eine Sequenz gleichzeitig verarbeitet wird)

    # Modell laden
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    loss_fn = nn.MSELoss()
    # Evaluierung durchführen
    dataloader = create_dataloader(file_path, batch_size, sequence_length, future_steps)
    evaluate(model, dataloader, loss_fn, future_steps=future_steps)

    """
    model_path = os.path.join( os.getcwd() ,"rnn_norm_model_future_steps.pth")  # Pfad zu deinem Modell
    file_path = os.path.join( os.getcwd() ,"lorenz_attractor_dataset_test.csv")  # Pfad zu deinem Test-Dataset
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location = torch.device("cpu") , weights_only = True))
    plot_all_features(model, file_path, start_point, future_steps, sequence_length) 
    