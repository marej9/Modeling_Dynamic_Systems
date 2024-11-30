"""
hier wird rnn mit class torch.nn.RNN implementiert
"""
from pathlib import Path
import torch  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

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

    return train_losses

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")

def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate(model, data, epochs, loss_fn, future_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_losses = {}
    
    print("=> Starting evaluating")

    all_losses = list()
        
    for X, Y in data:  # X, Y are batches
        

        # Send tensors to the device
        X, Y = X.to(device), Y.to(device)

        # Clear gradients
        

        prediction = model(X, future_steps = future_steps)
        loss = loss_fn(prediction, Y)
        # prediction of shape (batch_size, future_steps, output_size=features)
        # Y of shape (batch_size, future_steps, output_size=features)
        
        all_losses.append(loss.item())


        
    # Store epoch loss
    loss = np.mean(all_losses) # mean value for all batches in one epoch
    print(f"==> loss:{loss}")


def evaluate_and_plot(model, file_path, sequence_length, future_steps, start_index=0):
    """data = pd.read_csv(file_path).iloc[1:, 1:].values  # Load data
    data = torch.tensor(data, dtype=torch.float32)

    # Normalize data
    mean = data.mean(dim=0) 
    std = data.std(dim=0) 
    data = (data - mean) / std

    # Get a sequence of data for prediction (choose a starting index)
    input_seq = data[start_index:start_index+sequence_length].unsqueeze(0).to(device)
    
    # Predict future steps
    with torch.no_grad():
        predicted_output = model(input_seq, future_steps=future_steps)
    
    # Denormalize predictions
    predicted_output = (predicted_output.squeeze(0) * std + mean).cpu()

    # Get true values for comparison
    target_seq = data[start_index+sequence_length:start_index+sequence_length+future_steps].cpu().numpy()
    target_seq = target_seq * std + mean

    # Plot results
    plt.figure(figsize=(15, 10))
    for i in range(predicted_output.shape[-1]):
        plt.subplot(predicted_output.shape[-1], 1, i + 1)
        plt.plot(predicted_output[:, i], label=f'Predicted Feature {i + 1}', linestyle='--')
        plt.plot(target_seq[:, i], label=f'Actual Feature {i + 1}')
        plt.xlabel('Time Steps')
        plt.ylabel(f'Feature {i + 1}')
        plt.legend()

    plt.tight_layout()
    plt.show()"""

if __name__ == "__main__":
    # Hyperparameter
    file_path = "/home_net/ge36xax/projects/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_attractor_dataset.csv"

    sequence_length = 60  # Eingabelänge (Anzahl Zeitpunkte)
    future_steps = 10     # Anzahl der vorherzusagenden Schritte
    batch_size = 256
    epochs = 500
    hidden_size = 80
    input_size = 3
    output_size = 3
    learning_rate = 0.0001
    num_layers = 2
    """
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
    
    model_path = "rnn_norm_model_future_steps.pth"  # Pfad zu deinem Modell
    file_path = "lorenz_attractor_dataset_test.csv"  # Pfad zu deinem Test-Dataset
    sequence_length = 60  # Eingabelänge
    future_steps = 10     # Anzahl der vorherzusagenden Schritte
    batch_size = 1        # Batchgröße (für Evaluation irrelevant, da nur eine Sequenz gleichzeitig verarbeitet wird)

    # Modell laden
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modell in den Evaluierungsmodus versetzen
    loss_fn = nn.MSELoss()
    # Evaluierung durchführen
    dataloader = create_dataloader(file_path, batch_size, sequence_length, future_steps)
    avg_loss = evaluate(model, dataloader, epochs, loss_fn, future_steps=future_steps)