"""
RNN character generator
RNN implementation with Dense layers
There is an RNN layer in pytorch, but in this case we will be using
normal Dense layers to demonstrate the difference between
RNN and Normal feedforward networks.
This is a character level generator, which means it will create character by character
You can input any text file and it will generate characters based on that text
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")

class RNN(nn.Module):
    """
    Basic RNN block. This represents a single layer of RNN

    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int ) -> None:
        """
        input_size: Number of features of your input vector
        hidden_size: Number of hidden neurons
        output_size: Number of features of your output vector
        """ 
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.i2h = nn.Linear(input_size, hidden_size, bias= False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        """ 
        Returns computed output and tanh(i2h + h2h)

        Inputs
        ------
        x: Input vector
        hidden_state: Previous hidden state

        Outputs
        -------
        out: Linear output (without activation because of how pythorch works)
        hidden_state: New hidden state matrix
        """

        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(hidden_state + x)
        out = self.h2o(hidden_state)
        return out, hidden_state
    
    def init_zero_hidden(self, batch_size = 1) -> torch.Tensor:
        """ 
        Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """

        return torch.zeros(batch_size, self.hidden_size, requires_grad = False)
    


# Custom Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path):
        """
        Reads CSV file and prepares data for batching.
        Assumes the CSV has columns: Time, X, Y, Z.
        """
        # Load data
        data = pd.read_csv(file_path)
        
        # Drop the time column (assume it's not needed for the model)
        self.data = data.iloc[1:, 1:].values  # Extract X, Y, Z
        
        # Create inputs (X) and targets (Y)
        self.X = self.data[:-1]  # All rows except the last
        self.Y = self.data[1:]   # All rows except the first (shifted)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return a single sample (input, target)
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.Y[index], dtype=torch.float32)


def create_dataloader(file_path, batch_size, shuffle=False):
    """
    Creates a DataLoader from the given file path.
    """
    dataset = TimeSeriesDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# Train Function
def train(model: nn.Module, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module):
    """
    Trains the model for the specified number of epochs
    Input
    -----
    model: RNN model to train
    data: Iterable DataLoader
    optimizer: Optimizer to use for each epoch
    loss_fn: Function to calculate loss
    """
    train_losses = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    print("=> Starting training")

    for epoch in range(epochs):
        
        epoch_losses = list()
        
        for X, Y in data:  # X, Y are batches
            # Skip last batch if it doesn't match the batch_size
            if X.shape[0] != model.batch_size:
                continue

            # Initialize hidden state
            hidden = model.init_zero_hidden(batch_size=model.batch_size)

            # Send tensors to the device
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            output, hidden = model(X, hidden)
            loss = loss_fn(output, Y)

            # Backward pass
            loss.backward()

            # Clip gradients to avoid vanishing/exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)

            # Update parameters
            optimizer.step()

            # Record batch loss
            epoch_losses.append(loss.item())

        # Store epoch loss
        train_losses[epoch] = torch.tensor(epoch_losses).mean().item()
        print(f"=> Epoch: {epoch + 1}/{epochs}, Loss: {train_losses[epoch]}")

    return train_losses


if __name__ == "__main__":
    # Hyperparameter
    file_path = "/home_net/ge36xax/projects/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_attractor_dataset.csv"  # Pfad zur CSV-Datei
    batch_size = 32
    input_size = 3  # X, Y, Z (3 Features)
    hidden_size = 128
    output_size = 3  # X, Y, Z (3 Features)
    learning_rate = 0.001
    epochs = 20

    # DataLoader erstellen
    dataloader = create_dataloader(file_path, batch_size)

    # Modell, Optimizer und Verlustfunktion initialisieren
    model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Modell trainieren
    train_losses = train(model, dataloader, epochs, optimizer, loss_fn)
    
    # Trainingsergebnisse speichern oder auswerten (optional)
    print("Training abgeschlossen.")   
    

        
