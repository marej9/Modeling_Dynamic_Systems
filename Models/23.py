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
        self.sequence_length = sequence_length
        self.future_steps = future_steps

        self.X, self.Y = [], []
       
        for i in range(len(data) - sequence_length - future_steps + 1):
            self.X.append(data[i:i+sequence_length])
            self.Y.append(data[i+sequence_length: i+sequence_length+future_steps])     

        # Create inputs (X) and targets (Y)
        self.X = np.array(self.X)  # All rows except the last
        self.Y = np.array(self.Y)   # All rows except the first (shifted)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return a single sample (input, target)
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.Y[index], dtype=torch.float32)
        

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
            

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Store epoch loss
        train_losses[epoch] = np.mean(epoch_losses)
        print(f"=> Epoch: {epoch + 1}/{epochs}, Loss: {train_losses[epoch]:.4f}") 

    return train_losses

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")

def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name, weights_only=True))
    model.eval()
    print(f"Model loaded from {file_name}")
    return model

def evaluate(model, dataloader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            output, _ = model(X, model.init_zero_hidden(batch_size=X.shape[0]).to(device))

            loss = loss_fn(output, Y)
            total_loss += loss.item() * X.shape[0]  # Weighted by batch size
            total_samples += X.shape[0]

    avg_loss = total_loss / total_samples
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    # Hyperparameter
    file_path = "/home_net/ge36xax/projects/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_attractor_dataset.csv"

    sequence_length = 20  # Eingabelänge (Anzahl Zeitpunkte)
    future_steps = 10     # Anzahl der vorherzusagenden Schritte
    batch_size = 196
    epochs = 300
    hidden_size = 16
    input_size = 3
    output_size = 3
    learning_rate = 0.0001
    # DataLoader mit Sequenzlänge und Zukunftsschritten
    dataloader = create_dataloader(file_path, batch_size, sequence_length, future_steps)

    # Modell initialisieren
    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, num_classes=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    # Training
    train_losses = train(model, dataloader, epochs, optimizer, loss_fn, future_steps=future_steps)
    save_model(model, 'rnn_model_future_steps.pth')
    print("training abgeschlossen")
    
    """save_model(model, 'rnn_model.pth')

    # Modell nach dem Training laden und evaluieren
    loaded_model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    loaded_model = load_model(loaded_model, 'rnn_model.pth')

    # Testen und Evaluieren (optional, ein Testdatensatz muss vorhanden sein)
    test_file_path = 'D:/Master_EI/FP/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_attractor_dataset_test.csv'  # Beispiel Testdatensatz
    test_dataloader = create_dataloader(test_file_path, batch_size)
    evaluate(loaded_model, test_dataloader, loss_fn)

    print("Training abgeschlossen und Modell evaluiert.")"""

        
