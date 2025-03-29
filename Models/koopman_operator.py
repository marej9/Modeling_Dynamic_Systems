import importlib
import kooplearn
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from time import perf_counter
from kooplearn.data import traj_to_contexts
from kooplearn.models import Kernel, NystroemKernel
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


def split_and_normalize_dataset(file_path, train_ratio=0.85):

    data = pd.read_csv(file_path).iloc[1:, 1:].values.astype(np.float32)
    
    dataset_size = len(data) 

    train_size = int(train_ratio * dataset_size)

    train_data = data[:train_size ]
    test_data = data[train_size:]

    #compute mean and standard deviation over features (columns)
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)

    # Apply normalization
    normalized_train_data = (train_data - train_mean) / train_std
    normalized_test_data = (test_data - train_mean) / train_std

    return normalized_train_data, normalized_test_data

def full_pred(model, dataset,future_steps):
    prediction = dataset[:,:1,:]
    #prediction = model.predict(dataset, t=1, predict_observables=False, reencode_every= 1)
    for i in range(1, future_steps):   
        pred = model.predict(dataset, t=i, predict_observables=False, reencode_every= i)
        prediction = np.concatenate((prediction, pred), axis=1)
        #prediction = np.append(prediction,pred, axis=1)
    return prediction

if __name__ == "__main__":
    
    reduced_rank = False
    rank = 200
    num_centers = 250
    tikhonov_reg = 1e-8

    dataset_path = "/home_net/ge36xax/projects/Modeling_Dynamic_Systems/DynSys_and_DataSets/lorenz_system/lorenz_system_data.csv"
    sequence_lengths = [0.5, 1.0, 2.0, 5.0]
    sequence_lengths = [int(x * 50) for x in sequence_lengths]
    #sequence_lengths = [int(x * 50) for x in sequence_lengths]
    results = {
        "hyperparameters": {
            "reduced_rank" : reduced_rank,
            "rank" : rank,
            "num_centers" : num_centers,
            "tikhonov_reg" : tikhonov_reg
        },
        "training_results": []
    }

    for sequence in sequence_lengths:

        train, test = split_and_normalize_dataset(dataset_path)
        dataset = {
            "train": torch.tensor(train, device=device),
            "test": torch.tensor(test, device=device)
        }
        # From trajectories to context windows
        contexts_train = {k: traj_to_contexts(v.cpu().numpy()) for k, v in dataset.items()} # Converting the trajectories to contexts
        contexts_test = {k: traj_to_contexts(v.cpu().numpy(), context_window_len=sequence) for k, v in dataset.items()} # Converting the trajectories to contexts    

        # Instantiating the RBF kernel and its length scale as the median of the pairwise distances of the dataset
        data_pdist = pdist(dataset['train'].cpu().numpy())  # Kopiere den Tensor auf die CPU und konvertiere ihn in ein NumPy-Array
        kernel = RBF(length_scale=np.quantile(data_pdist, 0.5)/2)
        model = Kernel(kernel=kernel, reduced_rank=reduced_rank, svd_solver='randomized', tikhonov_reg=tikhonov_reg, rank=rank, rng_seed=42)
            # Kopiere den Tensor zurück auf die GPU
        dataset['train'] = dataset['train'].to(device)
        dataset['test'] = dataset['test'].to(device)
        # Training
        start_time = time.time()

        model, fit_time = timer(model.fit)(contexts_train['train'])
        sequence = contexts_test["test"].shape[1]
        X_pred = full_pred(model, contexts_test["test"][:,:2,:], sequence)
        #X_pred = full_pred(model, contexts_test["test"], sequence)
        X_true = contexts_test['test']

        # train loss
        mse_loss = np.mean((X_pred - X_true)**2)
        end_time = time.time()

        training_result = {
            "mse_loss": mse_loss,
            "sequence_length": sequence,
            "just_train_time": fit_time,
            "training_time": end_time - start_time 
        }
        
        results["training_results"].append(training_result)
        
        # Save results to JSON file in the same folder as the dataset file
        folder_path = os.path.dirname(dataset_path)
        json_file_path = os.path.join(folder_path, 'koopman_lorenz_gpu.json')
        
        with open(json_file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)










