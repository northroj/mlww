import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import mlww.generate as generate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class NeutronDataset(Dataset):
    def __init__(self, input_filepath, output_filepath, start_idx=0, end_idx=None):
        self.input_data, self.output_data = self.load_data(input_filepath, output_filepath, start_idx, end_idx)
    
    def load_data(self, input_filepath, output_filepath, start_idx, end_idx):
        """
        Load the data from the input and output hdf5 files
        """
        with h5py.File(output_filepath, 'r') as f:
            case_numbers = sorted([int(key.split('_')[1]) for key in f.keys()])
            if end_idx is None:
                end_idx = case_numbers[-1]
        
        with h5py.File(input_filepath, 'r') as f:
            inputs = [f[f'case_{i}'][()] for i in range(start_idx, end_idx + 1) if f'case_{i}' in f]
            inputs = np.array(inputs)
        with h5py.File(output_filepath, 'r') as f:
            outputs = [f[f'case_{i}'][()] for i in range(start_idx, end_idx + 1) if f'case_{i}' in f]
            outputs = np.array(outputs)

        return inputs, outputs
    
    def normalize_source(self):
        """
        Normalize the source strength (index 2 in input parameters) for each case
        so that the sum over all (x, y, z) cells is 1.
        """
        source_strengths = self.input_data[..., 2]  # Extract source strength values
        total_source = np.sum(source_strengths, axis=(1, 2, 3), keepdims=True)  # Sum over x, y, z
        total_source[total_source == 0] = 1  # Avoid division by zero
        self.input_data[..., 2] /= total_source  # Normalize

    def normalize_results(self):
        """
        Normalize the output data for each case so that the maximum value in each case is 1.
        """
        max_values = np.max(self.output_data, axis=(1, 2, 3), keepdims=True)  # Max over x, y, z
        max_values[max_values == 0] = 1  # Avoid division by zero
        self.output_data /= max_values  # Normalize

    def normalize_xs(self):
        """
        Normalize the capture cross section (index 0) for each case so that the maximum in each case is 1.
        The scattering cross section (index 1) is scaled by the same factor to maintain its ratio with capture.
        """
        max_capture = np.max(self.input_data[..., 0], axis=(1, 2, 3), keepdims=True)  # Max over x, y, z
        max_capture[max_capture == 0] = 1  # Avoid division by zero
        self.input_data[..., 0] /= max_capture  # Normalize capture cross section
        self.input_data[..., 1] /= max_capture  # Scale scatter cross section by the same factor
    
    def to_torch(self):
        """
        Convert input and output data to torch tensors
        """
        self.input_data = torch.tensor(self.input_data, dtype=torch.float32)
        self.output_data = torch.tensor(self.output_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.conv_layers(x).squeeze(1)  # Remove channel dim

class TrainCNN:
    def __init__(self, train_dataset, val_dataset, batch_size=16, max_epochs=100, lr=0.001, tolerance=1e-6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.model = CNNModel().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.train_errors = []
        self.val_errors = []
    
    def train(self):
        """ Train the machine learning model
        """
        self.model.train()
        prev_val_loss = float('inf')
        for epoch in range(self.max_epochs):
            total_error = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.permute(0, 4, 1, 2, 3)  # Change shape to (batch, channels, x, y, z)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                error = torch.abs(outputs - targets) / (targets + 1e-8) * 100
                total_error += error.mean().item()
            
            val_error = self.validate()
            self.train_errors.append(total_error / len(self.train_loader))
            self.val_errors.append(val_error)
            print(f"Epoch {epoch+1}/{self.max_epochs}, Train Error: {self.train_errors[-1]:.2f}%, Val Error: {self.val_errors[-1]:.2f}%")
            
            if abs(prev_val_loss - val_error) < self.tolerance:
                print("Validation error converged, stopping training.")
                break
            prev_val_loss = val_error
    
    def validate(self):
        self.model.eval()
        total_error = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.permute(0, 4, 1, 2, 3)
                outputs = self.model(inputs)
                error = torch.abs(outputs - targets) / (targets + 1e-8) * 100
                total_error += error.mean().item()
        return total_error / len(self.val_loader)
    
    def get_train_errors(self):
        return self.train_errors
    
    def get_val_errors(self):
        return self.val_errors
    
    def plot_errors(self):
        """ Plot the training and validation accuracy as a function of epoch
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_errors) + 1), self.train_errors, label='Training Error (%)')
        plt.plot(range(1, len(self.val_errors) + 1), self.val_errors, label='Validation Error (%)')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Error (%)')
        plt.title('Training and Validation Mean Error Over Epochs')
        plt.legend()
        plt.show()
    
    def save_model(self, path, model_name):
        """
        Save the model to a .pt file

        Parameters
        ----------
        path : string
            path to the directory to store the trained model file
        model_name : string
            name of the model
        """
        model_filepath = os.path.join(path, model_name + ".pt")
        if os.path.exists(model_filepath):
            os.remove(model_filepath)
        torch.save(self.model.state_dict(), model_filepath)
        print(f"Model saved to {model_filepath}")

class ModelLoader:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict_flux(self, input_grid):
        """
        Predict the flux for a set if input data
        """
        input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0).permute(0, 4, 1, 2, 3)  # Reshape for CNN
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return output_tensor.squeeze(0).cpu().numpy()

    def compare_flux(self, input_filepath, output_filepath, case_number=0, normalize_source = True, normalize_results = True, normalize_xs = True):
        """
        Produce the actual flux and predicted flux for a given case in the training data

        Parameters
        ----------
        input_filepath : string
            path to the input hdf5 data
        output_filepath : string
            path to the output hdf5 data
        case_number : int
            case number to test

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            actual output of the MC/DC simulation and predicted output of the machine learning model
        """
        data = NeutronDataset(input_filepath, output_filepath, start_idx=case_number, end_idx=case_number)
        if normalize_source:
            data.normalize_source()
        if normalize_results:
            data.normalize_results()
        if normalize_xs:
            data.normalize_xs()
        actual_output = data.output_data[0,:,:,:]
        input_data = data.input_data[0,:,:,:,:]
        
        predicted_output = self.predict_flux(input_data)
        return actual_output, predicted_output
    
    def plot_compare_flux(self, actual_output, predicted_output, z_slice=0):
        output = np.expand_dims(actual_output, axis=-1)
        predicted = np.expand_dims(predicted_output, axis=-1)
        sim_plot = generate.RandomGeneration(output.shape[0],output.shape[1],output.shape[2])
        pred_plot = generate.RandomGeneration(predicted.shape[0],predicted.shape[1],predicted.shape[2])
        sim_plot.plot_2d_grid(output, data_type=0, z_slice=z_slice, title="Simulated Neutron Flux")
        pred_plot.plot_2d_grid(predicted, data_type=0, z_slice=z_slice, title="Predicted Neutron Flux")