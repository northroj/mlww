import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class NeutronDataset(Dataset):
    def __init__(self, input_filepath, output_filepath, start_idx=0, end_idx=None):
        self.input_data, self.output_data = self.load_data(input_filepath, output_filepath, start_idx, end_idx)
    
    def load_data(self, input_filepath, output_filepath, start_idx, end_idx):
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

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

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
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_errors) + 1), self.train_errors, label='Training Error (%)')
        plt.plot(range(1, len(self.val_errors) + 1), self.val_errors, label='Validation Error (%)')
        plt.xlabel('Epochs')
        plt.ylabel('Error (%)')
        plt.title('Training and Validation Error Over Epochs')
        plt.legend()
        plt.show()
    
    def save_model(self, path, model_name):
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
        input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0).permute(0, 4, 1, 2, 3)  # Reshape for CNN
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return output_tensor.squeeze(0).cpu().numpy()

    def compare_flux(self, input_filepath, output_filepath, case_number=0):
        case_key = f"case_{case_number}"
        with h5py.File(input_filepath, 'r') as f:
            if case_key not in f:
                raise ValueError(f"Case {case_number} not found in input file.")
            input_data = f[case_key][()]
        with h5py.File(output_filepath, 'r') as f:
            if case_key not in f:
                raise ValueError(f"Case {case_number} not found in output file.")
            actual_output = f[case_key][()]
        
        predicted_output = self.predict_flux(input_data)
        return actual_output, predicted_output