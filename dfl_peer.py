"""DFL Peer implementation - core training and aggregation logic"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from model import init_model, get_model_parameters, set_model_parameters, average_models, fedprox_aggregate, get_model_size_bytes
from config import AGGREGATE_AVG, AGGREGATE_FEDPROX


class DFLPeer:
    """Decentralized Federated Learning Peer
    
    Handles local training, evaluation, and model aggregation
    """
    
    def __init__(self, peer_id, train_data, test_data, batch_size=32, 
                 device="cpu", aggregate_method=AGGREGATE_AVG, mu=0.01, learning_rate=0.01,
                 input_dim=None, output_dim=None):
        """Initialize DFL Peer
        
        Args:
            peer_id: Unique peer identifier
            train_data: Training dataset subset
            test_data: Test dataset subset
            batch_size: Batch size for training
            device: Device to run on (cpu/cuda)
            aggregate_method: 'avg' or 'prox'
            mu: FedProx proximal term coefficient
            learning_rate: Learning rate for optimizer
            input_dim: Input dimension for model (auto-detected if None)
            output_dim: Output dimension for model (auto-detected if None)
        """
        self.peer_id = peer_id
        self.device = device
        self.aggregate_method = aggregate_method
        self.mu = mu
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Auto-detect input/output dimensions from data if not provided
        if input_dim is None or output_dim is None:
            sample_data, sample_label = train_data[0]
            if input_dim is None:
                input_dim = sample_data.numel()  # Total number of elements
            if output_dim is None:
                # For autoencoder, output_dim = input_dim (reconstruction)
                # But we keep label detection for backward compatibility
                all_labels = [train_data[i][1].item() for i in range(len(train_data))]
                output_dim = max(all_labels) + 1
        
        # Initialize model (autoencoder for bearing anomaly detection)
        self.model = init_model(input_dim=input_dim, output_dim=output_dim, device=device, model_type="autoencoder")
        self.global_model = None  # Keep reference for FedProx
        
        # Data loaders
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Track number of samples
        self.num_train_samples = len(train_data)
        self.num_test_samples = len(test_data)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # MSE for autoencoder reconstruction
        
        # Training history
        self.train_loss_history = []
        self.eval_loss_history = []
        self.eval_mse_history = []  # MSE instead of accuracy
    
    def train_local(self, epochs=1):
        """Train model locally for specified epochs
        
        Args:
            epochs: Number of local training epochs
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                # For autoencoder, input = target (reconstruction)
                target = data  # Reconstruct the input itself
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Add proximal term if using FedProx and global model exists
                if self.aggregate_method == AGGREGATE_FEDPROX and self.global_model is not None:
                    proximal_term = 0.0
                    for w, w_g in zip(self.model.parameters(), self.global_model.parameters()):
                        proximal_term += ((w - w_g) ** 2).sum()
                    loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(self.train_loader)
        
        avg_loss = total_loss / epochs
        self.train_loss_history.append(avg_loss)
        return avg_loss
    
    def evaluate_local(self):
        """Evaluate model on local test data
        
        Returns:
            Tuple of (avg_loss, mse)
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                # For autoencoder, reconstruct the input
                target = data
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate MSE per sample
                mse = torch.mean((output - target) ** 2)
                total_mse += mse.item()
                num_batches += 1
        
        avg_loss = total_loss / len(self.test_loader)
        avg_mse = total_mse / len(self.test_loader)
        
        self.eval_loss_history.append(avg_loss)
        self.eval_mse_history.append(avg_mse)
        
        return avg_loss, avg_mse
    
    def aggregate_models(self, neighbor_models):
        """Aggregate models from neighbors
        
        Args:
            neighbor_models: List of dicts containing:
                - 'parameters': model state dict
                - 'num_samples': number of training samples
        """
        if not neighbor_models:
            return
        
        # Extract parameters and weights
        model_params = [nm['parameters'] for nm in neighbor_models]
        weights = [nm['num_samples'] for nm in neighbor_models]
        
        # Get current model parameters
        current_params = get_model_parameters(self.model)
        
        # Aggregate based on method
        if self.aggregate_method == AGGREGATE_AVG:
            # Simple weighted averaging
            aggregated = average_models(model_params, weights)
        elif self.aggregate_method == AGGREGATE_FEDPROX:
            # FedProx with proximal term
            aggregated = fedprox_aggregate(current_params, model_params, self.mu, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregate_method}")
        
        # Update model
        set_model_parameters(self.model, aggregated)
        
        # Keep global model reference for FedProx
        if self.aggregate_method == AGGREGATE_FEDPROX:
            self.global_model = copy.deepcopy(self.model)
    
    def get_model_info(self):
        """Get model information for sharing
        
        Returns:
            Dict with model parameters and metadata
        """
        return {
            'parameters': get_model_parameters(self.model),
            'num_samples': self.num_train_samples,
            'peer_id': self.peer_id
        }
    
    def set_aggregate_method(self, method, mu=None):
        """Update aggregation method
        
        Args:
            method: 'avg' or 'prox'
            mu: FedProx coefficient (optional)
        """
        self.aggregate_method = method
        if mu is not None:
            self.mu = mu
    
    def get_model_size_bytes(self):
        """Get model size in bytes"""
        return get_model_size_bytes(self.model)
