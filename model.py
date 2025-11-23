"""Neural network models for DFL experiments"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_INPUT_DIM, MODEL_HIDDEN_DIM, MODEL_OUTPUT_DIM


class BearingAutoencoder(nn.Module):
    """Autoencoder for bearing anomaly detection"""
    
    def __init__(self, input_size: int = 8, latent_size: int = 4, hidden_size: int = 64):
        super(BearingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
    
    def forward(self, x):
        # Handle both batched and unbatched input
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten if needed
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class SimpleNN(nn.Module):
    """Simple feedforward neural network for classification"""
    
    def __init__(self, input_dim=MODEL_INPUT_DIM, hidden_dim=MODEL_HIDDEN_DIM, output_dim=MODEL_OUTPUT_DIM):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Handle both image data (need flatten) and tabular data (already flat)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten for image data
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def init_model(input_dim=None, output_dim=None, device="cpu", model_type="autoencoder"):
    """Initialize a new model instance
    
    Args:
        input_dim: Input dimension (if None, uses default from config)
        output_dim: Output dimension (if None, uses default from config)
        device: Device to place model on
        model_type: Type of model ('autoencoder' or 'classifier')
    """
    if model_type == "autoencoder":
        # For bearing anomaly detection
        if input_dim is None:
            input_dim = 8  # 8 bearing sensors
        model = BearingAutoencoder(input_size=input_dim, hidden_size=64, latent_size=4)
    else:
        # For classification tasks
        if input_dim is None:
            input_dim = MODEL_INPUT_DIM
        if output_dim is None:
            output_dim = MODEL_OUTPUT_DIM
        model = SimpleNN(input_dim=input_dim, output_dim=output_dim)
    
    model.to(device)
    return model


def get_model_parameters(model):
    """Get model parameters as a state dict"""
    return model.state_dict()


def set_model_parameters(model, parameters):
    """Set model parameters from a state dict"""
    model.load_state_dict(parameters)


def get_model_size_bytes(model):
    """Estimate model size in bytes"""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.data.element_size() * param.data.nelement()
    return total_bytes


def average_models(models, weights=None):
    """Average multiple models with optional weights
    
    Args:
        models: List of model state dicts
        weights: Optional list of weights (e.g., number of samples)
    
    Returns:
        Averaged state dict
    """
    if not models:
        return None
    
    if weights is None:
        weights = [1.0] * len(models)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Average parameters
    avg_state = {}
    for key in models[0].keys():
        avg_state[key] = sum(w * model[key] for w, model in zip(weights, models))
    
    return avg_state


def fedprox_aggregate(local_model, neighbor_models, mu=0.01, weights=None):
    """FedProx aggregation with proximal term
    
    Args:
        local_model: Current local model state dict
        neighbor_models: List of neighbor model state dicts
        mu: Proximal term coefficient
        weights: Optional weights for averaging
    
    Returns:
        Aggregated state dict with proximal regularization
    """
    if not neighbor_models:
        return local_model
    
    # First average neighbor models
    avg_neighbor = average_models(neighbor_models, weights)
    
    # Apply proximal term: (1-mu) * avg_neighbor + mu * local_model
    aggregated = {}
    for key in local_model.keys():
        aggregated[key] = (1 - mu) * avg_neighbor[key] + mu * local_model[key]
    
    return aggregated
