"""Data loading and distribution utilities"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DISTRIBUTION_IID, DISTRIBUTION_NON_IID, DISTRIBUTION_LABEL_SKEW


class BearingDataset(Dataset):
    """Bearing anomaly detection dataset from CSV (for autoencoder)"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of features (sensor readings)
            labels: numpy array of labels (not used for autoencoder, but kept for compatibility)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)  # Not used for autoencoder
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return features twice - autoencoder will use input as target
        return self.features[idx], self.labels[idx]


class BearingDatasetLoader:
    """Bearing dataset loader with automatic download and distribution"""
    
    def __init__(self, csv_filename=None, test_size=0.2, random_state=42):
        """
        Args:
            csv_filename: Path to CSV file (if None, downloads from GitHub)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        # Download dataset if not provided
        if csv_filename is None:
            github_urls = [
                "https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_2.csv",
                "https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_1.csv"
            ]
            os.makedirs("processed", exist_ok=True)
            for url in github_urls:
                try:
                    filename = url.split('/')[-1]
                    filename = os.path.join("processed", filename)
                    print(f"Downloading dataset from {url}...")
                    urllib.request.urlretrieve(url, filename)
                    csv_filename = filename
                    print(f"Downloaded to {filename}")
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
            
            if csv_filename is None:
                raise RuntimeError("Failed to download dataset from all URLs")
        
        # Load CSV data
        print(f"Loading data from {csv_filename}...")
        df = pd.read_csv(csv_filename)
        
        # Check if there's a label column (last column or column named 'label', 'class', etc.)
        label_columns = ['label', 'class', 'target', 'y']
        label_col = None
        
        # Check for known label column names
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        # If no label column found, check if last column is non-numeric (likely label)
        if label_col is None:
            last_col = df.columns[-1]
            if df[last_col].dtype == 'object' or df[last_col].dtype == 'string':
                label_col = last_col
        
        # Extract features and labels
        if label_col is not None:
            # Has label column
            X = df.drop(columns=[label_col]).values
            y = df[label_col].values
        else:
            # No label column - create synthetic labels based on clustering or use row groups
            print("  Warning: No label column found. Creating synthetic labels based on data distribution...")
            X = df.values
            
            # Simple approach: divide data into equal groups as classes
            num_samples = len(X)
            samples_per_class = num_samples // 4  # Create 4 classes
            y = np.array([i // samples_per_class for i in range(num_samples)])
            y = np.minimum(y, 3)  # Cap at 4 classes (0,1,2,3)
            unique_labels = np.unique(y)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert labels to integers starting from 0
        if y.dtype == 'object' or y.dtype.kind in ['U', 'S']:
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        else:
            y = y.astype(int)
            unique_labels = np.unique(y)
        
        self.num_features = X.shape[1]
        self.num_classes = len(unique_labels)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.train_dataset = BearingDataset(X_train, y_train)
        self.test_dataset = BearingDataset(X_test, y_test)
        
        print(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.test_dataset)} test")
        print(f"Features: {self.num_features}, Classes: {self.num_classes}")
    
    def distribute_data(self, num_peers, distribution_type=DISTRIBUTION_IID, alpha=0.5, peer_fractions=None):
        """Distribute data among peers
        
        Args:
            num_peers: Number of peers
            distribution_type: 'iid', 'non_iid', or 'label_skew'
            alpha: Dirichlet parameter for non-iid distribution
            peer_fractions: Optional list of fractions (0-1) for each peer's data amount
                          If None, data is split equally. Must sum to <= 1.0
        
        Returns:
            List of (train_subset, test_subset) tuples for each peer
        """
        if distribution_type == DISTRIBUTION_IID:
            return self._distribute_iid(num_peers, peer_fractions)
        elif distribution_type == DISTRIBUTION_NON_IID:
            return self._distribute_non_iid(num_peers, alpha, peer_fractions)
        elif distribution_type == DISTRIBUTION_LABEL_SKEW:
            return self._distribute_label_skew(num_peers, peer_fractions=peer_fractions)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    def _distribute_iid(self, num_peers, peer_fractions=None):
        """IID distribution - random split
        
        Args:
            num_peers: Number of peers
            peer_fractions: Optional list of fractions for each peer's data amount
        """
        train_size = len(self.train_dataset)
        indices = np.random.permutation(train_size)
        
        # Calculate split sizes based on fractions
        if peer_fractions is not None:
            if len(peer_fractions) != num_peers:
                raise ValueError(f"peer_fractions length {len(peer_fractions)} must equal num_peers {num_peers}")
            if sum(peer_fractions) > 1.0:
                raise ValueError(f"peer_fractions sum {sum(peer_fractions)} must be <= 1.0")
            
            split_sizes = [int(train_size * frac) for frac in peer_fractions]
        else:
            split_size = train_size // num_peers
            split_sizes = [split_size] * num_peers
            # Add remainder to last peer
            split_sizes[-1] = train_size - sum(split_sizes[:-1])
        
        peer_data = []
        start_idx = 0
        for i in range(num_peers):
            end_idx = start_idx + split_sizes[i]
            peer_indices = indices[start_idx:end_idx]
            
            train_subset = Subset(self.train_dataset, peer_indices)
            # Use same test set for all peers
            test_subset = self.test_dataset
            
            peer_data.append((train_subset, test_subset))
            start_idx = end_idx
        
        return peer_data
    
    def _distribute_non_iid(self, num_peers, alpha=0.5, peer_fractions=None):
        """Non-IID distribution using Dirichlet distribution
        
        Args:
            num_peers: Number of peers
            alpha: Dirichlet parameter
            peer_fractions: Optional list of fractions for each peer's data amount
        """
        # Get labels
        targets = np.array([self.train_dataset[i][1].item() for i in range(len(self.train_dataset))])
        num_classes = self.num_classes
        
        # Calculate target sizes for each peer
        train_size = len(self.train_dataset)
        if peer_fractions is not None:
            if len(peer_fractions) != num_peers:
                raise ValueError(f"peer_fractions length must equal num_peers")
            if sum(peer_fractions) > 1.0:
                raise ValueError(f"peer_fractions sum must be <= 1.0")
            target_sizes = [int(train_size * frac) for frac in peer_fractions]
        else:
            target_sizes = None
        
        # Use Dirichlet distribution to assign samples to peers
        peer_indices = [[] for _ in range(num_peers)]
        
        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * num_peers)
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            
            # Split class indices according to proportions
            splits = np.split(class_indices, proportions)
            for peer_id, split in enumerate(splits):
                peer_indices[peer_id].extend(split)
        
        # Create subsets (trim to target sizes if specified)
        peer_data = []
        for i, indices in enumerate(peer_indices):
            np.random.shuffle(indices)
            
            # Trim to target size if peer_fractions specified
            if target_sizes is not None and len(indices) > target_sizes[i]:
                indices = indices[:target_sizes[i]]
            
            train_subset = Subset(self.train_dataset, indices)
            test_subset = self.test_dataset
            peer_data.append((train_subset, test_subset))
        
        return peer_data
    
    def _distribute_label_skew(self, num_peers, labels_per_peer=2, peer_fractions=None):
        """Label skew - each peer gets only a few labels
        
        Args:
            num_peers: Number of peers
            labels_per_peer: Number of labels each peer gets
            peer_fractions: Optional list of fractions for each peer's data amount
        """
        targets = np.array([self.train_dataset[i][1].item() for i in range(len(self.train_dataset))])
        num_classes = self.num_classes
        train_size = len(self.train_dataset)
        
        # Calculate target sizes
        if peer_fractions is not None:
            if len(peer_fractions) != num_peers:
                raise ValueError(f"peer_fractions length must equal num_peers")
            if sum(peer_fractions) > 1.0:
                raise ValueError(f"peer_fractions sum must be <= 1.0")
            target_sizes = [int(train_size * frac) for frac in peer_fractions]
        else:
            target_sizes = None
        
        peer_indices = [[] for _ in range(num_peers)]
        
        # Assign labels to peers (round-robin with wrapping)
        for peer_id in range(num_peers):
            assigned_labels = [(peer_id * labels_per_peer + i) % num_classes 
                              for i in range(labels_per_peer)]
            
            for label in assigned_labels:
                label_indices = np.where(targets == label)[0]
                peer_indices[peer_id].extend(label_indices)
        
        # Shuffle and create subsets (trim to target sizes if specified)
        peer_data = []
        for i, indices in enumerate(peer_indices):
            np.random.shuffle(indices)
            
            # Trim to target size if peer_fractions specified
            if target_sizes is not None and len(indices) > target_sizes[i]:
                indices = indices[:target_sizes[i]]
            
            train_subset = Subset(self.train_dataset, indices)
            test_subset = self.test_dataset
            peer_data.append((train_subset, test_subset))
        
        return peer_data


def create_data_loaders(train_subset, test_subset, batch_size=32):
    """Create train and test data loaders"""
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
