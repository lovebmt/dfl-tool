"""
Decentralized Federated Learning (DFL) for Bearing Anomaly Detection
=====================================================================

Architecture:
- No central server
- Peer-to-Peer (P2P) communication
- Ring topology: Each peer connects to 2 neighbors (left & right)
- Local aggregation: Each peer aggregates models from its neighbors
- Asynchronous training rounds

Key differences from Centralized FL:
- Communication: Peer ↔ Peer (not Client ↔ Server)
- Aggregation: At each peer (not at central server)
- Topology: Ring (not Star)
- No single point of failure
"""

import os
import math
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Disable Ray/Flower since we're doing pure P2P without simulation framework
os.makedirs("reports_dfl", exist_ok=True)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

NUM_PEERS = 10
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DEVICE = "cpu"
VERBOSE = False  # Set to True for detailed logs

# ============================================================================
# DATA LOADING & MODEL (reuse from centralized version)
# ============================================================================

csv_filename = None

if csv_filename is None:
    local_paths = [
        "processed/bearing_merged_2.csv",
        "processed/bearing_merged_1.csv",
    ]
    for path in local_paths:
        if os.path.exists(path):
            csv_filename = path
            break

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
            urllib.request.urlretrieve(url, filename)
            csv_filename = filename
            break
        except Exception as e:
            continue

class BearingAutoencoder(nn.Module):
    def __init__(self, input_size: int = 8, latent_size: int = 4, hidden_size: int = 16):
        super().__init__()
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
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class BearingDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return {"x": x, "y": x}

def _split_partitions_balanced(values: np.ndarray, partition_id: int, num_partitions: int):
    N = values.shape[0]
    part_size = math.ceil(N / num_partitions)
    start = partition_id * part_size
    end = min(start + part_size, N)
    if start >= N:
        raise RuntimeError(f"partition_id={partition_id} exceeds data size")
    return values[start:end]

def _split_partitions_imbalanced(values: np.ndarray, partition_id: int, num_partitions: int):
    N = values.shape[0]
    if num_partitions == 10:
        ratios = [0.30, 0.05, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.1, 0.01]
    elif num_partitions == 5:
        ratios = [0.35, 0.25, 0.20, 0.12, 0.08]
    elif num_partitions == 3:
        ratios = [0.50, 0.30, 0.20]
    else:
        ratios = [2 ** (-i) for i in range(num_partitions)]
        ratios = [r / sum(ratios) for r in ratios]
    sizes = [int(N * r) for r in ratios]
    sizes[-1] = N - sum(sizes[:-1])
    start = sum(sizes[:partition_id])
    end = start + sizes[partition_id]
    return values[start:end]

def load_data(partition_id: int, num_partitions: int, batch_size: int = BATCH_SIZE, 
              csv_path: str = None, data_distribution: str = "balanced"):
    if csv_path is None:
        csv_path = csv_filename
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 8:
        raise ValueError(f"CSV has {num_df.shape[1]} numeric columns, expected >= 8")
    num_df = num_df.iloc[:, :8]
    num_df.columns = ['B1_a', 'B1_b', 'B2_a', 'B2_b', 'B3_a', 'B3_b', 'B4_a', 'B4_b']
    num_df = num_df.round(4)
    values = num_df.values
    N = values.shape[0]
    if N == 0:
        raise RuntimeError("CSV has no data")
    if data_distribution == "balanced":
        values_part = _split_partitions_balanced(values, partition_id, num_partitions)
    elif data_distribution == "imbalanced":
        values_part = _split_partitions_imbalanced(values, partition_id, num_partitions)
    else:
        raise ValueError(f"Unknown distribution: {data_distribution}")
    n_train = int(len(values_part) * 0.8)
    train_vals = values_part[:n_train]
    test_vals = values_part[n_train:]
    train_dataset = BearingDataset(train_vals)
    test_dataset = BearingDataset(test_vals)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def train_model(net, trainloader, epochs, device, lr=0.001):
    net.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    avg_loss = epoch_loss / len(trainloader)
    return avg_loss

def test_model(net, testloader, device):
    net.to(device)
    criterion = torch.nn.MSELoss()
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            output = net(x)
            loss = criterion(output, y)
            test_loss += loss.item()
    avg_loss = test_loss / len(testloader)
    return avg_loss

# ============================================================================
# DECENTRALIZED FL: RING TOPOLOGY & P2P COMMUNICATION
# ============================================================================

@dataclass
class PeerModel:
    """Container for model parameters and metadata"""
    parameters: List[np.ndarray]
    peer_id: int
    round_num: int
    train_loss: float = 0.0
    eval_loss: float = 0.0
    num_samples: int = 0

class RingTopology:
    """Manages ring topology: each peer has left and right neighbors"""
    def __init__(self, num_peers: int):
        self.num_peers = num_peers
        self.peers = list(range(num_peers))
    
    def get_left_neighbor(self, peer_id: int) -> int:
        """Get left neighbor in ring"""
        return (peer_id - 1) % self.num_peers
    
    def get_right_neighbor(self, peer_id: int) -> int:
        """Get right neighbor in ring"""
        return (peer_id + 1) % self.num_peers
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get both neighbors for a peer"""
        return [self.get_left_neighbor(peer_id), self.get_right_neighbor(peer_id)]

class DFLPeer:
    """
    Decentralized Federated Learning Peer
    
    Each peer:
    1. Trains on local data
    2. Shares model with neighbors in ring
    3. Receives models from neighbors
    4. Aggregates received models locally (average)
    5. Repeats without central coordination
    """
    
    def __init__(
        self,
        peer_id: int,
        num_peers: int,
        topology: RingTopology,
        local_epochs: int = 1,
        lr: float = 0.001,
        data_distribution: str = "balanced",
        device: str = "cpu",
        verbose: bool = False
    ):
        self.peer_id = peer_id
        self.num_peers = num_peers
        self.topology = topology
        self.local_epochs = local_epochs
        self.lr = lr
        self.data_distribution = data_distribution
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Local model
        self.model = BearingAutoencoder()
        self.model.to(self.device)
        
        # Load local data partition
        self.trainloader, self.testloader = load_data(
            partition_id=self.peer_id,
            num_partitions=self.num_peers,
            batch_size=BATCH_SIZE,
            data_distribution=self.data_distribution,
        )
        
        # Storage for received models from neighbors
        self.received_models: Dict[int, PeerModel] = {}
        
        # History tracking
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.current_round = 0
        
        if self.verbose:
            print(f"[Peer {self.peer_id}] Initialized with {len(self.trainloader.dataset)} training samples")
            print(f"[Peer {self.peer_id}] Neighbors: {self.topology.get_neighbors(self.peer_id)}")
    
    def get_parameters(self) -> List[np.ndarray]:
        """Extract model parameters as numpy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def train_local(self) -> float:
        """Train model on local data"""
        train_loss = train_model(
            self.model, 
            self.trainloader, 
            epochs=self.local_epochs,
            device=self.device,
            lr=self.lr
        )
        self.train_losses.append(train_loss)
        return train_loss
    
    def evaluate_local(self) -> float:
        """Evaluate model on local test data"""
        eval_loss = test_model(self.model, self.testloader, device=self.device)
        self.eval_losses.append(eval_loss)
        return eval_loss
    
    def create_model_message(self, train_loss: float = 0.0, eval_loss: float = 0.0) -> PeerModel:
        """Create message containing model to share with neighbors"""
        return PeerModel(
            parameters=self.get_parameters(),
            peer_id=self.peer_id,
            round_num=self.current_round,
            train_loss=train_loss,
            eval_loss=eval_loss,
            num_samples=len(self.trainloader.dataset)
        )
    
    def receive_model(self, peer_model: PeerModel):
        """Receive model from a neighbor"""
        self.received_models[peer_model.peer_id] = peer_model
        if self.verbose:
            print(f"[Peer {self.peer_id}] Received model from Peer {peer_model.peer_id} (Round {peer_model.round_num})")
    
    def aggregate_models(self):
        """
        Aggregate own model with received neighbor models
        Strategy: Simple average (FedAvg-style but decentralized)
        """
        if len(self.received_models) == 0:
            if self.verbose:
                print(f"[Peer {self.peer_id}] No neighbor models to aggregate")
            return
        
        # Include own model in aggregation
        all_models = [self.create_model_message()] + list(self.received_models.values())
        
        # Simple average (equal weights)
        if self.verbose:
            print(f"[Peer {self.peer_id}] Aggregating {len(all_models)} models (self + neighbors)")
        
        aggregated_params = []
        for layer_idx in range(len(all_models[0].parameters)):
            layer_params = [model.parameters[layer_idx] for model in all_models]
            avg_layer = np.mean(layer_params, axis=0)
            aggregated_params.append(avg_layer)
        
        # Update local model with aggregated parameters
        self.set_parameters(aggregated_params)
        
        # Clear received models for next round
        self.received_models.clear()
    
    def run_round(self) -> Tuple[float, float]:
        """
        Execute one round of DFL:
        1. Train on local data
        2. Evaluate
        3. Prepare model for sharing (done by coordinator)
        """
        if self.verbose:
            print(f"[Peer {self.peer_id}] ========== Round {self.current_round} ==========")
        
        # Train locally
        train_loss = self.train_local()
        if self.verbose:
            print(f"[Peer {self.peer_id}] Training loss: {train_loss:.6f}")
        
        # Evaluate locally
        eval_loss = self.evaluate_local()
        if self.verbose:
            print(f"[Peer {self.peer_id}] Evaluation loss: {eval_loss:.6f}")
        
        self.current_round += 1
        
        return train_loss, eval_loss

class DFLCoordinator:
    """
    Coordinator for DFL simulation (NOT a central server)
    Only used to orchestrate P2P communication in simulation
    In real deployment, this would be replaced by actual P2P network protocol
    """
    
    def __init__(self, peers: List[DFLPeer], topology: RingTopology, verbose: bool = False):
        self.peers = peers
        self.topology = topology
        self.num_peers = len(peers)
        self.verbose = verbose
        
        # Global metrics for analysis
        self.global_train_losses: List[float] = []
        self.global_eval_losses: List[float] = []
    
    def run_round(self, round_num: int):
        """
        Simulate one round of P2P communication and aggregation
        
        Steps:
        1. Each peer trains locally
        2. Each peer shares model with neighbors (P2P)
        3. Each peer aggregates received models
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"DECENTRALIZED FL - ROUND {round_num}")
            print(f"{'='*80}\n")
        
        # Step 1: All peers train locally
        peer_results = []
        for peer in self.peers:
            train_loss, eval_loss = peer.run_round()
            peer_results.append((peer.peer_id, train_loss, eval_loss))
        
        # Step 2: P2P Model Exchange (Ring topology)
        if self.verbose:
            print(f"\n--- P2P Model Exchange (Ring Topology) ---")
        for peer in self.peers:
            # Share model with neighbors
            neighbors = self.topology.get_neighbors(peer.peer_id)
            model_msg = peer.create_model_message(
                train_loss=peer.train_losses[-1],
                eval_loss=peer.eval_losses[-1]
            )
            
            # Send to each neighbor
            for neighbor_id in neighbors:
                neighbor_peer = self.peers[neighbor_id]
                neighbor_peer.receive_model(model_msg)
                if self.verbose:
                    print(f"[P2P] Peer {peer.peer_id} → Peer {neighbor_id}")
        
        # Step 3: Each peer aggregates locally
        if self.verbose:
            print(f"\n--- Local Aggregation at Each Peer ---")
        for peer in self.peers:
            peer.aggregate_models()
        
        # Calculate global metrics (for analysis only)
        avg_train_loss = np.mean([train for _, train, _ in peer_results])
        avg_eval_loss = np.mean([eval for _, _, eval in peer_results])
        self.global_train_losses.append(avg_train_loss)
        self.global_eval_losses.append(avg_eval_loss)
        
        # Summary - always show progress
        if round_num % 5 == 0 or round_num == 0:  # Show every 5 rounds
            print(f"\nRound {round_num}: Avg Train Loss={avg_train_loss:.6f}, Avg Eval Loss={avg_eval_loss:.6f}")
        
        if self.verbose:
            print(f"\n--- Round {round_num} Summary ---")
            print(f"Average Training Loss: {avg_train_loss:.6f}")
            print(f"Average Evaluation Loss: {avg_eval_loss:.6f}")
            
            # Show individual peer losses
            for peer_id, train, eval in peer_results:
                print(f"  Peer {peer_id}: Train={train:.6f}, Eval={eval:.6f}")
    
    def run_training(self, num_rounds: int):
        """Run multiple rounds of DFL"""
        print(f"\n{'#'*80}")
        print(f"STARTING DECENTRALIZED FEDERATED LEARNING")
        print(f"Peers: {self.num_peers} | Rounds: {num_rounds} | Topology: Ring")
        print(f"{'#'*80}\n")
        
        for round_num in range(num_rounds):
            self.run_round(round_num)
        
        print(f"\n{'#'*80}")
        print(f"DECENTRALIZED FL COMPLETED!")
        print(f"{'#'*80}\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_ring_topology(topology: RingTopology, save_path: str = "reports_dfl/ring_topology.png"):
    """Visualize the ring topology"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    
    # Title
    ax.text(0, 1.3, 'Decentralized FL: Ring Topology', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))
    
    num_peers = topology.num_peers
    radius = 1.0
    
    # Draw peers in a circle
    for i in range(num_peers):
        angle = 2 * np.pi * i / num_peers - np.pi/2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Draw peer node
        circle = plt.Circle((x, y), 0.15, color='lightgreen', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f'P{i}', fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)
        
        # Draw connection to right neighbor (clockwise)
        right_neighbor = topology.get_right_neighbor(i)
        angle_right = 2 * np.pi * right_neighbor / num_peers - np.pi/2
        x_right = radius * np.cos(angle_right)
        y_right = radius * np.sin(angle_right)
        
        # Arrow to right neighbor
        ax.annotate('', xy=(x_right, y_right), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6))
    
    # Legend
    ax.text(0, -1.2, 'Each peer connects to 2 neighbors (left & right)', 
            fontsize=10, ha='center', style='italic')
    ax.text(0, -1.35, 'Blue arrows: P2P communication flow', 
            fontsize=10, ha='center', color='blue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ring topology visualization saved to {save_path}")

def plot_peer_losses_comparison(coordinator_balanced: DFLCoordinator, coordinator_imbalanced: DFLCoordinator, 
                                 save_path: str = "reports_dfl/peer_losses.png"):
    """Plot individual peer losses comparison between balanced and imbalanced data distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Row 1: Balanced - Training and Evaluation losses
    ax1 = axes[0, 0]
    for peer in coordinator_balanced.peers:
        rounds = list(range(1, len(peer.train_losses) + 1))
        ax1.plot(rounds, peer.train_losses, marker='o', label=f'P{peer.peer_id}', alpha=0.7, markersize=3, linewidth=1.5)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('Balanced: Individual Peer Training Losses', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)
    
    ax2 = axes[0, 1]
    for peer in coordinator_balanced.peers:
        rounds = list(range(1, len(peer.eval_losses) + 1))
        ax2.plot(rounds, peer.eval_losses, marker='s', label=f'P{peer.peer_id}', alpha=0.7, markersize=3, linewidth=1.5)
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Evaluation Loss (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Balanced: Individual Peer Evaluation Losses', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)
    
    # Row 2: Imbalanced - Training and Evaluation losses
    ax3 = axes[1, 0]
    for peer in coordinator_imbalanced.peers:
        rounds = list(range(1, len(peer.train_losses) + 1))
        ax3.plot(rounds, peer.train_losses, marker='o', label=f'P{peer.peer_id}', alpha=0.7, markersize=3, linewidth=1.5)
    ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
    ax3.set_title('Imbalanced: Individual Peer Training Losses', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)
    
    ax4 = axes[1, 1]
    for peer in coordinator_imbalanced.peers:
        rounds = list(range(1, len(peer.eval_losses) + 1))
        ax4.plot(rounds, peer.eval_losses, marker='s', label=f'P{peer.peer_id}', alpha=0.7, markersize=3, linewidth=1.5)
    ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Evaluation Loss (MSE)', fontsize=12, fontweight='bold')
    ax4.set_title('Imbalanced: Individual Peer Evaluation Losses', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)
    
    plt.suptitle('DFL: Individual Peer Losses (Balanced vs Imbalanced)', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Peer losses comparison plot saved to {save_path}")

def visualize_sensor_data(save_path: str = "reports_dfl/sensor_data_visualization.png"):
    """Visualize bearing sensor data"""
    df = pd.read_csv(csv_filename)
    num_df = df.select_dtypes(include=[np.number])
    sensor_names = list(num_df.columns)[:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(sensor_names):
        axes[i].plot(df[col][:1000], linewidth=0.5, alpha=0.7)
        axes[i].set_title(f'{col} (first 1000 samples)', fontweight='bold')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle('Bearing Sensor Data Visualization', fontsize=14, fontweight='bold', y=1.002)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sensor data visualization saved to {save_path}")

def analyze_data_distribution(num_peers: int, save_path: str = "reports_dfl/data_distribution_visualization.png"):
    """Analyze and visualize data distribution across peers"""
    balanced_train_sizes = []
    imbalanced_train_sizes = []
    
    for i in range(num_peers):
        train_loader, _ = load_data(i, num_peers, data_distribution="balanced")
        balanced_train_sizes.append(len(train_loader.dataset))
        
        train_loader, _ = load_data(i, num_peers, data_distribution="imbalanced")
        imbalanced_train_sizes.append(len(train_loader.dataset))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart for balanced distribution
    ax = axes[0, 0]
    peer_ids = list(range(num_peers))
    ax.bar(peer_ids, balanced_train_sizes, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Peer ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Training Samples', fontsize=12, fontweight='bold')
    ax.set_title('IID (Balanced) Data Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(balanced_train_sizes):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Pie chart for balanced distribution
    ax = axes[0, 1]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, num_peers))
    wedges, texts, autotexts = ax.pie(balanced_train_sizes, labels=[f'P{i}' for i in range(num_peers)],
                                        autopct='%1.1f%%', colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    ax.set_title('IID (Balanced) Distribution Percentage', fontsize=14, fontweight='bold')
    
    # Bar chart for imbalanced distribution
    ax = axes[1, 0]
    ax.bar(peer_ids, imbalanced_train_sizes, color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Peer ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Training Samples', fontsize=12, fontweight='bold')
    ax.set_title('Non-IID (Imbalanced) Data Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(imbalanced_train_sizes):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Pie chart for imbalanced distribution
    ax = axes[1, 1]
    wedges, texts, autotexts = ax.pie(imbalanced_train_sizes, labels=[f'P{i}' for i in range(num_peers)],
                                        autopct='%1.1f%%', colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    ax.set_title('Non-IID (Imbalanced) Distribution Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data distribution visualization saved to {save_path}")
    
    return balanced_train_sizes, imbalanced_train_sizes

def test_anomaly_detection(model: BearingAutoencoder, device: torch.device):
    """Test anomaly detection with multiple scenarios"""
    df = pd.read_csv(csv_filename)
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.iloc[:, :8]
    
    # Calculate threshold
    train_loader, test_loader = load_data(0, 1, batch_size=BATCH_SIZE, data_distribution="balanced")
    criterion = torch.nn.MSELoss()
    model.eval()
    all_errors = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            output = model(x)
            errors = torch.mean((x - output) ** 2, dim=1).cpu().numpy()
            all_errors.extend(errors)
    
    all_errors = np.array(all_errors)
    threshold_95 = np.percentile(all_errors, 95)
    threshold_mean_2std = np.mean(all_errors) + 2 * np.std(all_errors)
    anomaly_threshold = threshold_95
    
    # Test function
    def test_bearing_sample(sensor_values, model, threshold, device, sample_name="Test Sample"):
        x = torch.tensor(sensor_values, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            reconstructed = model(x)
        x_np = x.cpu().numpy()[0]
        recon_np = reconstructed.cpu().numpy()[0]
        error = np.mean((x_np - recon_np) ** 2)
        is_anomaly = error > threshold
        return error, is_anomaly, x_np, recon_np
    
    # Test scenarios
    normal_sample = num_df.iloc[100].values
    error_normal, is_anomaly_normal, _, _ = test_bearing_sample(
        normal_sample, model, anomaly_threshold, device, "Normal Sample"
    )
    
    anomaly_sample_1 = normal_sample.copy()
    anomaly_sample_1[0] = anomaly_sample_1[0] * 10
    error_anomaly_1, is_anomaly_1, _, _ = test_bearing_sample(
        anomaly_sample_1, model, anomaly_threshold, device, "Scenario 1: Sensor Error"
    )
    
    anomaly_sample_2 = normal_sample.copy() * 3
    error_anomaly_2, is_anomaly_2, _, _ = test_bearing_sample(
        anomaly_sample_2, model, anomaly_threshold, device, "Scenario 2: High Vibration"
    )
    
    anomaly_sample_3 = normal_sample.copy()
    anomaly_sample_3[2] = -0.5
    anomaly_sample_3[3] = -0.3
    error_anomaly_3, is_anomaly_3, _, _ = test_bearing_sample(
        anomaly_sample_3, model, anomaly_threshold, device, "Scenario 3: Negative Values"
    )
    
    test_results = [
        ("Normal Sample", error_normal, is_anomaly_normal),
        ("Scenario 1: Sensor Error", error_anomaly_1, is_anomaly_1),
        ("Scenario 2: High Vibration", error_anomaly_2, is_anomaly_2),
        ("Scenario 3: Negative Values", error_anomaly_3, is_anomaly_3)
    ]
    
    return test_results, anomaly_threshold

def create_anomaly_detection_comparison(model_balanced: BearingAutoencoder, model_imbalanced: BearingAutoencoder,
                                        device: torch.device, save_path: str = "reports_dfl/anomaly_detection_comparison.png"):
    """Create anomaly detection comparison for balanced and imbalanced models"""
    
    # Test balanced model
    test_results_balanced, threshold_balanced = test_anomaly_detection(model_balanced, device)
    
    # Test imbalanced model  
    test_results_imbalanced, threshold_imbalanced = test_anomaly_detection(model_imbalanced, device)
    
    # Visualization - 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Balanced (Left)
    ax = axes[0]
    names = [name for name, _, _ in test_results_balanced]
    errors = [error for _, error, _ in test_results_balanced]
    colors = ['green' if not is_anom else 'red' for _, _, is_anom in test_results_balanced]
    bars = ax.bar(range(len(names)), errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=threshold_balanced, color='orange', linestyle='--', linewidth=2,
              label=f'Threshold: {threshold_balanced:.6f}')
    for i, (bar, error) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Test Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Balanced: Anomaly Detection', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Normal'),
        Patch(facecolor='red', alpha=0.7, label='Anomaly'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold_balanced:.6f}')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    
    # Imbalanced (Right)
    ax = axes[1]
    names = [name for name, _, _ in test_results_imbalanced]
    errors = [error for _, error, _ in test_results_imbalanced]
    colors = ['green' if not is_anom else 'red' for _, _, is_anom in test_results_imbalanced]
    bars = ax.bar(range(len(names)), errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=threshold_imbalanced, color='orange', linestyle='--', linewidth=2,
              label=f'Threshold: {threshold_imbalanced:.6f}')
    for i, (bar, error) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Test Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Imbalanced: Anomaly Detection', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Normal'),
        Patch(facecolor='red', alpha=0.7, label='Anomaly'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold_imbalanced:.6f}')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    
    plt.suptitle('DFL Anomaly Detection: Balanced vs Imbalanced', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Anomaly detection comparison saved to {save_path}")
    
    return test_results_balanced, threshold_balanced

def create_mse_distribution_comparison(model_balanced: BearingAutoencoder, model_imbalanced: BearingAutoencoder, 
                                       device: torch.device, save_path: str = "reports_dfl/mse_distribution_threshold.png"):
    """Create MSE distribution comparison for balanced and imbalanced models"""
    
    # Calculate errors for balanced model
    train_loader_balanced, test_loader_balanced = load_data(0, 1, batch_size=BATCH_SIZE, data_distribution="balanced")
    model_balanced.eval()
    all_errors_balanced = []
    with torch.no_grad():
        for batch in test_loader_balanced:
            x = batch['x'].to(device)
            output = model_balanced(x)
            errors = torch.mean((x - output) ** 2, dim=1).cpu().numpy()
            all_errors_balanced.extend(errors)
    
    all_errors_balanced = np.array(all_errors_balanced)
    threshold_95_balanced = np.percentile(all_errors_balanced, 95)
    threshold_mean_2std_balanced = np.mean(all_errors_balanced) + 2 * np.std(all_errors_balanced)
    
    # Calculate errors for imbalanced model
    train_loader_imbalanced, test_loader_imbalanced = load_data(0, 1, batch_size=BATCH_SIZE, data_distribution="imbalanced")
    model_imbalanced.eval()
    all_errors_imbalanced = []
    with torch.no_grad():
        for batch in test_loader_imbalanced:
            x = batch['x'].to(device)
            output = model_imbalanced(x)
            errors = torch.mean((x - output) ** 2, dim=1).cpu().numpy()
            all_errors_imbalanced.extend(errors)
    
    all_errors_imbalanced = np.array(all_errors_imbalanced)
    threshold_95_imbalanced = np.percentile(all_errors_imbalanced, 95)
    threshold_mean_2std_imbalanced = np.mean(all_errors_imbalanced) + 2 * np.std(all_errors_imbalanced)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Row 1: Balanced
    # Histogram (Balanced)
    ax = axes[0, 0]
    n, bins, patches = ax.hist(all_errors_balanced, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(threshold_95_balanced, color='red', linestyle='--', linewidth=2, label=f'95th Percentile: {threshold_95_balanced:.6f}')
    ax.axvline(threshold_mean_2std_balanced, color='orange', linestyle='--', linewidth=2, label=f'Mean + 2σ: {threshold_mean_2std_balanced:.6f}')
    ax.axvline(np.mean(all_errors_balanced), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_errors_balanced):.6f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Balanced: MSE Distribution with Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cumulative distribution (Balanced)
    ax = axes[0, 1]
    sorted_errors_balanced = np.sort(all_errors_balanced)
    cumulative_balanced = np.arange(1, len(sorted_errors_balanced) + 1) / len(sorted_errors_balanced) * 100
    ax.plot(sorted_errors_balanced, cumulative_balanced, color='steelblue', linewidth=2.5, label='Balanced')
    ax.axvline(threshold_95_balanced, color='red', linestyle='--', linewidth=2, label=f'95th Percentile: {threshold_95_balanced:.6f}')
    ax.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.axvline(threshold_mean_2std_balanced, color='orange', linestyle='--', linewidth=2, label=f'Mean + 2σ: {threshold_mean_2std_balanced:.6f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Balanced: Cumulative Distribution of MSE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Imbalanced
    # Histogram (Imbalanced)
    ax = axes[1, 0]
    n, bins, patches = ax.hist(all_errors_imbalanced, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(threshold_95_imbalanced, color='red', linestyle='--', linewidth=2, label=f'95th Percentile: {threshold_95_imbalanced:.6f}')
    ax.axvline(threshold_mean_2std_imbalanced, color='darkred', linestyle='--', linewidth=2, label=f'Mean + 2σ: {threshold_mean_2std_imbalanced:.6f}')
    ax.axvline(np.mean(all_errors_imbalanced), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_errors_imbalanced):.6f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Imbalanced: MSE Distribution with Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cumulative distribution (Imbalanced)
    ax = axes[1, 1]
    sorted_errors_imbalanced = np.sort(all_errors_imbalanced)
    cumulative_imbalanced = np.arange(1, len(sorted_errors_imbalanced) + 1) / len(sorted_errors_imbalanced) * 100
    ax.plot(sorted_errors_imbalanced, cumulative_imbalanced, color='orange', linewidth=2.5, label='Imbalanced')
    ax.axvline(threshold_95_imbalanced, color='red', linestyle='--', linewidth=2, label=f'95th Percentile: {threshold_95_imbalanced:.6f}')
    ax.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.axvline(threshold_mean_2std_imbalanced, color='darkred', linestyle='--', linewidth=2, label=f'Mean + 2σ: {threshold_mean_2std_imbalanced:.6f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Imbalanced: Cumulative Distribution of MSE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('MSE Distribution Comparison: Balanced vs Imbalanced', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MSE distribution comparison saved to {save_path}")
    
    return {
        'balanced': {
            'mean': float(np.mean(all_errors_balanced)),
            'std': float(np.std(all_errors_balanced)),
            'threshold_95': float(threshold_95_balanced),
            'threshold_mean_2std': float(threshold_mean_2std_balanced)
        },
        'imbalanced': {
            'mean': float(np.mean(all_errors_imbalanced)),
            'std': float(np.std(all_errors_imbalanced)),
            'threshold_95': float(threshold_95_imbalanced),
            'threshold_mean_2std': float(threshold_mean_2std_imbalanced)
        }
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_dfl_experiment(
    num_peers: int = NUM_PEERS,
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    data_distribution: str = "balanced",
    device: str = DEVICE,
    generate_visualizations: bool = True
):
    """Run Decentralized FL experiment"""
    
    print(f"\n{'='*80}")
    print(f"DECENTRALIZED FEDERATED LEARNING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Number of Peers: {num_peers}")
    print(f"  - Communication: Peer-to-Peer (P2P)")
    print(f"  - Topology: Ring")
    print(f"  - Aggregation: Local at each peer")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Local Epochs: {local_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Data Distribution: {data_distribution}")
    print(f"  - Device: {device}")
    print(f"{'='*80}\n")
    
    # Create ring topology
    topology = RingTopology(num_peers)
    
    # Generate visualizations only once (for first experiment)
    if generate_visualizations:
        # Visualize topology
        visualize_ring_topology(topology)
        
        # Visualize sensor data
        visualize_sensor_data()
    
    # Analyze data distribution
    balanced_sizes, imbalanced_sizes = analyze_data_distribution(num_peers)
    
    # Create peers
    peers = []
    for peer_id in range(num_peers):
        peer = DFLPeer(
            peer_id=peer_id,
            num_peers=num_peers,
            topology=topology,
            local_epochs=local_epochs,
            lr=learning_rate,
            data_distribution=data_distribution,
            device=device,
            verbose=VERBOSE
        )
        peers.append(peer)
    
    # Create coordinator (for simulation only)
    coordinator = DFLCoordinator(peers, topology, verbose=VERBOSE)
    
    # Run training
    coordinator.run_training(num_rounds)
    
    # Calculate anomaly threshold (without creating visualizations)
    df = pd.read_csv(csv_filename)
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.iloc[:, :8]
    
    train_loader, test_loader = load_data(0, 1, batch_size=BATCH_SIZE, data_distribution="balanced")
    peers[0].model.eval()
    all_errors = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(peers[0].device)
            output = peers[0].model(x)
            errors = torch.mean((x - output) ** 2, dim=1).cpu().numpy()
            all_errors.extend(errors)
    
    all_errors = np.array(all_errors)
    anomaly_threshold = np.percentile(all_errors, 95)
    
    # Test scenarios
    def test_bearing_sample(sensor_values, model, threshold, device):
        x = torch.tensor(sensor_values, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            reconstructed = model(x)
        x_np = x.cpu().numpy()[0]
        recon_np = reconstructed.cpu().numpy()[0]
        error = np.mean((x_np - recon_np) ** 2)
        is_anomaly = error > threshold
        return error, is_anomaly
    
    normal_sample = num_df.iloc[100].values
    error_normal, is_anomaly_normal = test_bearing_sample(
        normal_sample, peers[0].model, anomaly_threshold, peers[0].device
    )
    
    anomaly_sample_1 = normal_sample.copy()
    anomaly_sample_1[0] = anomaly_sample_1[0] * 10
    error_anomaly_1, is_anomaly_1 = test_bearing_sample(
        anomaly_sample_1, peers[0].model, anomaly_threshold, peers[0].device
    )
    
    anomaly_sample_2 = normal_sample.copy() * 3
    error_anomaly_2, is_anomaly_2 = test_bearing_sample(
        anomaly_sample_2, peers[0].model, anomaly_threshold, peers[0].device
    )
    
    anomaly_sample_3 = normal_sample.copy()
    anomaly_sample_3[2] = -0.5
    anomaly_sample_3[3] = -0.3
    error_anomaly_3, is_anomaly_3 = test_bearing_sample(
        anomaly_sample_3, peers[0].model, anomaly_threshold, peers[0].device
    )
    
    test_results = [
        ("Normal Sample", error_normal, is_anomaly_normal),
        ("Scenario 1: Sensor Error", error_anomaly_1, is_anomaly_1),
        ("Scenario 2: High Vibration", error_anomaly_2, is_anomaly_2),
        ("Scenario 3: Negative Values", error_anomaly_3, is_anomaly_3)
    ]
    
    # Collect experiment data for JSON export
    experiment_data = {
        "data_distribution": {
            "balanced": [int(s) for s in balanced_sizes],
            "imbalanced": [int(s) for s in imbalanced_sizes]
        },
        "convergence": {
            "train_losses": {
                "max": float(np.max(coordinator.global_train_losses)),
                "min": float(np.min(coordinator.global_train_losses)),
                "avg": float(np.mean(coordinator.global_train_losses)),
                "median": float(np.median(coordinator.global_train_losses))
            },
            "eval_losses": {
                "max": float(np.max(coordinator.global_eval_losses)),
                "min": float(np.min(coordinator.global_eval_losses)),
                "avg": float(np.mean(coordinator.global_eval_losses)),
                "median": float(np.median(coordinator.global_eval_losses))
            },
            "initial_train_loss": float(coordinator.global_train_losses[0]),
            "final_train_loss": float(coordinator.global_train_losses[-1]),
            "train_loss_reduction": float(coordinator.global_train_losses[0] - coordinator.global_train_losses[-1]),
            "initial_eval_loss": float(coordinator.global_eval_losses[0]),
            "final_eval_loss": float(coordinator.global_eval_losses[-1]),
            "eval_loss_reduction": float(coordinator.global_eval_losses[0] - coordinator.global_eval_losses[-1])
        },
        "individual_peers": [
            {
                "peer_id": peer.peer_id,
                "train_losses": {
                    "max": float(np.max(peer.train_losses)),
                    "min": float(np.min(peer.train_losses)),
                    "avg": float(np.mean(peer.train_losses)),
                    "median": float(np.median(peer.train_losses))
                },
                "eval_losses": {
                    "max": float(np.max(peer.eval_losses)),
                    "min": float(np.min(peer.eval_losses)),
                    "avg": float(np.mean(peer.eval_losses)),
                    "median": float(np.median(peer.eval_losses))
                },
                "final_train_loss": float(peer.train_losses[-1]),
                "final_eval_loss": float(peer.eval_losses[-1]),
                "num_samples": len(peer.trainloader.dataset)
            }
            for peer in peers
        ],
        "anomaly_detection": {
            "threshold": float(anomaly_threshold),
            "test_results": [
                {
                    "name": name,
                    "error": float(error),
                    "is_anomaly": bool(is_anomaly)
                }
                for name, error, is_anomaly in test_results
            ]
        }
    }
    
    # Save final model for anomaly detection
    final_peer = peers[0]  # Use first peer's final model
    
    print(f"\n{'='*80}")
    print(f"Experiment ({data_distribution}) completed")
    print(f"{'='*80}\n")
    
    return coordinator, final_peer, experiment_data

if __name__ == "__main__":
    # Run DFL experiments
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: DFL with BALANCED data")
    print("="*80)
    coordinator_balanced, final_peer_balanced, data_balanced = run_dfl_experiment(
        num_peers=NUM_PEERS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        data_distribution="balanced",
        device=DEVICE,
        generate_visualizations=True
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: DFL with IMBALANCED data")
    print("="*80)
    coordinator_imbalanced, final_peer_imbalanced, data_imbalanced = run_dfl_experiment(
        num_peers=NUM_PEERS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        data_distribution="imbalanced",
        device=DEVICE,
        generate_visualizations=False  # Skip duplicate visualizations
    )
    
    # Plot peer losses comparison
    plot_peer_losses_comparison(coordinator_balanced, coordinator_imbalanced)
    
    # Create anomaly detection comparison for both balanced and imbalanced
    print(f"\n{'='*80}")
    print("Creating Anomaly Detection Comparison...")
    print(f"{'='*80}\n")
    test_results, anomaly_threshold = create_anomaly_detection_comparison(
        final_peer_balanced.model, final_peer_imbalanced.model, final_peer_balanced.device
    )
    
    # Create MSE distribution comparison for both balanced and imbalanced
    print(f"\n{'='*80}")
    print("Creating MSE Distribution Comparison...")
    print(f"{'='*80}\n")
    mse_stats = create_mse_distribution_comparison(final_peer_balanced.model, final_peer_imbalanced.model, final_peer_balanced.device)
    
    # Compare balanced vs imbalanced
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    rounds = list(range(1, len(coordinator_balanced.global_train_losses) + 1))
    ax.plot(rounds, coordinator_balanced.global_train_losses, marker='o', color='steelblue', linewidth=2, label='Balanced')
    ax.plot(rounds, coordinator_imbalanced.global_train_losses, marker='s', color='orange', linewidth=2, label='Imbalanced')
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Training Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('DFL: Balanced vs Imbalanced (Training)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1]
    ax.plot(rounds, coordinator_balanced.global_eval_losses, marker='o', color='steelblue', linewidth=2, label='Balanced')
    ax.plot(rounds, coordinator_imbalanced.global_eval_losses, marker='s', color='orange', linewidth=2, label='Imbalanced')
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Evaluation Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('DFL: Balanced vs Imbalanced (Evaluation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('reports_dfl/experiments_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final loss comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = ["DFL\nBalanced", "DFL\nImbalanced"]
    final_train_losses = [
        coordinator_balanced.global_train_losses[-1],
        coordinator_imbalanced.global_train_losses[-1]
    ]
    colors_bar = ["steelblue", "orange"]
    bars = ax.bar(exp_names, final_train_losses, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel("Final Training Loss (MSE)", fontsize=12, fontweight='bold')
    ax.set_title("DFL Final Loss Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports_dfl/final_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive JSON report (combines all text reports into one JSON file)
    comprehensive_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "architecture": "Decentralized Federated Learning (DFL)",
            "communication": "Peer-to-Peer (P2P)",
            "topology": "Ring",
            "aggregation": "Local at each peer"
        },
        "configuration": {
            "num_peers": NUM_PEERS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "device": DEVICE
        },
        "data_distribution_analysis": {
            "balanced": {
                "peers": [
                    {
                        "peer_id": i,
                        "train_samples": int(data_balanced["data_distribution"]["balanced"][i])
                    }
                    for i in range(NUM_PEERS)
                ],
                "total_samples": sum(data_balanced["data_distribution"]["balanced"]),
                "mean_samples": float(np.mean(data_balanced["data_distribution"]["balanced"])),
                "std_samples": float(np.std(data_balanced["data_distribution"]["balanced"]))
            },
            "imbalanced": {
                "peers": [
                    {
                        "peer_id": i,
                        "train_samples": int(data_imbalanced["data_distribution"]["imbalanced"][i]),
                        "percentage": float(data_imbalanced["data_distribution"]["imbalanced"][i] / 
                                          sum(data_imbalanced["data_distribution"]["imbalanced"]) * 100)
                    }
                    for i in range(NUM_PEERS)
                ],
                "total_samples": sum(data_imbalanced["data_distribution"]["imbalanced"]),
                "mean_samples": float(np.mean(data_imbalanced["data_distribution"]["imbalanced"])),
                "std_samples": float(np.std(data_imbalanced["data_distribution"]["imbalanced"]))
            }
        },
        "experiments": {
            "balanced": {
                "data_distribution": "balanced",
                "convergence": data_balanced["convergence"],
                "individual_peers": data_balanced["individual_peers"],
                "anomaly_detection": data_balanced["anomaly_detection"]
            },
            "imbalanced": {
                "data_distribution": "imbalanced",
                "convergence": data_imbalanced["convergence"],
                "individual_peers": data_imbalanced["individual_peers"],
                "anomaly_detection": data_imbalanced["anomaly_detection"]
            }
        },
        "summary_table": {
            "balanced": {
                "experiment": "DFL Balanced",
                "final_train_loss": float(coordinator_balanced.global_train_losses[-1]),
                "final_eval_loss": float(coordinator_balanced.global_eval_losses[-1]),
                "train_loss_reduction": float(coordinator_balanced.global_train_losses[0] - 
                                             coordinator_balanced.global_train_losses[-1]),
                "eval_loss_reduction": float(coordinator_balanced.global_eval_losses[0] - 
                                            coordinator_balanced.global_eval_losses[-1])
            },
            "imbalanced": {
                "experiment": "DFL Imbalanced",
                "final_train_loss": float(coordinator_imbalanced.global_train_losses[-1]),
                "final_eval_loss": float(coordinator_imbalanced.global_eval_losses[-1]),
                "train_loss_reduction": float(coordinator_imbalanced.global_train_losses[0] - 
                                             coordinator_imbalanced.global_train_losses[-1]),
                "eval_loss_reduction": float(coordinator_imbalanced.global_eval_losses[0] - 
                                            coordinator_imbalanced.global_eval_losses[-1])
            }
        },
        "generated_files": {
            "visualizations": [
                {
                    "filename": "sensor_data_visualization.png",
                    "description": "Visualizes 8 bearing sensor channels showing time-series data from the first 1000 samples. Displays raw sensor readings to understand data patterns and characteristics.",
                    "purpose": "Data exploration and understanding sensor behavior"
                },
                {
                    "filename": "mse_distribution_threshold.png",
                    "description": "Shows MSE (Mean Squared Error) distribution histograms and cumulative distribution curves for both balanced and imbalanced models. Includes anomaly detection thresholds (95th percentile and Mean+2σ).",
                    "purpose": "Threshold determination for anomaly detection and error distribution analysis"
                },
                {
                    "filename": "anomaly_detection_comparison.png",
                    "description": "Compares anomaly detection performance between balanced and imbalanced models across 4 test scenarios: Normal, Sensor Error, High Vibration, and Negative Values. Shows reconstruction errors with threshold lines.",
                    "purpose": "Evaluate model effectiveness in detecting bearing anomalies"
                },
                {
                    "filename": "data_distribution_visualization.png",
                    "description": "Displays training data distribution across 10 peers using bar charts and pie charts. Shows both IID (balanced) and Non-IID (imbalanced) distributions with sample counts and percentages.",
                    "purpose": "Understand data partitioning strategy and imbalance levels across peers"
                },
                {
                    "filename": "ring_topology.png",
                    "description": "Illustrates the decentralized federated learning ring topology where each peer (P0-P9) connects to 2 neighbors. Blue arrows show P2P communication flow in a circular arrangement.",
                    "purpose": "Visualize DFL network architecture and peer-to-peer communication structure"
                },
                {
                    "filename": "peer_losses.png",
                    "description": "Shows individual peer training and evaluation losses over 50 rounds for both balanced and imbalanced scenarios. Each peer's loss trajectory is plotted separately to observe convergence patterns.",
                    "purpose": "Monitor individual peer performance and identify convergence issues"
                },
                {
                    "filename": "experiments_comparison.png",
                    "description": "Compares average training and evaluation losses between balanced and imbalanced experiments across all rounds. Shows convergence trends and final performance differences.",
                    "purpose": "Compare overall model performance under different data distribution strategies"
                },
                {
                    "filename": "final_loss_comparison.png",
                    "description": "Bar chart comparing final training losses between DFL Balanced and DFL Imbalanced experiments. Shows exact loss values for quick performance comparison.",
                    "purpose": "Quick visual summary of final model performance"
                }
            ],
            "reports": [
                "dfl_results.json"
            ]
        }
    }
    
    # Save comprehensive JSON report
    with open('reports_dfl/dfl_results.json', 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print("📊 JSON REPORT GENERATED")
    print(f"{'='*80}")
    print(f"  📄 reports_dfl/dfl_results.json")
    print(f"{'='*80}\n")
    
    print(f"\nComparison plots saved to: reports_dfl/")
    print(f"  - experiments_comparison.png")
    print(f"  - final_loss_comparison.png")
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80 + "\n")

