"""Coordinator for orchestrating decentralized federated learning"""

import threading
import queue
import time
import logging
from typing import Dict, List, Optional
import numpy as np

from dfl_peer import DFLPeer
from peer_worker import PeerWorker
from topology import RingTopology, create_topology, BaseTopology
from messages import ControlMessage, StatusMessage, create_control_message
from data_utils import BearingDatasetLoader, create_data_loaders


logger = logging.getLogger(__name__)


class Coordinator:
    """Coordinator for DFL simulation
    
    Manages peer workers, orchestrates training rounds, and tracks metrics
    """
    
    def __init__(self):
        """Initialize coordinator"""
        self.peers: List[DFLPeer] = []
        self.workers: List[PeerWorker] = []
        self.topology: Optional[BaseTopology] = None
        
        # Message queues
        self.control_queues: Dict[int, queue.Queue] = {}
        self.inboxes: Dict[int, queue.Queue] = {}
        self.status_queue = queue.Queue()
        
        # State
        self.initialized = False
        self.running = False
        self.current_round = 0
        
        # Metrics tracking
        self.metrics_history = []  # Per-round global metrics
        self.peer_history = {}  # Per-peer metrics history
        self.bandwidth_history = []  # Per-round bandwidth
        self.cumulative_bandwidth = None  # Cumulative bandwidth matrix
        
        # Logs
        self.logs = []
        self.max_logs = 1000
        
        # Configuration
        self.config = {}
        
        logger.info("Coordinator initialized")
    
    def initialize(self, num_peers: int, hops: List[int] = None,
                  data_distribution: str = "iid", local_epochs: int = 1,
                  learning_rate: float = 0.01, batch_size: int = 32,
                  device: str = "cpu", latency_ms: float = 0.0,
                  drop_prob: float = 0.0, aggregate_method: str = "avg",
                  mu: float = 0.01, dataset: str = "bearing", csv_path: str = None,
                  peer_data_fractions: List[float] = None,
                  topology_type: str = "ring", topology_params: Dict = None):
        """Initialize the DFL system
        
        Args:
            num_peers: Number of peers
            hops: List of hop distances for ring topology (deprecated, use topology_params)
            data_distribution: Data distribution type ('iid', 'non_iid', 'label_skew')
            local_epochs: Local training epochs per round
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to use ('cpu' or 'cuda')
            latency_ms: Simulated latency in milliseconds
            drop_prob: Message drop probability
            aggregate_method: Aggregation method ('avg' or 'prox')
            mu: FedProx proximal term
            dataset: Dataset to use ('bearing')
            csv_path: Path to CSV file (for bearing dataset, optional)
            peer_data_fractions: List of fractions (0-1) for each peer's data amount (optional)
            topology_type: Type of topology ('ring', 'line', 'mesh', 'star', 'full')
            topology_params: Dict of topology-specific parameters
                Ring: {'hops': [1, 2]}
                Line: {'bidirectional': True}
                Mesh: {'connectivity': 0.5} or {'edges': [(0,1), (1,2)]}
                Star: {'center_peer': 0}
        """
        if self.initialized:
            raise RuntimeError("Coordinator already initialized. Call reset() first.")
        
        self._log(f"Initializing {num_peers} peers with {data_distribution} data distribution on {dataset} dataset")
        
        # Handle backward compatibility for hops parameter
        if topology_params is None:
            topology_params = {}
        if hops is not None and topology_type == "ring" and 'hops' not in topology_params:
            topology_params['hops'] = hops
        
        # Save configuration
        self.config = {
            'num_peers': num_peers,
            'hops': hops or [1],  # Keep for backward compatibility
            'topology_type': topology_type,
            'topology_params': topology_params,
            'data_distribution': data_distribution,
            'local_epochs': local_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'device': device,
            'latency_ms': latency_ms,
            'drop_prob': drop_prob,
            'aggregate_method': aggregate_method,
            'mu': mu,
            'dataset': dataset,
            'csv_path': csv_path
        }
        
        # Create topology
        self.topology = create_topology(topology_type, num_peers, **topology_params)
        self._log(f"Created {topology_type} topology: {self.topology.get_topology_info()}")
        
        # Load and distribute data
        if dataset.lower() == "bearing":
            data_loader = BearingDatasetLoader(csv_filename=csv_path)
            peer_data = data_loader.distribute_data(num_peers, data_distribution, peer_fractions=peer_data_fractions)
            # Get input/output dimensions from the dataset
            input_dim = data_loader.num_features
            output_dim = data_loader.num_classes
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Currently only 'bearing' is supported.")
        
        self._log(f"Dataset loaded: input_dim={input_dim}, output_dim={output_dim}")
        
        # Initialize cumulative bandwidth matrix
        self.cumulative_bandwidth = np.zeros((num_peers, num_peers))
        
        # Create peers and queues
        for peer_id in range(num_peers):
            train_data, test_data = peer_data[peer_id]
            
            # Create DFLPeer
            peer = DFLPeer(
                peer_id=peer_id,
                train_data=train_data,
                test_data=test_data,
                batch_size=batch_size,
                device=device,
                aggregate_method=aggregate_method,
                mu=mu,
                learning_rate=learning_rate,
                input_dim=input_dim,
                output_dim=output_dim
            )
            self.peers.append(peer)
            
            # Create queues
            self.control_queues[peer_id] = queue.Queue()
            self.inboxes[peer_id] = queue.Queue()
            
            # Initialize peer history
            self.peer_history[peer_id] = {
                'train_loss': [],
                'eval_loss': [],
                'eval_mse': [],  # MSE for anomaly detection
                'enabled': [],
                'sent_bytes': [],
                'recv_bytes': []
            }
        
        # Create workers (not started yet)
        for peer_id, peer in enumerate(self.peers):
            worker = PeerWorker(
                dfl_peer=peer,
                inbox=self.inboxes[peer_id],
                control_q=self.control_queues[peer_id],
                status_queue=self.status_queue,
                topology=self.topology,
                peer_inboxes=self.inboxes,
                latency_ms=latency_ms,
                drop_prob=drop_prob,
                local_epochs=local_epochs
            )
            self.workers.append(worker)
        
        self.initialized = True
        self._log(f"Initialized {num_peers} peers successfully")
    
    def start(self):
        """Start all worker threads"""
        if not self.initialized:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")
        
        if self.running:
            raise RuntimeError("Coordinator already running")
        
        self._log("Starting all peer workers")
        for worker in self.workers:
            worker.start()
        
        self.running = True
        self._log("All peer workers started")
    
    def step(self, timeout: float = 30.0) -> Dict:
        """Execute one training round
        
        Args:
            timeout: Maximum time to wait for peer responses
        
        Returns:
            Dict with round metrics
        """
        if not self.running:
            raise RuntimeError("Coordinator not running. Call start() first.")
        
        self.current_round += 1
        round_id = self.current_round
        
        self._log(f"Starting round {round_id}")
        
        # Send START_ROUND to all peers
        control_msg = create_control_message(
            cmd="START_ROUND",
            round_id=round_id,
            timeout=timeout
        )
        
        for peer_id in range(len(self.peers)):
            try:
                self.control_queues[peer_id].put(control_msg, timeout=1.0)
            except queue.Full:
                self._log(f"Warning: control queue full for peer {peer_id}", level="WARNING")
        
        # Collect status messages
        statuses = self._collect_status_messages(round_id, timeout)
        
        # Aggregate metrics
        metrics = self._aggregate_metrics(round_id, statuses)
        
        # Track bandwidth
        self._track_bandwidth(statuses)
        
        self._log(f"Round {round_id} complete: train_loss={metrics['global_train_loss']:.4f}, "
                 f"eval_loss={metrics['global_eval_loss']:.4f}, "
                 f"eval_mse={metrics['global_eval_mse']:.4f}")
        
        return metrics
    
    def stop(self):
        """Stop all worker threads"""
        if not self.running:
            return
        
        self._log("Stopping all peer workers")
        
        # Send STOP_THREAD to all workers
        stop_msg = create_control_message(cmd="STOP_THREAD")
        for peer_id in range(len(self.peers)):
            try:
                self.control_queues[peer_id].put(stop_msg, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for threads to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.running = False
        self._log("All peer workers stopped")
    
    def reset(self):
        """Reset coordinator state"""
        if self.running:
            self.stop()
        
        self.peers.clear()
        self.workers.clear()
        self.control_queues.clear()
        self.inboxes.clear()
        
        # Clear queues
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                break
        
        self.topology = None
        self.initialized = False
        self.current_round = 0
        self.metrics_history.clear()
        self.peer_history.clear()
        self.bandwidth_history.clear()
        self.cumulative_bandwidth = None
        self.logs.clear()
        
        self._log("Coordinator reset")
    
    def toggle_peer(self, peer_id: int, enabled: bool, fetch_model: bool = False):
        """Enable or disable a peer
        
        Args:
            peer_id: Peer to toggle
            enabled: True to enable, False to disable
            fetch_model: Whether to fetch model from neighbors when enabling
        """
        if peer_id < 0 or peer_id >= len(self.peers):
            raise ValueError(f"Invalid peer_id: {peer_id}")
        
        cmd = "ENABLE" if enabled else "DISABLE"
        msg = create_control_message(cmd=cmd, fetch_model=fetch_model)
        
        try:
            self.control_queues[peer_id].put(msg, timeout=1.0)
            self._log(f"Peer {peer_id} {'enabled' if enabled else 'disabled'}")
        except queue.Full:
            self._log(f"Failed to toggle peer {peer_id}: queue full", level="ERROR")
    
    def set_neighbors(self, peer_id: int = None, neighbors: List[int] = None, 
                     hops: List[int] = None):
        """Update topology neighbors
        
        Args:
            peer_id: Specific peer to update (None for global)
            neighbors: Custom neighbor list
            hops: New hop distances
        """
        if self.topology is None:
            raise RuntimeError("Topology not initialized")
        
        self.topology.set_neighbors(peer_id=peer_id, neighbors=neighbors, 
                                    hops=set(hops) if hops else None)
        self._log(f"Topology updated: peer_id={peer_id}, neighbors={neighbors}, hops={hops}")
    
    def set_aggregate_method(self, peer_id: int = None, aggregate_method: str = "avg", 
                           mu: float = 0.01):
        """Set aggregation method for peer(s)
        
        Args:
            peer_id: Specific peer (None for all peers)
            aggregate_method: 'avg' or 'prox'
            mu: FedProx coefficient
        """
        msg = create_control_message(
            cmd="SET_AGGREGATE",
            aggregate_method=aggregate_method,
            mu=mu
        )
        
        if peer_id is not None:
            # Set for specific peer
            try:
                self.control_queues[peer_id].put(msg, timeout=1.0)
                self._log(f"Peer {peer_id} aggregation method set to {aggregate_method}")
            except queue.Full:
                self._log(f"Failed to set aggregate method for peer {peer_id}", level="ERROR")
        else:
            # Set for all peers
            for pid in range(len(self.peers)):
                try:
                    self.control_queues[pid].put(msg, timeout=1.0)
                except queue.Full:
                    pass
            self._log(f"All peers aggregation method set to {aggregate_method}")
    
    def get_status(self, peer_id: int = None) -> Dict:
        """Get current status
        
        Args:
            peer_id: Specific peer (None for all peers)
        
        Returns:
            Status dictionary
        """
        if peer_id is not None:
            if peer_id < 0 or peer_id >= len(self.peers):
                raise ValueError(f"Invalid peer_id: {peer_id}")
            
            return {
                'peer_id': peer_id,
                'enabled': self.workers[peer_id].enabled,
                'thread_alive': self.workers[peer_id].is_alive(),
                'neighbors': self.topology.get_neighbors(peer_id) if self.topology else [],
                'history': self.peer_history.get(peer_id, {})
            }
        else:
            # All peers
            return {
                'num_peers': len(self.peers),
                'current_round': self.current_round,
                'running': self.running,
                'initialized': self.initialized,
                'peers': [
                    {
                        'peer_id': i,
                        'enabled': worker.enabled,
                        'thread_alive': worker.is_alive(),
                        'neighbors': self.topology.get_neighbors(i) if self.topology else [],
                        'num_train_samples': self.peers[i].num_train_samples if i < len(self.peers) else 0
                    }
                    for i, worker in enumerate(self.workers)
                ]
            }
    
    def get_bandwidth(self, round_id: int = None) -> Dict:
        """Get bandwidth statistics
        
        Args:
            round_id: Specific round (None for all rounds)
        
        Returns:
            Bandwidth statistics
        """
        if round_id is not None:
            if round_id <= 0 or round_id > len(self.bandwidth_history):
                raise ValueError(f"Invalid round_id: {round_id}")
            return self.bandwidth_history[round_id - 1]
        else:
            return {
                'per_round': self.bandwidth_history,
                'cumulative_matrix': self.cumulative_bandwidth.tolist() if self.cumulative_bandwidth is not None else []
            }
    
    def get_metrics(self) -> Dict:
        """Get all metrics history"""
        return {
            'global_metrics': self.metrics_history,
            'peer_metrics': self.peer_history,
            'current_round': self.current_round
        }
    
    def get_logs(self, limit: int = 100) -> List[str]:
        """Get recent logs"""
        return self.logs[-limit:]
    
    def _collect_status_messages(self, round_id: int, timeout: float) -> List[StatusMessage]:
        """Collect status messages from peers"""
        statuses = []
        expected_count = len(self.peers)
        start_time = time.time()
        
        while len(statuses) < expected_count:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                self._log(f"Timeout collecting status messages: {len(statuses)}/{expected_count} received",
                         level="WARNING")
                break
            
            try:
                status = self.status_queue.get(timeout=min(remaining_time, 1.0))
                if status.round_id == round_id:
                    statuses.append(status)
            except queue.Empty:
                continue
        
        return statuses
    
    def _aggregate_metrics(self, round_id: int, statuses: List[StatusMessage]) -> Dict:
        """Aggregate metrics from peer statuses"""
        if not statuses:
            return {
                'round': round_id,
                'global_train_loss': float('inf'),
                'global_eval_loss': float('inf'),
                'global_eval_mse': 0.0,
                'num_active_peers': 0
            }
        
        # Filter enabled peers
        active_statuses = [s for s in statuses if s.enabled]
        
        if not active_statuses:
            return {
                'round': round_id,
                'global_train_loss': float('inf'),
                'global_eval_loss': float('inf'),
                'global_eval_mse': 0.0,
                'num_active_peers': 0
            }
        
        # Weighted average by number of samples
        total_samples = sum(s.num_samples for s in active_statuses)
        
        global_train_loss = sum(s.train_loss * s.num_samples for s in active_statuses) / total_samples
        global_eval_loss = sum(s.eval_loss * s.num_samples for s in active_statuses) / total_samples
        global_eval_mse = sum(s.eval_mse * s.num_samples for s in active_statuses) / total_samples
        
        # Update peer history
        for status in statuses:
            peer_id = status.peer_id
            self.peer_history[peer_id]['train_loss'].append(status.train_loss)
            self.peer_history[peer_id]['eval_loss'].append(status.eval_loss)
            self.peer_history[peer_id]['eval_mse'].append(status.eval_mse)
            self.peer_history[peer_id]['enabled'].append(status.enabled)
            self.peer_history[peer_id]['sent_bytes'].append(status.sent_bytes)
            self.peer_history[peer_id]['recv_bytes'].append(status.recv_bytes)
        
        # Save global metrics
        metrics = {
            'round': round_id,
            'global_train_loss': global_train_loss,
            'global_eval_loss': global_eval_loss,
            'global_eval_mse': global_eval_mse,
            'num_active_peers': len(active_statuses),
            'num_total_peers': len(statuses)
        }
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _track_bandwidth(self, statuses: List[StatusMessage]):
        """Track bandwidth from peer statuses"""
        round_bandwidth = {
            'total_sent': sum(s.sent_bytes for s in statuses),
            'total_recv': sum(s.recv_bytes for s in statuses),
            'per_peer': {}
        }
        
        # Track per-peer bandwidth
        for status in statuses:
            peer_id = status.peer_id
            round_bandwidth['per_peer'][peer_id] = {
                'sent': status.sent_bytes,
                'recv': status.recv_bytes,
                'neighbors': status.neighbors
            }
            
            # Update cumulative matrix (approximate: distribute sent bytes equally to neighbors)
            if status.neighbors and status.sent_bytes > 0:
                bytes_per_neighbor = status.sent_bytes / len(status.neighbors)
                for neighbor_id in status.neighbors:
                    self.cumulative_bandwidth[peer_id][neighbor_id] += bytes_per_neighbor
        
        self.bandwidth_history.append(round_bandwidth)
    
    def _log(self, message: str, level: str = "INFO"):
        """Add log message"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        
        # Trim logs if too many
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Also log to logger
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
