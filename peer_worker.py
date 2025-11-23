"""PeerWorker thread implementation"""

import threading
import queue
import time
import random
import logging
from typing import Dict, List, Optional

from dfl_peer import DFLPeer
from messages import ModelMessage, ControlMessage, StatusMessage, create_model_message, create_status_message
from topology import RingTopology


logger = logging.getLogger(__name__)


class PeerWorker(threading.Thread):
    """Worker thread for a DFL peer
    
    Handles message processing, training coordination, and model exchange
    """
    
    def __init__(self, dfl_peer: DFLPeer, inbox: queue.Queue, 
                 control_q: queue.Queue, status_queue: queue.Queue,
                 topology: RingTopology, peer_inboxes: Dict[int, queue.Queue],
                 latency_ms: float = 0.0, drop_prob: float = 0.0,
                 local_epochs: int = 1):
        """Initialize PeerWorker
        
        Args:
            dfl_peer: DFLPeer instance
            inbox: Queue for incoming MODEL messages
            control_q: Queue for CONTROL messages from coordinator
            status_queue: Queue for sending STATUS back to coordinator
            topology: Topology instance
            peer_inboxes: Dict mapping peer_id -> inbox queue (for sending messages)
            latency_ms: Simulated latency in milliseconds
            drop_prob: Probability of dropping a message (0.0-1.0)
            local_epochs: Number of local training epochs per round
        """
        super().__init__(daemon=True)
        self.peer = dfl_peer
        self.inbox = inbox
        self.control_q = control_q
        self.status_queue = status_queue
        self.topology = topology
        self.peer_inboxes = peer_inboxes
        
        self.enabled = True
        self.running = True
        self.latency_ms = latency_ms
        self.drop_prob = drop_prob
        self.local_epochs = local_epochs
        
        # Bandwidth tracking
        self.sent_bytes = 0
        self.recv_bytes = 0
        self.round_sent_bytes = 0
        self.round_recv_bytes = 0
        
        logger.info(f"PeerWorker {self.peer.peer_id} initialized")
    
    def run(self):
        """Main thread loop"""
        logger.info(f"PeerWorker {self.peer.peer_id} started")
        
        while self.running:
            try:
                # Wait for control message with timeout
                msg = self.control_q.get(timeout=1.0)
                self._handle_control_message(msg)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Peer {self.peer.peer_id} error: {e}", exc_info=True)
        
        logger.info(f"PeerWorker {self.peer.peer_id} stopped")
    
    def _handle_control_message(self, msg: ControlMessage):
        """Handle control message from coordinator"""
        cmd = msg.cmd
        
        if cmd == "STOP_THREAD":
            self.running = False
            logger.info(f"Peer {self.peer.peer_id} received STOP_THREAD")
        
        elif cmd == "ENABLE":
            self.enabled = True
            if msg.fetch_model:
                self._fetch_model_from_neighbors()
            logger.info(f"Peer {self.peer.peer_id} enabled")
        
        elif cmd == "DISABLE":
            self.enabled = False
            logger.info(f"Peer {self.peer.peer_id} disabled")
        
        elif cmd == "SET_AGGREGATE":
            if msg.aggregate_method:
                self.peer.set_aggregate_method(msg.aggregate_method, msg.mu)
                logger.info(f"Peer {self.peer.peer_id} aggregation method set to {msg.aggregate_method}")
        
        elif cmd == "START_ROUND":
            self._execute_round(msg)
    
    def _execute_round(self, msg: ControlMessage):
        """Execute a training round"""
        round_id = msg.round_id
        
        # Reset round bandwidth counters
        self.round_sent_bytes = 0
        self.round_recv_bytes = 0
        
        if not self.enabled:
            # Send status even if disabled (so coordinator knows we're alive but idle)
            self._send_status(round_id, train_loss=0.0, eval_loss=0.0, eval_mse=0.0)
            return
        
        try:
            # Step 1: Local training
            local_epochs = msg.local_epochs if msg.local_epochs else self.local_epochs
            train_loss = self.peer.train_local(epochs=local_epochs)
            
            # Step 2: Evaluate
            eval_loss, eval_mse = self.peer.evaluate_local()
            
            # Step 3: Send model to neighbors
            self._send_model_to_neighbors(round_id, train_loss)
            
            # Step 4: Receive models from neighbors
            neighbor_models = self._receive_models_from_neighbors(round_id, timeout=msg.timeout or 10.0)
            
            # Step 5: Aggregate models
            if neighbor_models:
                self.peer.aggregate_models(neighbor_models)
            
            # Step 6: Send status to coordinator
            self._send_status(round_id, train_loss, eval_loss, eval_mse)
            
        except Exception as e:
            logger.error(f"Peer {self.peer.peer_id} round {round_id} failed: {e}", exc_info=True)
            self._send_status(round_id, train_loss=float('inf'), eval_loss=float('inf'), eval_mse=0.0)
    
    def _send_model_to_neighbors(self, round_id: int, train_loss: float):
        """Send model to all neighbors"""
        neighbors = self.topology.get_neighbors(self.peer.peer_id)
        model_info = self.peer.get_model_info()
        
        for neighbor_id in neighbors:
            # Simulate message drop
            if random.random() < self.drop_prob:
                logger.debug(f"Peer {self.peer.peer_id} -> {neighbor_id}: message dropped")
                continue
            
            # Create model message
            msg = create_model_message(
                peer_id=self.peer.peer_id,
                round_id=round_id,
                parameters=model_info['parameters'],
                num_samples=model_info['num_samples'],
                train_loss=train_loss
            )
            
            # Calculate message size
            msg_size = msg.size_bytes()
            self.sent_bytes += msg_size
            self.round_sent_bytes += msg_size
            
            # Simulate latency
            if self.latency_ms > 0:
                time.sleep(self.latency_ms / 1000.0)
            
            # Send to neighbor's inbox
            if neighbor_id in self.peer_inboxes:
                try:
                    self.peer_inboxes[neighbor_id].put(msg, timeout=1.0)
                    logger.debug(f"Peer {self.peer.peer_id} -> {neighbor_id}: sent model ({msg_size} bytes)")
                except queue.Full:
                    logger.warning(f"Peer {self.peer.peer_id} -> {neighbor_id}: inbox full")
    
    def _receive_models_from_neighbors(self, round_id: int, timeout: float = 10.0) -> List[Dict]:
        """Receive models from neighbors
        
        Args:
            round_id: Current round ID
            timeout: Maximum time to wait for messages
        
        Returns:
            List of model info dicts
        """
        neighbors = self.topology.get_neighbors(self.peer.peer_id)
        expected_count = len(neighbors)
        received_models = []
        
        start_time = time.time()
        
        while len(received_models) < expected_count:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                logger.warning(f"Peer {self.peer.peer_id} round {round_id}: timeout waiting for models "
                             f"({len(received_models)}/{expected_count} received)")
                break
            
            try:
                msg = self.inbox.get(timeout=min(remaining_time, 1.0))
                
                # Filter for correct round
                if msg.round_id != round_id:
                    logger.debug(f"Peer {self.peer.peer_id}: ignoring message from wrong round {msg.round_id}")
                    continue
                
                # Track received bytes
                msg_size = msg.size_bytes()
                self.recv_bytes += msg_size
                self.round_recv_bytes += msg_size
                
                # Extract model info
                model_info = {
                    'parameters': msg.parameters,
                    'num_samples': msg.num_samples,
                    'train_loss': msg.train_loss
                }
                received_models.append(model_info)
                
                logger.debug(f"Peer {self.peer.peer_id} <- {msg.from_peer}: received model ({msg_size} bytes)")
                
            except queue.Empty:
                continue
        
        return received_models
    
    def _fetch_model_from_neighbors(self):
        """Fetch latest model from neighbors (for cold-start/rejoin)"""
        # In a real implementation, this would request the latest model
        # For now, we'll just clear the inbox
        try:
            while True:
                self.inbox.get_nowait()
        except queue.Empty:
            pass
        logger.info(f"Peer {self.peer.peer_id} fetched model from neighbors")
    
    def _send_status(self, round_id: int, train_loss: float, eval_loss: float, eval_mse: float):
        """Send status message to coordinator"""
        neighbors = self.topology.get_neighbors(self.peer.peer_id)
        
        status = create_status_message(
            peer_id=self.peer.peer_id,
            round_id=round_id,
            train_loss=train_loss,
            eval_loss=eval_loss,
            eval_mse=eval_mse,
            enabled=self.enabled,
            sent_bytes=self.round_sent_bytes,
            recv_bytes=self.round_recv_bytes,
            aggregate_method=self.peer.aggregate_method,
            num_samples=self.peer.num_train_samples,
            neighbors=neighbors
        )
        
        try:
            self.status_queue.put(status, timeout=1.0)
        except queue.Full:
            logger.error(f"Peer {self.peer.peer_id}: status queue full")
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
