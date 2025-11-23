"""Message protocol for peer communication"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import time


@dataclass
class ModelMessage:
    """Message containing model information from a peer"""
    type: str = "MODEL"
    from_peer: int = 0
    round_id: int = 0
    parameters: Optional[Dict] = None
    num_samples: int = 0
    train_loss: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def size_bytes(self) -> int:
        """Estimate message size in bytes"""
        if self.parameters is None:
            return 100  # Metadata only
        
        # Estimate based on model parameters
        total_bytes = 100  # Metadata overhead
        for key, tensor in self.parameters.items():
            total_bytes += tensor.element_size() * tensor.nelement()
        return total_bytes


@dataclass
class ControlMessage:
    """Control message from coordinator to peer"""
    type: str = "CONTROL"
    cmd: str = "START_ROUND"  # START_ROUND, ENABLE, DISABLE, STOP_THREAD, SET_AGGREGATE
    round_id: int = 0
    aggregate_method: Optional[str] = None
    mu: Optional[float] = None
    local_epochs: Optional[int] = None
    timeout: Optional[float] = None
    fetch_model: bool = False  # For cold-start/rejoin scenarios
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StatusMessage:
    """Status report from peer to coordinator"""
    type: str = "STATUS"
    peer_id: int = 0
    round_id: int = 0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    eval_mse: float = 0.0  # MSE for anomaly detection
    enabled: bool = True
    sent_bytes: int = 0
    recv_bytes: int = 0
    aggregate_method: str = "avg"
    num_samples: int = 0
    neighbors: list = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.neighbors is None:
            self.neighbors = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


def create_model_message(peer_id: int, round_id: int, parameters: Dict, 
                        num_samples: int, train_loss: float) -> ModelMessage:
    """Factory function to create model message"""
    return ModelMessage(
        from_peer=peer_id,
        round_id=round_id,
        parameters=parameters,
        num_samples=num_samples,
        train_loss=train_loss
    )


def create_control_message(cmd: str, round_id: int = 0, **kwargs) -> ControlMessage:
    """Factory function to create control message"""
    return ControlMessage(cmd=cmd, round_id=round_id, **kwargs)


def create_status_message(peer_id: int, round_id: int, **kwargs) -> StatusMessage:
    """Factory function to create status message"""
    return StatusMessage(peer_id=peer_id, round_id=round_id, **kwargs)
