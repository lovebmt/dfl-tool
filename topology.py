"""Topology management for decentralized network"""

import threading
from typing import Set, Dict, List, Optional


class RingTopology:
    """Ring topology with configurable neighbor hops
    
    Supports dynamic neighbor configuration and atomic updates
    """
    
    def __init__(self, num_peers: int, hops: Set[int] = None):
        """Initialize ring topology
        
        Args:
            num_peers: Total number of peers in the network
            hops: Set of hop distances (default: {1} for immediate neighbors)
        """
        self.num_peers = num_peers
        self.hops = hops if hops is not None else {1}
        
        # Optional per-peer custom neighbors (overrides hop-based calculation)
        self._custom_neighbors: Dict[int, List[int]] = {}
        
        # Lock for thread-safe topology updates
        self._lock = threading.Lock()
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get list of neighbors for a peer
        
        Args:
            peer_id: Peer identifier
        
        Returns:
            List of neighbor peer IDs
        """
        with self._lock:
            # Check if custom neighbors are defined for this peer
            if peer_id in self._custom_neighbors:
                return self._custom_neighbors[peer_id].copy()
            
            # Otherwise, calculate based on hops
            neighbors = []
            for h in self.hops:
                # Add neighbors at distance h in both directions
                right_neighbor = (peer_id + h) % self.num_peers
                left_neighbor = (peer_id - h) % self.num_peers
                
                if right_neighbor != peer_id:
                    neighbors.append(right_neighbor)
                if left_neighbor != peer_id and left_neighbor != right_neighbor:
                    neighbors.append(left_neighbor)
            
            return sorted(list(set(neighbors)))
    
    def set_neighbors(self, peer_id: Optional[int] = None, 
                     neighbors: Optional[List[int]] = None,
                     hops: Optional[Set[int]] = None):
        """Update topology configuration (atomic operation)
        
        Args:
            peer_id: Specific peer to update (None for global update)
            neighbors: Custom neighbor list for the peer
            hops: New global hop set (applies to all peers without custom neighbors)
        """
        with self._lock:
            if hops is not None:
                # Update global hops
                self.hops = set(hops)
            
            if peer_id is not None and neighbors is not None:
                # Set custom neighbors for specific peer
                self._custom_neighbors[peer_id] = list(neighbors)
            elif peer_id is not None and neighbors is None:
                # Clear custom neighbors for this peer (revert to hop-based)
                if peer_id in self._custom_neighbors:
                    del self._custom_neighbors[peer_id]
    
    def get_all_edges(self) -> List[tuple]:
        """Get all edges in the topology
        
        Returns:
            List of (source, destination) tuples
        """
        edges = []
        for peer_id in range(self.num_peers):
            neighbors = self.get_neighbors(peer_id)
            for neighbor in neighbors:
                edges.append((peer_id, neighbor))
        return edges
    
    def reset_custom_neighbors(self):
        """Clear all custom neighbor configurations"""
        with self._lock:
            self._custom_neighbors.clear()
    
    def get_topology_info(self) -> Dict:
        """Get topology configuration information
        
        Returns:
            Dict with topology details
        """
        with self._lock:
            return {
                'num_peers': self.num_peers,
                'hops': list(self.hops),
                'custom_neighbors': {k: v for k, v in self._custom_neighbors.items()},
                'total_edges': len(self.get_all_edges())
            }


class FullyConnectedTopology:
    """Fully connected topology - every peer connects to all others"""
    
    def __init__(self, num_peers: int):
        self.num_peers = num_peers
        self._lock = threading.Lock()
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get all other peers as neighbors"""
        return [i for i in range(self.num_peers) if i != peer_id]
    
    def set_neighbors(self, **kwargs):
        """No-op for fully connected topology"""
        pass
    
    def get_all_edges(self) -> List[tuple]:
        """Get all edges"""
        edges = []
        for i in range(self.num_peers):
            for j in range(self.num_peers):
                if i != j:
                    edges.append((i, j))
        return edges
    
    def get_topology_info(self) -> Dict:
        return {
            'type': 'fully_connected',
            'num_peers': self.num_peers,
            'total_edges': self.num_peers * (self.num_peers - 1)
        }


class StarTopology:
    """Star topology - one central peer, all others connect to it"""
    
    def __init__(self, num_peers: int, center_peer: int = 0):
        self.num_peers = num_peers
        self.center_peer = center_peer
        self._lock = threading.Lock()
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get neighbors based on star topology"""
        with self._lock:
            if peer_id == self.center_peer:
                # Center connects to all others
                return [i for i in range(self.num_peers) if i != self.center_peer]
            else:
                # Others only connect to center
                return [self.center_peer]
    
    def set_neighbors(self, center_peer: Optional[int] = None, **kwargs):
        """Update center peer"""
        with self._lock:
            if center_peer is not None:
                self.center_peer = center_peer
    
    def get_all_edges(self) -> List[tuple]:
        edges = []
        for i in range(self.num_peers):
            if i != self.center_peer:
                edges.append((i, self.center_peer))
                edges.append((self.center_peer, i))
        return edges
    
    def get_topology_info(self) -> Dict:
        return {
            'type': 'star',
            'num_peers': self.num_peers,
            'center_peer': self.center_peer,
            'total_edges': 2 * (self.num_peers - 1)
        }
