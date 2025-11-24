"""Topology management for decentralized network"""

import threading
from typing import Set, Dict, List, Optional
from abc import ABC, abstractmethod


class BaseTopology(ABC):
    """Abstract base class for network topologies"""
    
    def __init__(self, num_peers: int):
        """Initialize base topology
        
        Args:
            num_peers: Total number of peers in the network
        """
        self.num_peers = num_peers
        self._lock = threading.Lock()
    
    @abstractmethod
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get list of neighbors for a peer
        
        Args:
            peer_id: Peer identifier
        
        Returns:
            List of neighbor peer IDs
        """
        pass
    
    @abstractmethod
    def get_topology_info(self) -> Dict:
        """Get topology configuration information
        
        Returns:
            Dict with topology details
        """
        pass
    
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
    
    def set_neighbors(self, **kwargs):
        """Update topology configuration (optional, topology-specific)"""
        pass


class RingTopology(BaseTopology):
    """Ring topology with configurable neighbor hops
    
    Supports dynamic neighbor configuration and atomic updates
    """
    
    def __init__(self, num_peers: int, hops: Set[int] = None):
        """Initialize ring topology
        
        Args:
            num_peers: Total number of peers in the network
            hops: Set of hop distances (default: {1} for immediate neighbors)
        """
        super().__init__(num_peers)
        self.hops = hops if hops is not None else {1}
        
        # Optional per-peer custom neighbors (overrides hop-based calculation)
        self._custom_neighbors: Dict[int, List[int]] = {}
    
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
        # Get edges first (avoid deadlock - get_all_edges calls get_neighbors which uses lock)
        all_edges = self.get_all_edges()
        # Convert directed edges to undirected connections
        unique_connections = set()
        for src, dst in all_edges:
            # Add edge as sorted tuple to avoid counting (A,B) and (B,A) separately
            unique_connections.add(tuple(sorted([src, dst])))
        
        with self._lock:
            return {
                'type': 'ring',
                'num_peers': self.num_peers,
                'hops': list(self.hops),
                'custom_neighbors': {k: v for k, v in self._custom_neighbors.items()},
                'total_edges': len(unique_connections)
            }


class LineTopology(BaseTopology):
    """Line topology - peers connected in a linear chain
    
    Peer 0 -- Peer 1 -- Peer 2 -- ... -- Peer N-1
    No wraparound like ring topology
    """
    
    def __init__(self, num_peers: int, bidirectional: bool = True):
        """Initialize line topology
        
        Args:
            num_peers: Total number of peers in the network
            bidirectional: If True, edges go both directions (default)
        """
        super().__init__(num_peers)
        self.bidirectional = bidirectional
        self._custom_neighbors: Dict[int, List[int]] = {}
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get neighbors for a peer in line topology
        
        Args:
            peer_id: Peer identifier
        
        Returns:
            List of neighbor peer IDs
        """
        with self._lock:
            # Check if custom neighbors are defined
            if peer_id in self._custom_neighbors:
                return self._custom_neighbors[peer_id].copy()
            
            neighbors = []
            # Left neighbor (if exists)
            if peer_id > 0:
                neighbors.append(peer_id - 1)
            # Right neighbor (if exists)
            if peer_id < self.num_peers - 1:
                neighbors.append(peer_id + 1)
            
            return neighbors
    
    def set_neighbors(self, peer_id: Optional[int] = None, 
                     neighbors: Optional[List[int]] = None,
                     bidirectional: Optional[bool] = None):
        """Update topology configuration
        
        Args:
            peer_id: Specific peer to update (None for global update)
            neighbors: Custom neighbor list for the peer
            bidirectional: Update bidirectional setting
        """
        with self._lock:
            if bidirectional is not None:
                self.bidirectional = bidirectional
            
            if peer_id is not None and neighbors is not None:
                self._custom_neighbors[peer_id] = list(neighbors)
            elif peer_id is not None and neighbors is None:
                if peer_id in self._custom_neighbors:
                    del self._custom_neighbors[peer_id]
    
    def get_topology_info(self) -> Dict:
        """Get topology configuration information"""
        # Get edges first (avoid deadlock)
        all_edges = self.get_all_edges()
        # Convert directed edges to undirected connections
        unique_connections = set()
        for src, dst in all_edges:
            unique_connections.add(tuple(sorted([src, dst])))
        
        with self._lock:
            return {
                'type': 'line',
                'num_peers': self.num_peers,
                'bidirectional': self.bidirectional,
                'custom_neighbors': {k: v for k, v in self._custom_neighbors.items()},
                'total_edges': len(unique_connections)
            }


class MeshTopology(BaseTopology):
    """Mesh topology - arbitrary peer connections
    
    Supports custom connection patterns defined per peer
    """
    
    def __init__(self, num_peers: int, connectivity: float = 0.5, 
                 edges: Optional[List[tuple]] = None):
        """Initialize mesh topology
        
        Args:
            num_peers: Total number of peers in the network
            connectivity: Random connectivity ratio (0-1) if edges not specified
            edges: List of (source, dest) tuples for explicit edges
        """
        super().__init__(num_peers)
        self.connectivity = connectivity
        self._neighbors: Dict[int, List[int]] = {i: [] for i in range(num_peers)}
        
        if edges is not None:
            # Build from explicit edges
            for src, dst in edges:
                if dst not in self._neighbors[src]:
                    self._neighbors[src].append(dst)
        else:
            # Generate random mesh based on connectivity
            import random
            for i in range(num_peers):
                for j in range(num_peers):
                    if i != j and random.random() < connectivity:
                        if j not in self._neighbors[i]:
                            self._neighbors[i].append(j)
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get neighbors for a peer
        
        Args:
            peer_id: Peer identifier
        
        Returns:
            List of neighbor peer IDs
        """
        with self._lock:
            return self._neighbors.get(peer_id, []).copy()
    
    def set_neighbors(self, peer_id: Optional[int] = None, 
                     neighbors: Optional[List[int]] = None,
                     edges: Optional[List[tuple]] = None):
        """Update mesh topology
        
        Args:
            peer_id: Specific peer to update
            neighbors: New neighbor list for the peer
            edges: List of (source, dest) tuples to rebuild topology
        """
        with self._lock:
            if edges is not None:
                # Rebuild entire topology from edges
                self._neighbors = {i: [] for i in range(self.num_peers)}
                for src, dst in edges:
                    if dst not in self._neighbors[src]:
                        self._neighbors[src].append(dst)
            elif peer_id is not None and neighbors is not None:
                self._neighbors[peer_id] = list(neighbors)
    
    def add_edge(self, src: int, dst: int):
        """Add a single edge to the mesh
        
        Args:
            src: Source peer ID
            dst: Destination peer ID
        """
        with self._lock:
            if dst not in self._neighbors[src]:
                self._neighbors[src].append(dst)
    
    def remove_edge(self, src: int, dst: int):
        """Remove a single edge from the mesh
        
        Args:
            src: Source peer ID
            dst: Destination peer ID
        """
        with self._lock:
            if dst in self._neighbors[src]:
                self._neighbors[src].remove(dst)
    
    def get_topology_info(self) -> Dict:
        """Get topology configuration information"""
        # Get edges first (avoid deadlock)
        all_edges = self.get_all_edges()
        # Convert directed edges to undirected connections
        unique_connections = set()
        for src, dst in all_edges:
            unique_connections.add(tuple(sorted([src, dst])))
        
        with self._lock:
            return {
                'type': 'mesh',
                'num_peers': self.num_peers,
                'connectivity': self.connectivity,
                'neighbors': {k: v.copy() for k, v in self._neighbors.items()},
                'total_edges': len(unique_connections)
            }


class FullyConnectedTopology(BaseTopology):
    """Fully connected topology - every peer connects to all others (complete graph)"""
    
    def __init__(self, num_peers: int):
        """Initialize fully connected topology
        
        Args:
            num_peers: Total number of peers in the network
        """
        super().__init__(num_peers)
    
    def get_neighbors(self, peer_id: int) -> List[int]:
        """Get all other peers as neighbors
        
        Args:
            peer_id: Peer identifier
        
        Returns:
            List of all other peer IDs
        """
        return [i for i in range(self.num_peers) if i != peer_id]
    
    def get_topology_info(self) -> Dict:
        """Get topology configuration information"""
        # For fully connected graph: n*(n-1)/2 undirected edges
        num_undirected_edges = self.num_peers * (self.num_peers - 1) // 2
        return {
            'type': 'fully_connected',
            'num_peers': self.num_peers,
            'total_edges': num_undirected_edges
        }


def create_topology(topology_type: str, num_peers: int, **kwargs) -> BaseTopology:
    """Factory function to create topology instances
    
    Args:
        topology_type: Type of topology ('ring', 'line', 'mesh', 'full')
        num_peers: Number of peers in the network
        **kwargs: Topology-specific parameters
            Ring: hops (Set[int] or List[int])
            Line: bidirectional (bool)
            Mesh: connectivity (float), edges (List[tuple])
    
    Returns:
        Topology instance
    
    Examples:
        >>> topo = create_topology('ring', 5, hops=[1, 2])
        >>> topo = create_topology('line', 10, bidirectional=True)
        >>> topo = create_topology('mesh', 8, connectivity=0.5)
        >>> topo = create_topology('full', 4)
    """
    topology_type = topology_type.lower()
    
    if topology_type in ['ring', 'ring_topology']:
        hops = kwargs.get('hops', [1])
        if isinstance(hops, list):
            hops = set(hops)
        return RingTopology(num_peers, hops)
    
    elif topology_type in ['line', 'line_topology', 'chain']:
        bidirectional = kwargs.get('bidirectional', True)
        return LineTopology(num_peers, bidirectional)
    
    elif topology_type in ['mesh', 'mesh_topology']:
        connectivity = kwargs.get('connectivity', 0.5)
        edges = kwargs.get('edges', None)
        return MeshTopology(num_peers, connectivity, edges)
    
    elif topology_type in ['full', 'fully_connected', 'complete']:
        return FullyConnectedTopology(num_peers)
    
    else:
        raise ValueError(
            f"Unknown topology type: {topology_type}. "
            f"Supported types: ring, line, mesh, full"
        )
