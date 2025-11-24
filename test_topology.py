"""Unit tests for topology implementations"""

import unittest
from topology import (
    RingTopology, LineTopology, MeshTopology, 
    StarTopology, FullyConnectedTopology, create_topology
)


class TestRingTopology(unittest.TestCase):
    """Test Ring topology"""
    
    def test_single_hop(self):
        """Test ring with 1-hop neighbors"""
        topo = RingTopology(num_peers=5, hops={1})
        
        # Peer 0 should connect to 1 and 4
        self.assertEqual(sorted(topo.get_neighbors(0)), [1, 4])
        # Peer 2 should connect to 1 and 3
        self.assertEqual(sorted(topo.get_neighbors(2)), [1, 3])
    
    def test_multi_hop(self):
        """Test ring with multi-hop neighbors"""
        topo = RingTopology(num_peers=6, hops={1, 2})
        
        # Peer 0 should connect to 1, 2, 4, 5
        neighbors = topo.get_neighbors(0)
        self.assertEqual(sorted(neighbors), [1, 2, 4, 5])
    
    def test_topology_info(self):
        """Test topology info"""
        topo = RingTopology(num_peers=5, hops={1})
        info = topo.get_topology_info()
        
        self.assertEqual(info['type'], 'ring')
        self.assertEqual(info['num_peers'], 5)
        self.assertEqual(info['hops'], [1])


class TestLineTopology(unittest.TestCase):
    """Test Line topology"""
    
    def test_edge_peers(self):
        """Test edge peers have only one neighbor"""
        topo = LineTopology(num_peers=5)
        
        # First peer only connects to peer 1
        self.assertEqual(topo.get_neighbors(0), [1])
        # Last peer only connects to peer 3
        self.assertEqual(topo.get_neighbors(4), [3])
    
    def test_middle_peer(self):
        """Test middle peer has two neighbors"""
        topo = LineTopology(num_peers=5)
        
        # Middle peer connects to both neighbors
        self.assertEqual(sorted(topo.get_neighbors(2)), [1, 3])
    
    def test_topology_info(self):
        """Test topology info"""
        topo = LineTopology(num_peers=5, bidirectional=True)
        info = topo.get_topology_info()
        
        self.assertEqual(info['type'], 'line')
        self.assertEqual(info['num_peers'], 5)
        self.assertTrue(info['bidirectional'])


class TestMeshTopology(unittest.TestCase):
    """Test Mesh topology"""
    
    def test_custom_edges(self):
        """Test mesh with custom edges"""
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        topo = MeshTopology(num_peers=4, edges=edges)
        
        # Peer 0 should connect to 1 and 2
        self.assertEqual(sorted(topo.get_neighbors(0)), [1, 2])
        # Peer 1 should connect to 3
        self.assertEqual(topo.get_neighbors(1), [3])
    
    def test_add_remove_edge(self):
        """Test adding and removing edges"""
        topo = MeshTopology(num_peers=4, edges=[(0, 1)])
        
        # Add edge
        topo.add_edge(0, 2)
        self.assertEqual(sorted(topo.get_neighbors(0)), [1, 2])
        
        # Remove edge
        topo.remove_edge(0, 1)
        self.assertEqual(topo.get_neighbors(0), [2])
    
    def test_topology_info(self):
        """Test topology info"""
        edges = [(0, 1), (1, 2)]
        topo = MeshTopology(num_peers=3, edges=edges)
        info = topo.get_topology_info()
        
        self.assertEqual(info['type'], 'mesh')
        self.assertEqual(info['num_peers'], 3)


class TestStarTopology(unittest.TestCase):
    """Test Star topology"""
    
    def test_center_connections(self):
        """Test center peer connects to all others"""
        topo = StarTopology(num_peers=5, center_peer=0)
        
        # Center connects to all others
        self.assertEqual(sorted(topo.get_neighbors(0)), [1, 2, 3, 4])
    
    def test_leaf_connections(self):
        """Test leaf peers only connect to center"""
        topo = StarTopology(num_peers=5, center_peer=0)
        
        # Leaves only connect to center
        self.assertEqual(topo.get_neighbors(1), [0])
        self.assertEqual(topo.get_neighbors(4), [0])
    
    def test_topology_info(self):
        """Test topology info"""
        topo = StarTopology(num_peers=5, center_peer=2)
        info = topo.get_topology_info()
        
        self.assertEqual(info['type'], 'star')
        self.assertEqual(info['num_peers'], 5)
        self.assertEqual(info['center_peer'], 2)


class TestFullyConnectedTopology(unittest.TestCase):
    """Test Fully Connected topology"""
    
    def test_all_connections(self):
        """Test every peer connects to all others"""
        topo = FullyConnectedTopology(num_peers=4)
        
        # Each peer connects to all others
        self.assertEqual(sorted(topo.get_neighbors(0)), [1, 2, 3])
        self.assertEqual(sorted(topo.get_neighbors(2)), [0, 1, 3])
    
    def test_topology_info(self):
        """Test topology info"""
        topo = FullyConnectedTopology(num_peers=5)
        info = topo.get_topology_info()
        
        self.assertEqual(info['type'], 'fully_connected')
        self.assertEqual(info['num_peers'], 5)
        self.assertEqual(info['total_edges'], 20)  # 5 * 4


class TestTopologyFactory(unittest.TestCase):
    """Test topology factory function"""
    
    def test_create_ring(self):
        """Test creating ring topology"""
        topo = create_topology('ring', 5, hops=[1, 2])
        self.assertIsInstance(topo, RingTopology)
        self.assertEqual(topo.hops, {1, 2})
    
    def test_create_line(self):
        """Test creating line topology"""
        topo = create_topology('line', 5, bidirectional=False)
        self.assertIsInstance(topo, LineTopology)
        self.assertFalse(topo.bidirectional)
    
    def test_create_mesh(self):
        """Test creating mesh topology"""
        edges = [(0, 1), (1, 2)]
        topo = create_topology('mesh', 3, edges=edges)
        self.assertIsInstance(topo, MeshTopology)
    
    def test_create_star(self):
        """Test creating star topology"""
        topo = create_topology('star', 6, center_peer=3)
        self.assertIsInstance(topo, StarTopology)
        self.assertEqual(topo.center_peer, 3)
    
    def test_create_full(self):
        """Test creating fully connected topology"""
        topo = create_topology('full', 4)
        self.assertIsInstance(topo, FullyConnectedTopology)
    
    def test_invalid_topology(self):
        """Test invalid topology type raises error"""
        with self.assertRaises(ValueError):
            create_topology('invalid', 5)


class TestEdgeCounts(unittest.TestCase):
    """Test edge count calculations"""
    
    def test_ring_edges(self):
        """Test ring topology edge count"""
        topo = RingTopology(num_peers=5, hops={1})
        edges = topo.get_all_edges()
        self.assertEqual(len(edges), 10)  # 2 * 5
    
    def test_line_edges(self):
        """Test line topology edge count"""
        topo = LineTopology(num_peers=5)
        edges = topo.get_all_edges()
        self.assertEqual(len(edges), 8)  # 2 * (5-1)
    
    def test_star_edges(self):
        """Test star topology edge count"""
        topo = StarTopology(num_peers=5, center_peer=0)
        edges = topo.get_all_edges()
        self.assertEqual(len(edges), 8)  # 2 * (5-1)
    
    def test_full_edges(self):
        """Test fully connected topology edge count"""
        topo = FullyConnectedTopology(num_peers=5)
        edges = topo.get_all_edges()
        self.assertEqual(len(edges), 20)  # 5 * 4


if __name__ == '__main__':
    unittest.main()
