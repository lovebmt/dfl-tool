"""Quick test script to verify topology implementations"""

from topology import create_topology, RingTopology, LineTopology, MeshTopology, FullyConnectedTopology

def test_ring_topology():
    print("Testing Ring Topology...")
    topo = create_topology('ring', 5, hops=[1])
    assert topo.num_peers == 5
    neighbors = topo.get_neighbors(0)
    assert 1 in neighbors and 4 in neighbors
    info = topo.get_topology_info()
    assert info['type'] == 'ring'
    print("✓ Ring topology works")

def test_line_topology():
    print("Testing Line Topology...")
    topo = create_topology('line', 5, bidirectional=True)
    assert topo.num_peers == 5
    # Edge peers have 1 neighbor
    assert len(topo.get_neighbors(0)) == 1
    assert len(topo.get_neighbors(4)) == 1
    # Middle peers have 2 neighbors
    assert len(topo.get_neighbors(2)) == 2
    info = topo.get_topology_info()
    assert info['type'] == 'line'
    print("✓ Line topology works")

def test_mesh_topology():
    print("Testing Mesh Topology...")
    edges = [(0, 1), (1, 2), (2, 0)]
    topo = create_topology('mesh', 3, edges=edges)
    assert topo.num_peers == 3
    assert 1 in topo.get_neighbors(0)
    assert 2 in topo.get_neighbors(1)
    assert 0 in topo.get_neighbors(2)
    info = topo.get_topology_info()
    assert info['type'] == 'mesh'
    print("✓ Mesh topology works")

def test_fully_connected():
    print("Testing Fully Connected Topology...")
    topo = create_topology('full', 4)
    assert topo.num_peers == 4
    # Each peer connects to all others
    for i in range(4):
        neighbors = topo.get_neighbors(i)
        assert len(neighbors) == 3  # n-1 neighbors
        assert i not in neighbors
    info = topo.get_topology_info()
    assert info['type'] == 'fully_connected'
    assert info['total_edges'] == 12  # 4 * 3
    print("✓ Fully connected topology works")

def test_mesh_vs_fully_connected():
    print("\nComparing Mesh (connectivity=1.0) vs Fully Connected...")
    n = 5
    
    # Fully connected
    fc_topo = FullyConnectedTopology(n)
    fc_edges = set(fc_topo.get_all_edges())
    
    # Mesh with connectivity=1.0 (might not be exactly the same due to randomness)
    # But we can create a fully connected mesh explicitly
    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    mesh_topo = MeshTopology(n, edges=all_edges)
    mesh_edges = set(mesh_topo.get_all_edges())
    
    print(f"Fully Connected edges: {len(fc_edges)}")
    print(f"Mesh (all edges) edges: {len(mesh_edges)}")
    assert fc_edges == mesh_edges
    
    print("✓ Mesh can represent fully connected, but FullyConnected is simpler")

if __name__ == "__main__":
    print("="*60)
    print("TOPOLOGY VERIFICATION TESTS")
    print("="*60)
    
    test_ring_topology()
    test_line_topology()
    test_mesh_topology()
    test_fully_connected()
    test_mesh_vs_fully_connected()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nConclusion:")
    print("- StarTopology: REMOVED (centralized, not truly decentralized)")
    print("- MeshTopology: Flexible, can represent any graph")
    print("- FullyConnectedTopology: Optimized for complete graphs")
    print("- Mesh can simulate FullyConnected, but FC is clearer & faster")
