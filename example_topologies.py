"""Example demonstrating different network topologies for DFL

This script shows how to use various topologies:
- Ring: Peers connected in a ring with configurable hops
- Line: Peers connected in a linear chain
- Mesh: Arbitrary connections between peers
- Fully Connected: Every peer connected to all others
"""

from coordinator import Coordinator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_ring_topology():
    """Example: Ring topology with 1-hop neighbors"""
    print("\n" + "="*60)
    print("EXAMPLE 1: RING TOPOLOGY (1-hop)")
    print("="*60)
    
    coordinator = Coordinator()
    coordinator.initialize(
        num_peers=5,
        topology_type="ring",
        topology_params={'hops': [1]},  # Only immediate neighbors
        data_distribution="iid",
        local_epochs=1,
        dataset="bearing"
    )
    
    print("\nTopology Info:")
    print(coordinator.topology.get_topology_info())
    
    print("\nNeighbor connections:")
    for peer_id in range(5):
        neighbors = coordinator.topology.get_neighbors(peer_id)
        print(f"Peer {peer_id} -> Neighbors: {neighbors}")
    
    coordinator.reset()


def example_ring_multihop():
    """Example: Ring topology with multi-hop neighbors"""
    print("\n" + "="*60)
    print("EXAMPLE 2: RING TOPOLOGY (Multi-hop: 1 and 2 hops)")
    print("="*60)
    
    coordinator = Coordinator()
    coordinator.initialize(
        num_peers=6,
        topology_type="ring",
        topology_params={'hops': [1, 2]},  # 1-hop and 2-hop neighbors
        data_distribution="iid",
        local_epochs=1,
        dataset="bearing"
    )
    
    print("\nTopology Info:")
    print(coordinator.topology.get_topology_info())
    
    print("\nNeighbor connections:")
    for peer_id in range(6):
        neighbors = coordinator.topology.get_neighbors(peer_id)
        print(f"Peer {peer_id} -> Neighbors: {neighbors}")
    
    coordinator.reset()


def example_line_topology():
    """Example: Line topology (chain)"""
    print("\n" + "="*60)
    print("EXAMPLE 3: LINE TOPOLOGY (Bidirectional chain)")
    print("="*60)
    
    coordinator = Coordinator()
    coordinator.initialize(
        num_peers=5,
        topology_type="line",
        topology_params={'bidirectional': True},
        data_distribution="iid",
        local_epochs=1,
        dataset="bearing"
    )
    
    print("\nTopology Info:")
    print(coordinator.topology.get_topology_info())
    
    print("\nNeighbor connections (note: no wraparound like ring):")
    for peer_id in range(5):
        neighbors = coordinator.topology.get_neighbors(peer_id)
        print(f"Peer {peer_id} -> Neighbors: {neighbors}")
    
    coordinator.reset()


def example_mesh_topology():
    """Example: Mesh topology with custom edges"""
    print("\n" + "="*60)
    print("EXAMPLE 4: MESH TOPOLOGY (Custom edges)")
    print("="*60)
    
    # Define custom edges
    custom_edges = [
        (0, 1), (0, 2),
        (1, 2), (1, 3),
        (2, 3), (2, 4),
        (3, 4)
    ]
    
    coordinator = Coordinator()
    coordinator.initialize(
        num_peers=5,
        topology_type="mesh",
        topology_params={'edges': custom_edges},
        data_distribution="iid",
        local_epochs=1,
        dataset="bearing"
    )
    
    print("\nTopology Info:")
    print(coordinator.topology.get_topology_info())
    
    print("\nNeighbor connections:")
    for peer_id in range(5):
        neighbors = coordinator.topology.get_neighbors(peer_id)
        print(f"Peer {peer_id} -> Neighbors: {neighbors}")
    
    coordinator.reset()


def example_mesh_random():
    """Example: Mesh topology with random connectivity"""
    print("\n" + "="*60)
    print("EXAMPLE 5: MESH TOPOLOGY (Random connectivity: 0.4)")
    print("="*60)
    
    coordinator = Coordinator()
    coordinator.initialize(
        num_peers=5,
        topology_type="mesh",
        topology_params={'connectivity': 0.4},  # 40% connection probability
        data_distribution="iid",
        local_epochs=1,
        dataset="bearing"
    )
    
    print("\nTopology Info:")
    print(coordinator.topology.get_topology_info())
    
    print("\nNeighbor connections (random):")
    for peer_id in range(5):
        neighbors = coordinator.topology.get_neighbors(peer_id)
        print(f"Peer {peer_id} -> Neighbors: {neighbors}")
    
    coordinator.reset()


def example_fully_connected():
    """Example: Fully connected topology"""
    print("\n" + "="*60)
    print("EXAMPLE 6: FULLY CONNECTED TOPOLOGY")
    print("="*60)
    
    coordinator = Coordinator()
    coordinator.initialize(
        num_peers=5,
        topology_type="full",
        data_distribution="iid",
        local_epochs=1,
        dataset="bearing"
    )
    
    print("\nTopology Info:")
    print(coordinator.topology.get_topology_info())
    
    print("\nNeighbor connections (all-to-all):")
    for peer_id in range(5):
        neighbors = coordinator.topology.get_neighbors(peer_id)
        print(f"Peer {peer_id} -> Neighbors: {neighbors}")
    
    coordinator.reset()


def example_training_comparison():
    """Example: Compare training with different topologies"""
    print("\n" + "="*60)
    print("EXAMPLE 7: TRAINING COMPARISON")
    print("="*60)
    
    topologies = [
        ("ring", {'hops': [1]}),
        ("line", {'bidirectional': True}),
        ("mesh", {'connectivity': 0.5}),
        ("full", {})
    ]
    
    results = {}
    
    for topo_type, topo_params in topologies:
        print(f"\n--- Training with {topo_type.upper()} topology ---")
        
        coordinator = Coordinator()
        coordinator.initialize(
            num_peers=4,
            topology_type=topo_type,
            topology_params=topo_params,
            data_distribution="iid",
            local_epochs=2,
            learning_rate=0.001,
            batch_size=32,
            dataset="bearing"
        )
        
        coordinator.start()
        
        # Run 3 rounds
        for round_num in range(3):
            metrics = coordinator.step(timeout=10.0)
            print(f"Round {round_num + 1}: Loss={metrics.get('avg_loss', 0):.4f}, "
                  f"MSE={metrics.get('avg_mse', 0):.4f}")
        
        final_metrics = coordinator.get_metrics()
        results[topo_type] = final_metrics
        
        coordinator.stop()
        coordinator.reset()
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for topo_type in results:
        metrics = results[topo_type]
        if metrics and len(metrics) > 0:
            final = metrics[-1]
            print(f"{topo_type.upper():20s}: Loss={final.get('avg_loss', 0):.4f}, "
                  f"MSE={final.get('avg_mse', 0):.4f}")


if __name__ == "__main__":
    # Run all examples
    example_ring_topology()
    example_ring_multihop()
    example_line_topology()
    example_mesh_topology()
    example_mesh_random()
    example_fully_connected()
    
    # Uncomment to run training comparison (takes longer)
    # example_training_comparison()
    
    print("\n" + "="*60)
    print("All topology examples completed!")
    print("="*60)
