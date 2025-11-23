"""Example usage of DFL Tool"""

import requests
import time
import json

# Base URL
BASE_URL = "http://localhost:8000"


def print_response(response):
    """Pretty print API response"""
    print(json.dumps(response.json(), indent=2))
    print("-" * 80)


def example_basic_training():
    """Example 1: Basic training with 5 peers in ring topology"""
    print("\n=== Example 1: Basic Training ===\n")
    
    # Reset system
    print("Resetting system...")
    response = requests.post(f"{BASE_URL}/api/reset")
    print_response(response)
    
    # Initialize
    print("Initializing 5 peers in ring topology...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 5,
        "hops": [1],
        "data_distribution": "iid",
        "local_epochs": 1,
        "learning_rate": 0.01,
        "batch_size": 32
    })
    print_response(response)
    
    # Start
    print("Starting system...")
    response = requests.post(f"{BASE_URL}/api/start")
    print_response(response)
    
    # Run 10 rounds
    print("Running 10 training rounds...")
    for i in range(10):
        response = requests.post(f"{BASE_URL}/api/step", json={"timeout": 30.0})
        data = response.json()["data"]
        print(f"Round {data['round']}: "
              f"Train Loss={data['global_train_loss']:.4f}, "
              f"Eval Loss={data['global_eval_loss']:.4f}, "
              f"MSE={data['global_eval_mse']:.4f}")
    
    # Get final metrics
    print("\nFinal metrics:")
    response = requests.get(f"{BASE_URL}/api/metrics")
    print_response(response)
    
    # Stop
    print("Stopping system...")
    response = requests.post(f"{BASE_URL}/api/stop")
    print_response(response)


def example_fault_tolerance():
    """Example 2: Fault tolerance with node failure"""
    print("\n=== Example 2: Fault Tolerance ===\n")
    
    # Reset and initialize
    requests.post(f"{BASE_URL}/api/reset")
    
    print("Initializing 8 peers with latency and message drop...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 8,
        "hops": [1],
        "data_distribution": "iid",
        "latency_ms": 50.0,
        "drop_prob": 0.05
    })
    print_response(response)
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Run 5 rounds normally
    print("Running 5 rounds with all peers active...")
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        print(f"Round {data['round']}: MSE={data['global_eval_mse']:.4f}")
    
    # Disable peers 2 and 5
    print("\nDisabling peers 2 and 5...")
    requests.post(f"{BASE_URL}/api/toggle_node", json={
        "peer_id": 2,
        "enabled": False
    })
    requests.post(f"{BASE_URL}/api/toggle_node", json={
        "peer_id": 5,
        "enabled": False
    })
    
    # Run 5 more rounds
    print("Running 5 rounds with peers 2 and 5 disabled...")
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        print(f"Round {data['round']}: "
              f"MSE={data['global_eval_mse']:.4f}, "
              f"Active peers={data['num_active_peers']}/{data['num_total_peers']}")
    
    # Re-enable peers
    print("\nRe-enabling peers 2 and 5 with model fetching...")
    requests.post(f"{BASE_URL}/api/toggle_node", json={
        "peer_id": 2,
        "enabled": True,
        "fetch_model_from_neighbors": True
    })
    requests.post(f"{BASE_URL}/api/toggle_node", json={
        "peer_id": 5,
        "enabled": True,
        "fetch_model_from_neighbors": True
    })
    
    # Run final 5 rounds
    print("Running 5 final rounds with all peers active...")
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        print(f"Round {data['round']}: MSE={data['global_eval_mse']:.4f}")
    
    requests.post(f"{BASE_URL}/api/stop")


def example_heterogeneous_aggregation():
    """Example 3: Heterogeneous aggregation methods"""
    print("\n=== Example 3: Heterogeneous Aggregation ===\n")
    
    # Reset and initialize
    requests.post(f"{BASE_URL}/api/reset")
    
    print("Initializing 6 peers...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 6,
        "hops": [1],
        "data_distribution": "non_iid"
    })
    print_response(response)
    
    # Set different aggregation methods
    print("Setting aggregation methods:")
    print("- Peers 0-2: FedAvg")
    for peer_id in [0, 1, 2]:
        requests.post(f"{BASE_URL}/api/set_aggregate", json={
            "peer_id": peer_id,
            "aggregate_method": "avg"
        })
    
    print("- Peers 3-5: FedProx (mu=0.01)")
    for peer_id in [3, 4, 5]:
        requests.post(f"{BASE_URL}/api/set_aggregate", json={
            "peer_id": peer_id,
            "aggregate_method": "prox",
            "mu": 0.01
        })
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Run training
    print("\nRunning 15 training rounds...")
    for i in range(15):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        if (i + 1) % 5 == 0:
            print(f"Round {data['round']}: "
                  f"Train Loss={data['global_train_loss']:.4f}, "
                  f"MSE={data['global_eval_mse']:.4f}")
    
    requests.post(f"{BASE_URL}/api/stop")


def example_dynamic_topology():
    """Example 4: Dynamic topology changes"""
    print("\n=== Example 4: Dynamic Topology ===\n")
    
    # Reset and initialize
    requests.post(f"{BASE_URL}/api/reset")
    
    print("Initializing 10 peers with 1-hop ring...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 10,
        "hops": [1]
    })
    print_response(response)
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Initial topology
    print("Topology: 1-hop ring")
    response = requests.get(f"{BASE_URL}/api/topology")
    print_response(response)
    
    # Run with 1-hop
    print("Running 5 rounds with 1-hop neighbors...")
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        print(f"Round {data['round']}: MSE={data['global_eval_mse']:.4f}")
    
    # Expand to 2-hop
    print("\nExpanding to 2-hop ring...")
    requests.post(f"{BASE_URL}/api/set_neighbors", json={
        "hops": [1, 2]
    })
    
    response = requests.get(f"{BASE_URL}/api/topology")
    print_response(response)
    
    print("Running 5 rounds with 2-hop neighbors...")
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        print(f"Round {data['round']}: MSE={data['global_eval_mse']:.4f}")
    
    # Custom neighbors for peer 0
    print("\nSetting custom neighbors for peer 0: [1, 3, 5, 7, 9]")
    requests.post(f"{BASE_URL}/api/set_neighbors", json={
        "peer_id": 0,
        "neighbors": [1, 3, 5, 7, 9]
    })
    
    print("Running 5 final rounds...")
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        print(f"Round {data['round']}: MSE={data['global_eval_mse']:.4f}")
    
    requests.post(f"{BASE_URL}/api/stop")


def example_bandwidth_analysis():
    """Example 5: Bandwidth analysis"""
    print("\n=== Example 5: Bandwidth Analysis ===\n")
    
    # Reset and initialize
    requests.post(f"{BASE_URL}/api/reset")
    
    print("Initializing 5 peers...")
    requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 5,
        "hops": [1]
    })
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Run rounds
    print("Running 5 training rounds...")
    for i in range(5):
        requests.post(f"{BASE_URL}/api/step")
    
    # Get bandwidth statistics
    print("\nBandwidth statistics:")
    response = requests.get(f"{BASE_URL}/api/bandwidth")
    data = response.json()["data"]
    
    print("\nPer-round bandwidth:")
    for i, round_bw in enumerate(data["per_round"], 1):
        print(f"Round {i}: Total sent={round_bw['total_sent']} bytes, "
              f"Total recv={round_bw['total_recv']} bytes")
    
    print("\nCumulative bandwidth matrix:")
    import numpy as np
    matrix = np.array(data["cumulative_matrix"])
    print(matrix)
    
    requests.post(f"{BASE_URL}/api/stop")


if __name__ == "__main__":
    print("=" * 80)
    print("DFL Tool - Example Usage")
    print("=" * 80)
    print("\nMake sure the API server is running on http://localhost:8000")
    print("Start it with: python api.py")
    print("\n" + "=" * 80)
    
    try:
        # Test connection
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Connected to API server")
        
        # Run examples
        example_basic_training()
        time.sleep(2)
        
        example_fault_tolerance()
        time.sleep(2)
        
        example_heterogeneous_aggregation()
        time.sleep(2)
        
        example_dynamic_topology()
        time.sleep(2)
        
        example_bandwidth_analysis()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API server")
        print("Please start the server with: python api.py")
    except Exception as e:
        print(f"✗ Error: {e}")
