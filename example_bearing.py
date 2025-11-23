"""Example usage of DFL Tool with Bearing Dataset"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"


def print_response(response):
    """Pretty print API response"""
    print(json.dumps(response.json(), indent=2))
    print("-" * 80)


def example_bearing_basic():
    """Example: Basic training with bearing dataset"""
    print("\n=== Bearing Dataset - Basic Training ===\n")
    
    # Reset system
    print("Resetting system...")
    response = requests.post(f"{BASE_URL}/api/reset")
    print_response(response)
    
    # Initialize with bearing dataset (auto-downloads)
    print("Initializing 5 peers with bearing dataset (auto-downloading)...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 5,
        "hops": [1],
        "data_distribution": "iid",
        "local_epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dataset": "bearing",  # Use bearing dataset
        "csv_path": None  # Auto-download from GitHub
    })
    print_response(response)
    
    # Start
    print("Starting system...")
    response = requests.post(f"{BASE_URL}/api/start")
    print_response(response)
    
    # Run 20 rounds
    print("Running 20 training rounds...")
    for i in range(20):
        response = requests.post(f"{BASE_URL}/api/step", json={"timeout": 30.0})
        data = response.json()["data"]
        print(f"Round {data['round']:2d}: "
              f"Train Loss={data['global_train_loss']:.4f}, "
              f"Eval Loss={data['global_eval_loss']:.4f}, "
              f"Accuracy={data['global_eval_accuracy']:.4f}")
    
    # Get final metrics
    print("\nFinal metrics:")
    response = requests.get(f"{BASE_URL}/api/metrics")
    metrics = response.json()["data"]
    print(f"Total rounds: {metrics['current_round']}")
    if metrics['global_metrics']:
        final = metrics['global_metrics'][-1]
        print(f"Final MSE: {final['global_eval_mse']:.4f}")
    
    # Stop
    print("\nStopping system...")
    response = requests.post(f"{BASE_URL}/api/stop")
    print_response(response)


def example_bearing_non_iid():
    """Example: Non-IID data distribution with bearing dataset"""
    print("\n=== Bearing Dataset - Non-IID Distribution ===\n")
    
    # Reset and initialize
    requests.post(f"{BASE_URL}/api/reset")
    
    print("Initializing 6 peers with Non-IID bearing dataset...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 6,
        "hops": [1],
        "data_distribution": "non_iid",  # Non-IID distribution
        "local_epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dataset": "bearing"
    })
    print_response(response)
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Run 30 rounds
    print("Running 30 training rounds...")
    for i in range(30):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        if (i + 1) % 5 == 0:
            print(f"Round {data['round']:2d}: "
                  f"Train Loss={data['global_train_loss']:.4f}, "
                  f"Accuracy={data['global_eval_accuracy']:.4f}")
    
    requests.post(f"{BASE_URL}/api/stop")


def example_bearing_with_local_csv():
    """Example: Using local CSV file"""
    print("\n=== Bearing Dataset - Local CSV File ===\n")
    
    requests.post(f"{BASE_URL}/api/reset")
    
    # If you have a local CSV file, specify the path
    print("Initializing with local CSV file...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 4,
        "hops": [1],
        "data_distribution": "iid",
        "dataset": "bearing",
        "csv_path": "processed/bearing_merged_2.csv"  # Use downloaded file
    })
    print_response(response)
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Run 15 rounds
    print("Running 15 training rounds...")
    for i in range(15):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        if (i + 1) % 5 == 0:
            print(f"Round {data['round']:2d}: MSE={data['global_eval_mse']:.4f}")
    
    requests.post(f"{BASE_URL}/api/stop")


def example_bearing_fedprox():
    """Example: FedProx aggregation with bearing dataset"""
    print("\n=== Bearing Dataset - FedProx Aggregation ===\n")
    
    requests.post(f"{BASE_URL}/api/reset")
    
    print("Initializing with FedProx aggregation...")
    response = requests.post(f"{BASE_URL}/api/init", json={
        "num_peers": 5,
        "hops": [1],
        "data_distribution": "non_iid",
        "local_epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dataset": "bearing",
        "aggregate_method": "prox",  # Use FedProx
        "mu": 0.01  # Proximal term
    })
    print_response(response)
    
    requests.post(f"{BASE_URL}/api/start")
    
    # Run 25 rounds
    print("Running 25 training rounds with FedProx...")
    for i in range(25):
        response = requests.post(f"{BASE_URL}/api/step")
        data = response.json()["data"]
        if (i + 1) % 5 == 0:
            print(f"Round {data['round']:2d}: "
                  f"Train Loss={data['global_train_loss']:.4f}, "
                  f"Accuracy={data['global_eval_accuracy']:.4f}")
    
    requests.post(f"{BASE_URL}/api/stop")


if __name__ == "__main__":
    print("=" * 80)
    print("DFL Tool - Bearing Dataset Examples")
    print("=" * 80)
    print("\nMake sure the API server is running on http://localhost:8000")
    print("Start it with: python api.py")
    print("\n" + "=" * 80)
    
    try:
        # Test connection
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Connected to API server\n")
        
        # Run examples
        example_bearing_basic()
        
        print("\n" + "=" * 80)
        
        example_bearing_non_iid()
        
        print("\n" + "=" * 80)
        
        example_bearing_fedprox()
        
        print("\n" + "=" * 80)
        print("All bearing dataset examples completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API server")
        print("Please start the server with: python api.py")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
