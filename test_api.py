#!/usr/bin/env python3
"""
Test script for DFL Tool API endpoints
Tests all 13 API endpoints with bearing dataset
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000/api"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("1. Testing Health Check")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    return True

def test_init():
    """Test initialization endpoint"""
    print("\n" + "="*60)
    print("2. Testing Initialization")
    print("="*60)
    config = {
        "num_peers": 3,
        "hops": [1],
        "data_distribution": "iid",
        "local_epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dataset": "bearing"
    }
    print(f"Config: {json.dumps(config, indent=2)}")
    response = requests.post(f"{BASE_URL}/init", json=config)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert result["num_peers"] == 3
    return True

def test_status():
    """Test status endpoint"""
    print("\n" + "="*60)
    print("3. Testing Status Check")
    print("="*60)
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert result["state"] == "ready"
    return True

def test_topology():
    """Test topology endpoint"""
    print("\n" + "="*60)
    print("4. Testing Topology Query")
    print("="*60)
    response = requests.get(f"{BASE_URL}/topology")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert result["topology_type"] == "ring"
    return True

def test_peers():
    """Test peers list endpoint"""
    print("\n" + "="*60)
    print("5. Testing Peers List")
    print("="*60)
    response = requests.get(f"{BASE_URL}/peers")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Number of peers: {len(result['peers'])}")
    print(f"Peer 0: {json.dumps(result['peers'][0], indent=2)}")
    assert response.status_code == 200
    assert len(result["peers"]) == 3
    return True

def test_start():
    """Test start training endpoint"""
    print("\n" + "="*60)
    print("6. Testing Start Training")
    print("="*60)
    response = requests.post(f"{BASE_URL}/start")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    return True

def test_step():
    """Test step execution"""
    print("\n" + "="*60)
    print("7. Testing Training Step")
    print("="*60)
    response = requests.post(f"{BASE_URL}/step")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert result["current_round"] == 1
    return True

def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*60)
    print("8. Testing Metrics Retrieval")
    print("="*60)
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response keys: {list(result.keys())}")
    print(f"Global metrics: {json.dumps(result['global_metrics'], indent=2)}")
    assert response.status_code == 200
    assert "global_metrics" in result
    return True

def test_peer_detail():
    """Test peer detail endpoint"""
    print("\n" + "="*60)
    print("9. Testing Peer Detail (peer_id=0)")
    print("="*60)
    response = requests.get(f"{BASE_URL}/peers/0")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    return True

def test_stop():
    """Test stop training endpoint"""
    print("\n" + "="*60)
    print("10. Testing Stop Training")
    print("="*60)
    response = requests.post(f"{BASE_URL}/stop")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    return True

def test_reset():
    """Test reset endpoint"""
    print("\n" + "="*60)
    print("11. Testing Reset")
    print("="*60)
    response = requests.post(f"{BASE_URL}/reset")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    return True

def main():
    """Run all API tests"""
    print("\n" + "="*60)
    print("DFL Tool API Tests - Bearing Dataset")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    
    try:
        # Test sequence
        test_health()
        test_init()
        test_status()
        test_topology()
        test_peers()
        test_start()
        time.sleep(1)  # Wait for training to start
        test_step()
        time.sleep(1)  # Wait for step to complete
        test_metrics()
        test_peer_detail()
        test_stop()
        test_reset()
        
        print("\n" + "="*60)
        print("✓ All API tests passed!")
        print("="*60)
        print("\nAPI is working correctly with bearing dataset.")
        print("You can now:")
        print("  1. Access API docs: http://localhost:8000/docs")
        print("  2. Run full training: python example_bearing.py")
        print("  3. Test visualization: Check metrics from /api/metrics")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API server.")
        print("Make sure the server is running: python api.py")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
