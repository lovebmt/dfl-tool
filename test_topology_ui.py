"""Test script to verify topology UI and API updates"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_init_with_topologies():
    """Test initialization with different topologies"""
    
    topologies = [
        {
            'name': 'Ring 1-hop',
            'config': {
                'num_peers': 5,
                'topology_type': 'ring',
                'topology_params': {'hops': [1]},
                'data_distribution': 'iid',
                'local_epochs': 1,
                'dataset': 'bearing'
            }
        },
        {
            'name': 'Line Bidirectional',
            'config': {
                'num_peers': 5,
                'topology_type': 'line',
                'topology_params': {'bidirectional': True},
                'data_distribution': 'iid',
                'local_epochs': 1,
                'dataset': 'bearing'
            }
        },
        {
            'name': 'Mesh 40%',
            'config': {
                'num_peers': 5,
                'topology_type': 'mesh',
                'topology_params': {'connectivity': 0.4},
                'data_distribution': 'iid',
                'local_epochs': 1,
                'dataset': 'bearing'
            }
        },
        {
            'name': 'Fully Connected',
            'config': {
                'num_peers': 4,
                'topology_type': 'full',
                'topology_params': {},
                'data_distribution': 'iid',
                'local_epochs': 1,
                'dataset': 'bearing'
            }
        }
    ]
    
    for topo in topologies:
        print(f"\n{'='*60}")
        print(f"Testing: {topo['name']}")
        print(f"{'='*60}")
        
        # Reset first
        try:
            requests.post(f"{BASE_URL}/api/reset")
            time.sleep(1)
        except:
            pass
        
        # Initialize
        response = requests.post(f"{BASE_URL}/api/init", json=topo['config'])
        print(f"Init Response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data['message']}")
            if 'data' in data and 'topology_info' in data['data']:
                print(f"Topology Info: {json.dumps(data['data']['topology_info'], indent=2)}")
            
            # Get topology details
            topo_response = requests.get(f"{BASE_URL}/api/topology")
            if topo_response.status_code == 200:
                topo_data = topo_response.json()
                if 'data' in topo_data:
                    print(f"\nTopology Map:")
                    topology_map = topo_data['data'].get('topology', {})
                    for peer_id, neighbors in topology_map.items():
                        print(f"  Peer {peer_id} -> {neighbors}")
        else:
            print(f"Error: {response.text}")
        
        time.sleep(1)

def test_topology_endpoint():
    """Test the topology endpoint returns peer metrics"""
    
    print(f"\n{'='*60}")
    print("Testing Topology Endpoint with Metrics")
    print(f"{'='*60}")
    
    # Initialize a simple system
    config = {
        'num_peers': 3,
        'topology_type': 'ring',
        'topology_params': {'hops': [1]},
        'data_distribution': 'iid',
        'local_epochs': 1,
        'dataset': 'bearing'
    }
    
    try:
        requests.post(f"{BASE_URL}/api/reset")
        time.sleep(1)
    except:
        pass
    
    response = requests.post(f"{BASE_URL}/api/init", json=config)
    print(f"Init: {response.status_code}")
    
    if response.status_code == 200:
        # Get topology
        topo_response = requests.get(f"{BASE_URL}/api/topology")
        if topo_response.status_code == 200:
            data = topo_response.json()
            print(f"\nTopology Data Structure:")
            print(json.dumps(data['data'], indent=2))
        else:
            print(f"Error getting topology: {topo_response.text}")

if __name__ == "__main__":
    print("="*60)
    print("TOPOLOGY UI/API VERIFICATION TEST")
    print("="*60)
    print("\nMake sure the API server is running: python api.py")
    print("Then open http://localhost:8000/static/index.html in your browser")
    
    input("\nPress Enter to start tests...")
    
    test_init_with_topologies()
    test_topology_endpoint()
    
    print(f"\n{'='*60}")
    print("TESTS COMPLETED!")
    print("="*60)
    print("\nNow test in UI:")
    print("1. Open http://localhost:8000/static/index.html")
    print("2. Click 'Configure' button")
    print("3. Try different topology types from dropdown")
    print("4. Initialize and click 'View Topology'")
    print("5. You should see a visual topology map with peer metrics")
