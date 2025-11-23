# DFL Tool - Decentralized Federated Learning Simulation Tool

A comprehensive tool for simulating and visualizing Decentralized Federated Learning (DFL) with configurable network topologies, aggregation methods, and fault tolerance features.

**✨ NEW: Bearing Fault Detection Dataset Support!** - See [QUICKSTART_BEARING.md](QUICKSTART_BEARING.md) for quick start guide.

## 🌟 Features

### Core Capabilities
- **Multiple Datasets**: Bearing fault detection (CSV) and MNIST (images)
- **Decentralized Architecture**: Ring, star, and fully-connected topologies
- **Multiple Aggregation Methods**: FedAvg and FedProx with configurable parameters
- **Fault Tolerance**: Simulate node failures, network latency, and message drops
- **Real-time Monitoring**: Track training metrics, bandwidth usage, and system status
- **RESTful API**: Complete API for controlling and monitoring the simulation
- **Thread-based Simulation**: Realistic peer-to-peer communication using message queues

### Advanced Features
- **Dynamic Topology**: Change network connections during runtime
- **Data Distribution**: IID, Non-IID, and label-skewed data distributions
- **Bandwidth Tracking**: Per-round and cumulative bandwidth statistics
- **Cold Start Support**: Rejoin nodes with model fetching from neighbors
- **Heterogeneous Aggregation**: Different peers can use different aggregation methods

## 📋 Architecture

```
+--------------------+        +--------------------+
| Coordinator Thread | <-->   | PeerWorker Threads |
+--------------------+        +--------------------+
        |                                |
        | control_q / topology           | inbox (MODEL messages)
        | status_queue                   |
        v                                v
    Web API (FastAPI)
```

### Components

1. **DFLPeer**: Core logic for local training, evaluation, and model aggregation
2. **PeerWorker**: Thread wrapper handling message processing and coordination
3. **Coordinator**: Orchestrates training rounds, collects metrics, manages bandwidth
4. **Topology**: Manages network structure (Ring, Star, Fully Connected)
5. **Messages**: Protocol for MODEL, CONTROL, and STATUS messages
6. **API**: RESTful interface for system control and monitoring

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd dfl-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server

```bash
python api.py
```

The API server will start on `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Basic Usage Example (Bearing Dataset)

```bash
# 1. Initialize the system with 5 peers using bearing dataset
curl -X POST "http://localhost:8000/api/init" \
  -H "Content-Type: application/json" \
  -d '{
    "num_peers": 5,
    "hops": [1],
    "dataset": "bearing",
    "data_distribution": "iid",
    "local_epochs": 2,
    "learning_rate": 0.001,
    "batch_size": 64
  }'

# 2. Start the system
curl -X POST "http://localhost:8000/api/start"

# 3. Execute training rounds
curl -X POST "http://localhost:8000/api/step" \
  -H "Content-Type: application/json" \
  -d '{"timeout": 30.0}'

# 4. Get status
curl "http://localhost:8000/api/status"

# 5. Get metrics
curl "http://localhost:8000/api/metrics"

# 6. Get bandwidth statistics
curl "http://localhost:8000/api/bandwidth"

# 7. Stop the system
curl -X POST "http://localhost:8000/api/stop"
```

### Quick Start with Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Initialize with bearing dataset (auto-downloads)
requests.post(f"{BASE_URL}/api/init", json={
    "num_peers": 5,
    "dataset": "bearing",
    "data_distribution": "iid",
    "learning_rate": 0.001,
    "batch_size": 64
})

# Start and run 20 rounds
requests.post(f"{BASE_URL}/api/start")
for i in range(20):
    r = requests.post(f"{BASE_URL}/api/step")
    data = r.json()["data"]
    print(f"Round {i+1}: Accuracy = {data['global_eval_accuracy']:.4f}")

# Get results
metrics = requests.get(f"{BASE_URL}/api/metrics").json()
requests.post(f"{BASE_URL}/api/stop")
```

## 📊 Datasets

### Bearing Fault Detection Dataset (Default)
- **Type**: CSV tabular data
- **Features**: 8 numerical features from vibration signals
- **Classes**: Auto-detected from data
- **Source**: Auto-downloaded from GitHub or use local CSV
- **Documentation**: See [BEARING_DATASET.md](BEARING_DATASET.md)
- **Quick Start**: See [QUICKSTART_BEARING.md](QUICKSTART_BEARING.md)

### MNIST Dataset
- **Type**: Image classification
- **Features**: 28×28 grayscale images (784 features)
- **Classes**: 10 digits (0-9)
- **Source**: Auto-downloaded via torchvision

## 📚 API Reference

### Initialization & Control

#### `POST /api/init`
Initialize the DFL system with configuration.

**Request Body:**
```json
{
  "num_peers": 5,
  "hops": [1],
  "data_distribution": "iid",
  "local_epochs": 1,
  "learning_rate": 0.01,
  "batch_size": 32,
  "device": "cpu",
  "latency_ms": 0.0,
  "drop_prob": 0.0,
  "aggregate_method": "avg",
  "mu": 0.01
}
```

**Parameters:**
- `num_peers`: Number of peers (2-100)
- `hops`: List of hop distances for ring topology (e.g., [1] for immediate neighbors, [1,2] for 1 and 2 hops)
- `dataset`: **"bearing"** or "mnist" ← Choose dataset
- `csv_path`: Path to CSV file (for bearing dataset, optional - auto-downloads if None)
- `data_distribution`: "iid", "non_iid", or "label_skew"
- `local_epochs`: Local training epochs per round
- `learning_rate`: Learning rate for SGD optimizer (use 0.001 for bearing, 0.01 for MNIST)
- `batch_size`: Batch size for training (use 64 for bearing, 32 for MNIST)
- `device`: "cpu" or "cuda"
- `latency_ms`: Simulated network latency in milliseconds
- `drop_prob`: Probability of message drop (0.0-1.0)
- `aggregate_method`: "avg" (FedAvg) or "prox" (FedProx)
- `mu`: FedProx proximal term coefficient

#### `POST /api/start`
Start all peer worker threads.

**Request Body:**
```json
{
  "run_rounds": 10,
  "continuous": false
}
```

#### `POST /api/step`
Execute one training round.

**Request Body:**
```json
{
  "timeout": 30.0
}
```

#### `POST /api/stop`
Stop all peer worker threads.

#### `POST /api/reset`
Reset the entire system (stop threads, clear all state).

### Node Management

#### `POST /api/toggle_node`
Enable or disable a specific peer.

**Request Body:**
```json
{
  "peer_id": 2,
  "enabled": false,
  "fetch_model_from_neighbors": true
}
```

#### `POST /api/set_neighbors`
Update topology configuration.

**Request Body (Global hop update):**
```json
{
  "hops": [1, 2]
}
```

**Request Body (Per-peer custom neighbors):**
```json
{
  "peer_id": 0,
  "neighbors": [1, 3, 4]
}
```

#### `POST /api/set_aggregate`
Set aggregation method for peer(s).

**Request Body (All peers):**
```json
{
  "aggregate_method": "prox",
  "mu": 0.01
}
```

**Request Body (Specific peer):**
```json
{
  "peer_id": 2,
  "aggregate_method": "prox",
  "mu": 0.05
}
```

### Monitoring

#### `GET /api/status`
Get system or peer status.

**Query Parameters:**
- `peer_id` (optional): Get status for specific peer

**Response:**
```json
{
  "success": true,
  "message": "Status retrieved successfully",
  "data": {
    "num_peers": 5,
    "current_round": 10,
    "running": true,
    "peers": [...]
  }
}
```

#### `GET /api/metrics`
Get all training metrics and history.

**Response:**
```json
{
  "success": true,
  "data": {
    "global_metrics": [...],
    "peer_metrics": {...},
    "current_round": 10
  }
}
```

#### `GET /api/bandwidth`
Get bandwidth statistics.

**Query Parameters:**
- `round_id` (optional): Get bandwidth for specific round

**Response:**
```json
{
  "success": true,
  "data": {
    "per_round": [...],
    "cumulative_matrix": [[...]]
  }
}
```

#### `GET /api/logs`
Get recent log messages.

**Query Parameters:**
- `limit` (default: 100): Maximum number of log entries

#### `GET /api/topology`
Get current topology information.

## 🧪 Example Scenarios

### Scenario 1: Basic Ring Topology

```python
import requests

base_url = "http://localhost:8000"

# Initialize 5 peers in ring topology
requests.post(f"{base_url}/api/init", json={
    "num_peers": 5,
    "hops": [1],
    "data_distribution": "iid"
})

# Start system
requests.post(f"{base_url}/api/start")

# Run 20 training rounds
for i in range(20):
    response = requests.post(f"{base_url}/api/step")
    metrics = response.json()["data"]
    print(f"Round {i+1}: Loss={metrics['global_train_loss']:.4f}, "
          f"Accuracy={metrics['global_eval_accuracy']:.4f}")

# Get final metrics
metrics = requests.get(f"{base_url}/api/metrics").json()
```

### Scenario 2: Fault Tolerance Testing

```python
# Initialize with message drop probability
requests.post(f"{base_url}/api/init", json={
    "num_peers": 10,
    "hops": [1],
    "drop_prob": 0.1,  # 10% message drop
    "latency_ms": 100  # 100ms latency
})

requests.post(f"{base_url}/api/start")

# Run 5 rounds
for i in range(5):
    requests.post(f"{base_url}/api/step")

# Disable peer 3
requests.post(f"{base_url}/api/toggle_node", json={
    "peer_id": 3,
    "enabled": False
})

# Continue training
for i in range(5):
    requests.post(f"{base_url}/api/step")

# Re-enable peer 3 with model fetching
requests.post(f"{base_url}/api/toggle_node", json={
    "peer_id": 3,
    "enabled": True,
    "fetch_model_from_neighbors": True
})
```

### Scenario 3: Heterogeneous Aggregation

```python
# Initialize system
requests.post(f"{base_url}/api/init", json={
    "num_peers": 6,
    "hops": [1]
})

# Set different aggregation methods for different peers
# Peers 0-2 use FedAvg
for peer_id in [0, 1, 2]:
    requests.post(f"{base_url}/api/set_aggregate", json={
        "peer_id": peer_id,
        "aggregate_method": "avg"
    })

# Peers 3-5 use FedProx
for peer_id in [3, 4, 5]:
    requests.post(f"{base_url}/api/set_aggregate", json={
        "peer_id": peer_id,
        "aggregate_method": "prox",
        "mu": 0.01
    })

requests.post(f"{base_url}/api/start")

# Run training
for i in range(10):
    requests.post(f"{base_url}/api/step")
```

### Scenario 4: Dynamic Topology Changes

```python
# Start with 1-hop ring
requests.post(f"{base_url}/api/init", json={
    "num_peers": 8,
    "hops": [1]
})

requests.post(f"{base_url}/api/start")

# Run 10 rounds with 1-hop
for i in range(10):
    requests.post(f"{base_url}/api/step")

# Expand to 2-hop ring
requests.post(f"{base_url}/api/set_neighbors", json={
    "hops": [1, 2]
})

# Run 10 more rounds
for i in range(10):
    requests.post(f"{base_url}/api/step")

# Set custom neighbors for peer 0
requests.post(f"{base_url}/api/set_neighbors", json={
    "peer_id": 0,
    "neighbors": [1, 3, 5, 7]
})
```

## 🏗️ Project Structure

```
dfl-tool/
├── api.py                 # FastAPI REST API
├── coordinator.py         # Coordinator orchestration
├── peer_worker.py         # PeerWorker thread implementation
├── dfl_peer.py           # DFLPeer core logic
├── topology.py           # Topology management
├── messages.py           # Message protocol
├── model.py              # Neural network model
├── data_utils.py         # Data distribution utilities (Bearing & MNIST)
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── BEARING_DATASET.md   # Bearing dataset documentation
├── QUICKSTART_BEARING.md # Bearing quick start guide
├── example_bearing.py   # Bearing dataset examples
├── test_bearing.py      # Bearing dataset tests
└── examples.py          # General examples
```

## 🔧 Configuration

Default settings can be modified in `config.py`:

```python
DEFAULT_NUM_PEERS = 5
DEFAULT_HOPS = {1}
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"
DEFAULT_LATENCY_MS = 0
DEFAULT_DROP_PROB = 0.0

MODEL_INPUT_DIM = 784  # MNIST
MODEL_HIDDEN_DIM = 128
MODEL_OUTPUT_DIM = 10

API_HOST = "0.0.0.0"
API_PORT = 8000
```

## 📊 Metrics & Monitoring

### Global Metrics
- `global_train_loss`: Weighted average training loss across active peers
- `global_eval_loss`: Weighted average evaluation loss
- `global_eval_accuracy`: Weighted average accuracy
- `num_active_peers`: Number of enabled peers
- `num_total_peers`: Total number of peers

### Per-Peer Metrics
- `train_loss`: Local training loss history
- `eval_loss`: Local evaluation loss history
- `eval_accuracy`: Local accuracy history
- `sent_bytes`: Bandwidth sent per round
- `recv_bytes`: Bandwidth received per round
- `enabled`: Enable/disable status per round

### Bandwidth Statistics
- Per-round bandwidth per peer
- Cumulative bandwidth matrix (peer-to-peer)
- Total sent/received bytes

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**: Change the port in `config.py` or run with custom port
   ```python
   from api import run_server
   run_server(host="0.0.0.0", port=8001)
   ```

3. **CUDA out of memory**: Use CPU or reduce batch size
   ```json
   {"device": "cpu", "batch_size": 16}
   ```

4. **Timeout waiting for peers**: Increase timeout value
   ```json
   {"timeout": 60.0}
   ```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.
