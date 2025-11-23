# DFL Tool - Bearing Dataset Quick Reference

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the setup
python test_bearing.py

# 3. Start API server
python api.py
```

## 📊 Using Bearing Dataset

### Python API

```python
import requests

BASE_URL = "http://localhost:8000"

# Initialize with bearing dataset (auto-downloads)
requests.post(f"{BASE_URL}/api/init", json={
    "num_peers": 5,
    "dataset": "bearing",  # ← Use bearing dataset
    "data_distribution": "iid",
    "learning_rate": 0.001,
    "batch_size": 64
})

# Start training
requests.post(f"{BASE_URL}/api/start")

# Run rounds
for i in range(20):
    r = requests.post(f"{BASE_URL}/api/step")
    data = r.json()["data"]
    print(f"Round {i+1}: Acc={data['global_eval_accuracy']:.4f}")
```

### cURL

```bash
# Initialize
curl -X POST "http://localhost:8000/api/init" \
  -H "Content-Type: application/json" \
  -d '{
    "num_peers": 5,
    "dataset": "bearing",
    "data_distribution": "iid",
    "learning_rate": 0.001,
    "batch_size": 64
  }'

# Start
curl -X POST "http://localhost:8000/api/start"

# Run round
curl -X POST "http://localhost:8000/api/step"

# Get metrics
curl "http://localhost:8000/api/metrics"
```

## 🎯 Key Differences from MNIST

| Parameter | MNIST | Bearing Dataset |
|-----------|-------|-----------------|
| `dataset` | `"mnist"` | `"bearing"` |
| `learning_rate` | `0.01` | `0.001` ← Lower! |
| `batch_size` | `32` | `64` ← Larger! |
| `local_epochs` | `1` | `2` ← More! |
| Input features | 784 (28×28 image) | 8 (tabular) |
| Output classes | 10 | Auto-detected |

## 📁 Dataset Sources

The system automatically downloads from:
1. https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_2.csv
2. https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_1.csv

Or use local CSV:
```python
{
    "dataset": "bearing",
    "csv_path": "path/to/your/data.csv"
}
```

## 🔧 Configuration Options

### Data Distribution

```python
# IID (balanced)
{"data_distribution": "iid"}

# Non-IID (imbalanced via Dirichlet)
{"data_distribution": "non_iid"}

# Label Skew (each peer gets only some classes)
{"data_distribution": "label_skew"}
```

### Aggregation Methods

```python
# FedAvg (simple averaging)
{
    "aggregate_method": "avg"
}

# FedProx (with proximal term - better for Non-IID)
{
    "aggregate_method": "prox",
    "mu": 0.01
}
```

### Topology

```python
# Ring with 1-hop neighbors
{"hops": [1]}

# Ring with 1 and 2-hop neighbors
{"hops": [1, 2]}

# Custom neighbors for specific peer
# POST /api/set_neighbors
{
    "peer_id": 0,
    "neighbors": [1, 3, 5]
}
```

## 📝 Complete Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Reset system
requests.post(f"{BASE_URL}/api/reset")

# Initialize 5 peers with Non-IID bearing data
requests.post(f"{BASE_URL}/api/init", json={
    "num_peers": 5,
    "hops": [1],
    "data_distribution": "non_iid",
    "local_epochs": 2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "device": "cpu",
    "dataset": "bearing",
    "aggregate_method": "prox",  # FedProx for Non-IID
    "mu": 0.01
})

# Start system
requests.post(f"{BASE_URL}/api/start")

# Train for 30 rounds
print("Training...")
for i in range(30):
    r = requests.post(f"{BASE_URL}/api/step", json={"timeout": 30.0})
    data = r.json()["data"]
    
    if (i + 1) % 5 == 0:
        print(f"Round {data['round']:2d}: "
              f"Loss={data['global_train_loss']:.4f}, "
              f"Accuracy={data['global_eval_accuracy']:.4f}")

# Get final metrics
metrics = requests.get(f"{BASE_URL}/api/metrics").json()
final = metrics["data"]["global_metrics"][-1]

print(f"\nFinal Results:")
print(f"  Accuracy: {final['global_eval_accuracy']:.4f}")
print(f"  Train Loss: {final['global_train_loss']:.4f}")
print(f"  Eval Loss: {final['global_eval_loss']:.4f}")

# Get bandwidth stats
bandwidth = requests.get(f"{BASE_URL}/api/bandwidth").json()
print(f"\nBandwidth:")
print(f"  Total rounds: {len(bandwidth['data']['per_round'])}")
last_round = bandwidth['data']['per_round'][-1]
print(f"  Last round sent: {last_round['total_sent']} bytes")
print(f"  Last round recv: {last_round['total_recv']} bytes")

# Stop
requests.post(f"{BASE_URL}/api/stop")
```

## 🧪 Testing

```bash
# Test dataset loading
python test_bearing.py

# Run example scenarios
python example_bearing.py

# Quick start wizard
python quickstart.py
```

## 📚 Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI server |
| `coordinator.py` | Training orchestration |
| `dfl_peer.py` | Peer training logic |
| `data_utils.py` | **Dataset loading** |
| `model.py` | Neural network |
| `config.py` | Configuration |
| `example_bearing.py` | **Bearing examples** |
| `test_bearing.py` | **Dataset tests** |
| `BEARING_DATASET.md` | **Full documentation** |

## 🎓 Recommended Settings

### For Fast Testing (5 peers, IID)
```python
{
    "num_peers": 5,
    "data_distribution": "iid",
    "local_epochs": 1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "dataset": "bearing"
}
# Run ~10 rounds
```

### For Realistic Scenario (10 peers, Non-IID)
```python
{
    "num_peers": 10,
    "data_distribution": "non_iid",
    "local_epochs": 2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "dataset": "bearing",
    "aggregate_method": "prox",
    "mu": 0.01
}
# Run ~30 rounds
```

### For Fault Tolerance Testing
```python
{
    "num_peers": 8,
    "data_distribution": "iid",
    "latency_ms": 100,      # 100ms latency
    "drop_prob": 0.1,       # 10% message drop
    "dataset": "bearing"
}
# Then disable/enable peers during training
```

## 🔍 Monitoring

### Check Status
```bash
curl "http://localhost:8000/api/status"
```

### View Logs
```bash
curl "http://localhost:8000/api/logs?limit=50"
```

### Get Metrics
```bash
curl "http://localhost:8000/api/metrics"
```

### Interactive API Docs
Open browser: `http://localhost:8000/docs`

## ⚡ Performance Tips

1. **Use batch_size=64** for bearing dataset (faster than 32)
2. **Use learning_rate=0.001** (lower than MNIST)
3. **Use local_epochs=2** (better convergence)
4. **Use FedProx for Non-IID** (`aggregate_method="prox"`)
5. **Use CPU** unless you have GPU (dataset is small)

## 🐛 Troubleshooting

**Dataset won't download?**
```bash
# Download manually and place in processed/
mkdir -p processed
cd processed
wget https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_2.csv
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**Server won't start?**
```bash
# Check if port 8000 is available
lsof -i :8000
# Or use different port
python -c "from api import run_server; run_server(port=8001)"
```

## 📖 Full Documentation

- **Main README**: `README.md`
- **Bearing Dataset Guide**: `BEARING_DATASET.md`
- **API Documentation**: `http://localhost:8000/docs` (when server is running)
