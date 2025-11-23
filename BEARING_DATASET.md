# Bearing Dataset Integration

This document describes how to use the DFL Tool with the bearing fault detection dataset.

## Dataset Format

The bearing dataset is stored in CSV format with:
- **8 feature columns**: Numerical features extracted from bearing vibration signals
- **1 label column**: Bearing fault class (last column)

Example row:
```
-0.066,-0.127,0.059,-0.125,-0.09,-0.239,-0.183,0.015,1
```

## Automatic Dataset Download

The dataset is automatically downloaded from GitHub when you initialize the system without specifying a CSV path:

```python
# Auto-download from GitHub
response = requests.post("http://localhost:8000/api/init", json={
    "num_peers": 5,
    "dataset": "bearing",
    "csv_path": None  # Auto-download
})
```

The dataset will be downloaded to `processed/bearing_merged_2.csv` or `processed/bearing_merged_1.csv`.

## Using Local CSV File

If you already have the CSV file locally:

```python
response = requests.post("http://localhost:8000/api/init", json={
    "num_peers": 5,
    "dataset": "bearing",
    "csv_path": "path/to/your/bearing_data.csv"
})
```

## Data Distribution

The bearing dataset supports all three distribution types:

### 1. IID (Independent and Identically Distributed)
Each peer gets a random subset of the data:

```python
{
    "data_distribution": "iid"
}
```

### 2. Non-IID (Dirichlet Distribution)
Data is distributed non-uniformly using Dirichlet distribution:

```python
{
    "data_distribution": "non_iid",
    # Lower alpha = more non-IID (default: 0.5)
}
```

### 3. Label Skew
Each peer gets only a subset of classes:

```python
{
    "data_distribution": "label_skew"
}
```

## Model Architecture

The model automatically adapts to the bearing dataset:
- **Input dimension**: 8 (number of features)
- **Hidden dimension**: 64 (configurable in config.py)
- **Output dimension**: Automatically detected from number of unique classes
- **Architecture**: 3-layer feedforward neural network with dropout

## Complete Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Initialize with bearing dataset
requests.post(f"{BASE_URL}/api/init", json={
    "num_peers": 5,
    "hops": [1],
    "data_distribution": "iid",
    "local_epochs": 2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "dataset": "bearing"  # Use bearing dataset
})

# 2. Start the system
requests.post(f"{BASE_URL}/api/start")

# 3. Run training rounds
for i in range(20):
    response = requests.post(f"{BASE_URL}/api/step")
    data = response.json()["data"]
    print(f"Round {i+1}: Accuracy = {data['global_eval_accuracy']:.4f}")

# 4. Get final results
metrics = requests.get(f"{BASE_URL}/api/metrics").json()
print(f"Final accuracy: {metrics['data']['global_metrics'][-1]['global_eval_accuracy']:.4f}")

# 5. Stop
requests.post(f"{BASE_URL}/api/stop")
```

## Testing

Run the bearing dataset integration tests:

```bash
python test_bearing.py
```

This will test:
- Dataset loading and auto-download
- Data distribution (IID, Non-IID, Label Skew)
- Model compatibility
- DFLPeer training and evaluation

## Running Examples

Run the bearing dataset examples:

```bash
# Make sure API server is running
python api.py

# In another terminal, run examples
python example_bearing.py
```

The example script includes:
1. Basic training with IID distribution
2. Non-IID distribution training
3. FedProx aggregation
4. Using local CSV files

## Comparison: MNIST vs Bearing Dataset

| Feature | MNIST | Bearing Dataset |
|---------|-------|-----------------|
| Type | Image classification | Tabular classification |
| Input dim | 784 (28x28 images) | 8 features |
| Output dim | 10 classes | Variable (auto-detected) |
| Format | PyTorch dataset | CSV file |
| Download | Automatic via torchvision | Automatic from GitHub |
| Preprocessing | Normalization | StandardScaler normalization |

## Configuration

Update `config.py` to change default settings:

```python
# Default dataset
DEFAULT_DATASET = "bearing"  # or "mnist"

# Model parameters (for bearing)
MODEL_INPUT_DIM = 8
MODEL_HIDDEN_DIM = 64
MODEL_OUTPUT_DIM = 4  # Auto-detected from data
```

## API Parameters

When initializing, you can specify:

```json
{
  "num_peers": 5,
  "hops": [1],
  "data_distribution": "iid",
  "local_epochs": 2,
  "learning_rate": 0.001,
  "batch_size": 64,
  "device": "cpu",
  "dataset": "bearing",
  "csv_path": null,
  "aggregate_method": "avg",
  "mu": 0.01
}
```

## Troubleshooting

### Dataset Download Fails

If automatic download fails:
1. Download manually from:
   - https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_2.csv
   - https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_1.csv
2. Save to `processed/` folder
3. Specify path in `csv_path` parameter

### CSV Format Issues

Ensure your CSV file:
- Has 8 feature columns + 1 label column (9 total)
- Uses comma separator
- Has no header row
- Labels are in the last column

### Memory Issues

If you run out of memory:
- Reduce `batch_size` (e.g., 32 or 16)
- Reduce `num_peers`
- Use `device: "cpu"` instead of "cuda"

## Performance Tips

For better results with bearing dataset:
- Use `learning_rate: 0.001` (lower than MNIST)
- Use `local_epochs: 2` or more
- Use `batch_size: 64` for faster training
- Consider FedProx for Non-IID data: `aggregate_method: "prox"`
