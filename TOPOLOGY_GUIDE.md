# Network Topologies Guide

This guide explains the different network topologies supported by the DFL Tool and how to use them.

## Supported Topologies

### 1. Ring Topology

Peers are connected in a ring structure with configurable hop distances.

**Characteristics:**
- Each peer connects to neighbors at specified hop distances
- Supports multi-hop connections (e.g., 1-hop and 2-hop neighbors)
- Circular structure: messages can flow in both directions

**Parameters:**
- `hops`: List of hop distances (e.g., `[1]` for immediate neighbors, `[1, 2]` for 1 and 2-hop neighbors)

**Example:**
```python
coordinator.initialize(
    num_peers=6,
    topology_type="ring",
    topology_params={'hops': [1, 2]}  # 1-hop and 2-hop neighbors
)
```

**Use Cases:**
- Distributed systems with limited bandwidth
- Peer-to-peer networks
- Applications requiring balanced communication patterns

---

### 2. Line Topology

Peers are connected in a linear chain without wraparound.

**Characteristics:**
- Linear chain: Peer 0 -- Peer 1 -- Peer 2 -- ... -- Peer N-1
- Edge peers (first and last) have only one neighbor
- No circular connections (unlike ring)
- Can be bidirectional (default) or unidirectional

**Parameters:**
- `bidirectional`: Boolean indicating if edges go both ways (default: `True`)

**Example:**
```python
coordinator.initialize(
    num_peers=5,
    topology_type="line",
    topology_params={'bidirectional': True}
)
```

**Use Cases:**
- Sequential processing pipelines
- Hierarchical sensor networks
- Chain-of-trust scenarios

---

### 3. Mesh Topology

Arbitrary connections between peers with custom or random connectivity.

**Characteristics:**
- Flexible connection patterns
- Can be fully or partially connected
- Supports custom edge definitions or random connectivity

**Parameters:**
- `connectivity`: Float (0-1) for random mesh generation
- `edges`: List of (source, dest) tuples for explicit edge definition

**Example (Custom Edges):**
```python
custom_edges = [
    (0, 1), (0, 2),
    (1, 2), (1, 3),
    (2, 3), (2, 4),
    (3, 4)
]

coordinator.initialize(
    num_peers=5,
    topology_type="mesh",
    topology_params={'edges': custom_edges}
)
```

**Example (Random Connectivity):**
```python
coordinator.initialize(
    num_peers=8,
    topology_type="mesh",
    topology_params={'connectivity': 0.5}  # 50% connection probability
)
```

**Use Cases:**
- Complex network simulations
- Ad-hoc networks
- Custom communication patterns

---

### 4. Fully Connected Topology

Every peer is connected to all others (complete graph).

**Characteristics:**
- Maximum connectivity
- Every peer can directly communicate with any other peer
- Highest bandwidth requirements
- Fastest convergence (typically)

**Parameters:**
- None required

**Example:**
```python
coordinator.initialize(
    num_peers=5,
    topology_type="full"
)
```

**Use Cases:**
- Small-scale networks (< 10 peers)
- Scenarios requiring maximum communication
- Baseline comparisons

---

## API Usage

### Initialize with Topology

```python
from coordinator import Coordinator

coordinator = Coordinator()
coordinator.initialize(
    num_peers=6,
    topology_type="ring",  # or "line", "mesh", "full"
    topology_params={'hops': [1]},
    data_distribution="iid",
    local_epochs=2,
    dataset="bearing"
)
```

### REST API

```bash
curl -X POST "http://localhost:8000/api/init" \
  -H "Content-Type: application/json" \
  -d '{
    "num_peers": 6,
    "topology_type": "ring",
    "topology_params": {"hops": [1, 2]},
    "data_distribution": "iid",
    "local_epochs": 2
  }'
```

### Get Topology Information

```python
# Get topology details
info = coordinator.topology.get_topology_info()
print(info)

# Get neighbors for a specific peer
neighbors = coordinator.topology.get_neighbors(peer_id=0)
print(f"Peer 0 neighbors: {neighbors}")

# Get all edges
edges = coordinator.topology.get_all_edges()
print(f"Total edges: {len(edges)}")
```

---

## Topology Comparison

| Topology | Edges | Connectivity | Bandwidth | Convergence | Use Case |
|----------|-------|--------------|-----------|-------------|----------|
| **Ring (1-hop)** | 2N | Low | Low | Slower | P2P networks |
| **Ring (2-hop)** | 4N | Medium | Medium | Medium | Balanced systems |
| **Line** | 2(N-1) | Low | Low | Slowest | Sequential processing |
| **Mesh** | Variable | Variable | Variable | Variable | Custom patterns |
| **Fully Connected** | N(N-1) | Maximum | High | Fastest | Small networks |

*N = number of peers*

---

## Dynamic Topology Updates

You can update topology during runtime (use with caution):

### Ring Topology
```python
# Change hops
coordinator.topology.set_neighbors(hops={1, 2, 3})

# Set custom neighbors for a peer
coordinator.topology.set_neighbors(peer_id=0, neighbors=[1, 3, 5])
```

### Line Topology
```python
# Toggle bidirectional
coordinator.topology.set_neighbors(bidirectional=False)
```

### Mesh Topology
```python
# Add an edge
coordinator.topology.add_edge(src=0, dst=5)

# Remove an edge
coordinator.topology.remove_edge(src=0, dst=5)
```

---

## Performance Considerations

1. **Ring Topology**
   - Lower bandwidth usage
   - Slower convergence (multi-hop propagation)
   - Good for resource-constrained environments

2. **Line Topology**
   - Slowest convergence (end-to-end propagation)
   - Simplest structure
   - Suitable for sequential workflows

3. **Mesh Topology**
   - Balance between connectivity and bandwidth
   - Flexible and fault-tolerant
   - Complexity depends on connectivity

4. **Fully Connected**
   - Fastest convergence
   - Highest bandwidth requirements
   - Best for small networks (< 10 peers)

---

## Examples

See `example_topologies.py` for complete working examples of all topologies.

Run examples:
```bash
python example_topologies.py
```

---

## Backward Compatibility

The old `hops` parameter is still supported for ring topology:

```python
# Old way (still works)
coordinator.initialize(
    num_peers=6,
    hops=[1, 2]
)

# New way (recommended)
coordinator.initialize(
    num_peers=6,
    topology_type="ring",
    topology_params={'hops': [1, 2]}
)
```

---

## Troubleshooting

**Issue:** Slow convergence
- **Solution:** Try a more connected topology (star or fully connected)

**Issue:** High bandwidth usage
- **Solution:** Use ring or line topology with lower connectivity

**Issue:** Disconnected network
- **Solution:** Check topology edges, ensure all peers are reachable

---

## Note on Mesh vs Fully Connected

**Mesh Topology** is a general-purpose topology that can represent any connection pattern:
- Use `connectivity=1.0` for a fully connected mesh (same as FullyConnectedTopology)
- Use custom `edges` for specific patterns
- Use `connectivity < 1.0` for partial mesh networks

**Fully Connected Topology** is a specialized, optimized version for complete graphs where every peer connects to all others. It's simpler and more efficient than using Mesh with connectivity=1.0.
