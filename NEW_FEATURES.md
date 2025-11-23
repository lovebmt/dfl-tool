# New Features Guide - Custom Dataset Distribution & API Endpoint

## 1. Custom Dataset Amount per Peer

### Backend Support

The system now allows you to customize the amount of data each peer receives during training.

#### API Parameter: `peer_data_fractions`

```json
POST /api/init
{
  "num_peers": 3,
  "peer_data_fractions": [0.5, 0.3, 0.2],
  ...
}
```

**Rules:**
- Array length must equal `num_peers`
- Each value is between 0.0 and 1.0
- Total sum must be ≤ 1.0
- If not provided, data is distributed equally

**Example Scenarios:**

1. **Equal Distribution (default)**
   ```json
   "num_peers": 3
   // Each peer gets 33.3% of data
   ```

2. **Unequal Distribution**
   ```json
   "num_peers": 3,
   "peer_data_fractions": [0.5, 0.3, 0.2]
   // Peer 0: 50%, Peer 1: 30%, Peer 2: 20%
   ```

3. **Sparse Distribution**
   ```json
   "num_peers": 4,
   "peer_data_fractions": [0.4, 0.3, 0.2, 0.05]
   // Peer 3 has only 5% of data (simulates weak device)
   ```

### UI Interface

In the Configuration Modal:

1. Click **"⚙️ Configure & Initialize"**
2. Set **"Number of Peers"**
3. Check **"Customize data amount per peer"**
4. Adjust data fraction sliders for each peer
5. Verify total ≤ 1.0 (shown in green)
6. Click **"Initialize"**

**UI Features:**
- ✅ Dynamic peer inputs (updates when changing peer count)
- ✅ Real-time total calculation
- ✅ Visual warning if total > 1.0 (red text)
- ✅ Default: Equal distribution
- ✅ Validation before initialization

### Use Cases

**1. Simulating Heterogeneous Devices**
```json
{
  "num_peers": 5,
  "peer_data_fractions": [0.4, 0.3, 0.15, 0.1, 0.05]
}
```
- Peer 0: High-end server (40%)
- Peer 1: Mid-range device (30%)
- Peer 2-3: Mobile devices (15%, 10%)
- Peer 4: IoT device (5%)

**2. Simulating Data Imbalance**
```json
{
  "num_peers": 3,
  "peer_data_fractions": [0.7, 0.2, 0.1]
}
```
- Peer 0 has majority of data
- Tests model performance with data skew

**3. Partial Dataset Usage**
```json
{
  "num_peers": 3,
  "peer_data_fractions": [0.2, 0.2, 0.2]
}
```
- Only use 60% of total dataset
- Faster training for testing

---

## 2. Configurable API Endpoint

### UI Endpoint Selector

The dashboard now supports multiple API endpoint configurations:

#### Options:

**1. Auto-detect (Default)**
- Automatically detects if running on localhost
- Uses `http://localhost:8000/api` for local
- Uses `http://{current-host}:8000/api` for remote

**2. Localhost**
- Forces `http://localhost:8000/api`
- Use when running locally

**3. Custom**
- Enter any custom endpoint
- Examples:
  - `http://192.168.1.7:8000`
  - `http://10.0.0.5:8000`
  - `https://dfl-server.company.com`
- Automatically adds `/api` suffix if missing

### How to Use

#### Method 1: UI Selector (Top of Dashboard)

1. Open dashboard at `http://localhost:8000`
2. Find **"API Endpoint"** dropdown in header
3. Select option:
   - **Auto-detect**: Smart detection
   - **localhost**: Force local
   - **Custom**: Enter IP/hostname
4. For custom:
   - Enter endpoint: `http://192.168.1.7:8000`
   - Click **"Apply"**
   - Endpoint is saved in localStorage

#### Method 2: URL Parameter

```
http://localhost:8000/?api=http://192.168.1.7:8000/api
```

#### Method 3: Environment Variable (Future)

Create `.env` file:
```bash
API_HOST=192.168.1.7
API_PORT=8000
```

### Remote Access Example

**Scenario:** API server running on `192.168.1.7`, access from another device

**Server Side:**
```bash
# On 192.168.1.7
cd dfl-tool
python api.py
# Server starts on 0.0.0.0:8000 (accessible from network)
```

**Client Side:**
```
# Open browser on any device in same network
http://192.168.1.7:8000

# Or use localhost with custom endpoint
http://localhost:8000
# Then select "Custom" and enter: http://192.168.1.7:8000
```

### Network Configuration

**Enable Remote Access:**

1. **Check Firewall:**
   ```bash
   # macOS - Allow port 8000
   # System Preferences > Security & Privacy > Firewall > Firewall Options
   ```

2. **Find Server IP:**
   ```bash
   # macOS/Linux
   ifconfig | grep "inet "
   
   # Look for: inet 192.168.1.x
   ```

3. **Test Connection:**
   ```bash
   # From client device
   curl http://192.168.1.7:8000/api/health
   ```

---

## Complete Example: Heterogeneous Federated Learning

### Scenario
Simulate 5 devices with different data capacities accessing from different locations:

**Server Setup (192.168.1.7):**
```bash
cd dfl-tool
python api.py
```

**Client Setup (Any device):**

1. Open: `http://192.168.1.7:8000`
2. Select Custom endpoint: `http://192.168.1.7:8000`
3. Configure system:
   ```
   Number of Peers: 5
   ✓ Customize data amount per peer
   
   Peer 0: 0.40 (Server - 40%)
   Peer 1: 0.25 (Desktop - 25%)
   Peer 2: 0.15 (Laptop - 15%)
   Peer 3: 0.10 (Tablet - 10%)
   Peer 4: 0.10 (Phone - 10%)
   
   Total: 1.00 ✓
   
   Data Distribution: Non-IID
   Local Epochs: 2
   Learning Rate: 0.001
   Batch Size: 64
   ```

4. Click **Initialize**
5. Watch training with different data amounts!

---

## API Reference

### POST /api/init

**New Parameters:**

```typescript
{
  num_peers: number,              // 2-100
  peer_data_fractions?: number[], // Optional, default: equal
  // ... other parameters
}
```

**Validation:**
- `peer_data_fractions.length === num_peers`
- `sum(peer_data_fractions) <= 1.0`
- Each value: `0.0 <= value <= 1.0`

**Response:**
```json
{
  "success": true,
  "message": "Initialized 3 peers successfully",
  "data": {
    "config": {
      "num_peers": 3,
      "peer_data_fractions": [0.5, 0.3, 0.2]
    }
  }
}
```

---

## Testing

### Test Custom Data Distribution

```bash
# Terminal 1: Start server
python api.py

# Terminal 2: Test API
curl -X POST http://localhost:8000/api/init \
  -H "Content-Type: application/json" \
  -d '{
    "num_peers": 3,
    "peer_data_fractions": [0.5, 0.3, 0.2],
    "data_distribution": "iid",
    "dataset": "bearing"
  }'

# Check peer data amounts
curl http://localhost:8000/api/peers
```

### Test Remote Access

```bash
# From remote machine
curl http://192.168.1.7:8000/api/health

# If successful, access dashboard
open http://192.168.1.7:8000
```

---

## Benefits

### Custom Data Distribution
✅ Simulate real-world device heterogeneity  
✅ Test model robustness with data imbalance  
✅ Faster experiments with partial datasets  
✅ Model edge computing scenarios  

### Configurable Endpoint
✅ Access from multiple devices  
✅ Test distributed scenarios  
✅ Demo on different networks  
✅ Flexible deployment  
✅ Easy switching between environments  

---

## Troubleshooting

**Issue: Total fractions > 1.0**
- Solution: Reduce fractions until total ≤ 1.0 (shown in green)

**Issue: Cannot connect to remote endpoint**
- Check firewall settings
- Verify server is running on 0.0.0.0
- Confirm IP address is correct
- Test with `curl http://IP:8000/api/health`

**Issue: Custom endpoint not saving**
- Check browser localStorage is enabled
- Try using URL parameter method instead

**Issue: CORS error from remote**
- API already has CORS enabled (allow_origins=["*"])
- Check browser console for specific error
