"""FastAPI REST API for DFL Tool"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import uvicorn
import os

from coordinator import Coordinator
from config import (
    API_HOST, API_PORT, DEFAULT_NUM_PEERS, DEFAULT_HOPS,
    DEFAULT_LOCAL_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE, DEFAULT_LATENCY_MS, DEFAULT_DROP_PROB,
    DISTRIBUTION_IID, AGGREGATE_AVG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DFL Tool API",
    description="Decentralized Federated Learning Simulation Tool",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global coordinator instance
coordinator = Coordinator()


# Request/Response Models
class InitRequest(BaseModel):
    num_peers: int = Field(default=DEFAULT_NUM_PEERS, ge=2, le=100)
    hops: List[int] = Field(default=[1])  # Deprecated, use topology_params
    local_epochs: int = Field(default=DEFAULT_LOCAL_EPOCHS, ge=1)
    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)
    device: str = Field(default=DEFAULT_DEVICE)
    latency_ms: float = Field(default=DEFAULT_LATENCY_MS, ge=0)
    drop_prob: float = Field(default=DEFAULT_DROP_PROB, ge=0, le=1)
    aggregate_method: str = Field(default=AGGREGATE_AVG)
    mu: float = Field(default=0.01, ge=0)
    dataset: str = Field(default="bearing")  # Currently only "bearing" supported
    csv_path: Optional[str] = Field(default=None)  # Path to CSV file (optional)
    peer_data_fractions: Optional[List[float]] = Field(default=None)  # Data fraction per peer (optional)
    topology_type: str = Field(default="ring")  # ring, line, mesh, star, full
    topology_params: Optional[Dict[str, Any]] = Field(default=None)  # Topology-specific parameters


class StartRequest(BaseModel):
    run_rounds: Optional[int] = Field(default=None, ge=1)
    continuous: bool = Field(default=False)


class StepRequest(BaseModel):
    timeout: float = Field(default=30.0, gt=0)


class ToggleNodeRequest(BaseModel):
    peer_id: int = Field(ge=0)
    enabled: bool
    fetch_model_from_neighbors: bool = Field(default=False)


class SetNeighborsRequest(BaseModel):
    peer_id: Optional[int] = Field(default=None, ge=0)
    neighbors: Optional[List[int]] = None
    hops: Optional[List[int]] = None


class SetAggregateRequest(BaseModel):
    peer_id: Optional[int] = Field(default=None, ge=0)
    aggregate_method: str = Field(default=AGGREGATE_AVG)
    mu: float = Field(default=0.01, ge=0)


class StatusResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - serve the dashboard"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(
            static_file, 
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        )
    return {
        "name": "DFL Tool API",
        "version": "1.0.0",
        "status": "running",
        "dashboard": "/static/index.html"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/api/init", response_model=StatusResponse)
async def initialize_system(request: InitRequest):
    """Initialize the DFL system with specified configuration
    
    Creates topology, peers, and message queues (threads not started yet)
    
    Supports various topologies:
    - ring: hops-based ring network
    - line: linear chain topology
    - mesh: arbitrary connections
    - star: central hub topology
    - full: fully connected network
    """
    try:
        logger.info(f"[API] Initialize request: num_peers={request.num_peers}, "
                   f"topology_type={request.topology_type}, topology_params={request.topology_params}")
        
        if coordinator.initialized:
            raise HTTPException(status_code=400, detail="System already initialized. Call /api/reset first.")
        
        coordinator.initialize(
            num_peers=request.num_peers,
            hops=request.hops,  # Backward compatibility
            local_epochs=request.local_epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            device=request.device,
            latency_ms=request.latency_ms,
            drop_prob=request.drop_prob,
            aggregate_method=request.aggregate_method,
            mu=request.mu,
            dataset=request.dataset,
            csv_path=request.csv_path,
            peer_data_fractions=request.peer_data_fractions,
            topology_type=request.topology_type,
            topology_params=request.topology_params or {}
        )
        
        logger.info(f"[API] System initialized successfully with {request.num_peers} peers "
                   f"using {request.topology_type} topology")
        return StatusResponse(
            success=True,
            message=f"Initialized {request.num_peers} peers with {request.topology_type} topology",
            data={
                "config": coordinator.config,
                "topology_info": coordinator.topology.get_topology_info()
            }
        )
    
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start", response_model=StatusResponse)
async def start_system(request: StartRequest, background_tasks: BackgroundTasks):
    """Start peer worker threads and begin training
    
    Optionally run for specified number of rounds or continuously
    """
    try:
        logger.info(f"[API] Start request: run_rounds={request.run_rounds}, continuous={request.continuous}")
        
        if not coordinator.initialized:
            raise HTTPException(status_code=400, detail="System not initialized. Call /api/init first.")
        
        if coordinator.running:
            raise HTTPException(status_code=400, detail="System already running")
        
        coordinator.start()
        logger.info(f"[API] System started successfully")
        
        # If run_rounds specified, execute them in background
        if request.run_rounds:
            async def run_rounds():
                for i in range(request.run_rounds):
                    coordinator.step()
            
            background_tasks.add_task(run_rounds)
            message = f"Started system and scheduled {request.run_rounds} rounds"
        else:
            message = "Started system (use /api/step to execute rounds)"
        
        return StatusResponse(
            success=True,
            message=message,
            data={"running": True, "current_round": coordinator.current_round}
        )
    
    except Exception as e:
        logger.error(f"Start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/step", response_model=StatusResponse)
async def execute_step(request: StepRequest):
    """Execute one training round
    
    Coordinator sends START_ROUND, waits for STATUS, and aggregates metrics
    """
    try:
        logger.info(f"[API] Step request: timeout={request.timeout}")
        
        if not coordinator.running:
            raise HTTPException(status_code=400, detail="System not running. Call /api/start first.")
        
        metrics = coordinator.step(timeout=request.timeout)
        logger.info(f"[API] Step completed: round={metrics.get('round')}")
        
        return StatusResponse(
            success=True,
            message=f"Round {metrics['round']} completed",
            data=metrics
        )
    
    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop", response_model=StatusResponse)
async def stop_system():
    """Stop all peer worker threads and reset system completely (like fresh server start)"""
    try:
        logger.info(f"[API] Stop request - stopping and resetting system")
        
        # Always reset, even if not running
        coordinator.reset()
        logger.info(f"[API] System stopped and reset successfully")
        
        return StatusResponse(
            success=True,
            message="System stopped and reset to initial state",
            data={"initialized": False, "running": False, "current_round": 0}
        )
    
    except Exception as e:
        logger.error(f"Stop failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/toggle_node", response_model=StatusResponse)
async def toggle_node(request: ToggleNodeRequest):
    """Enable or disable a specific peer node"""
    try:
        if not coordinator.initialized:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        coordinator.toggle_peer(
            peer_id=request.peer_id,
            enabled=request.enabled,
            fetch_model=request.fetch_model_from_neighbors
        )
        
        return StatusResponse(
            success=True,
            message=f"Peer {request.peer_id} {'enabled' if request.enabled else 'disabled'}",
            data={"peer_id": request.peer_id, "enabled": request.enabled}
        )
    
    except Exception as e:
        logger.error(f"Toggle node failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/set_neighbors", response_model=StatusResponse)
async def set_neighbors(request: SetNeighborsRequest):
    """Update topology neighbor configuration"""
    try:
        if not coordinator.initialized:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        coordinator.set_neighbors(
            peer_id=request.peer_id,
            neighbors=request.neighbors,
            hops=request.hops
        )
        
        return StatusResponse(
            success=True,
            message="Topology updated successfully",
            data={
                "peer_id": request.peer_id,
                "neighbors": request.neighbors,
                "hops": request.hops
            }
        )
    
    except Exception as e:
        logger.error(f"Set neighbors failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/set_aggregate", response_model=StatusResponse)
async def set_aggregate_method(request: SetAggregateRequest):
    """Set aggregation method (AVG/FedProx) for peer(s)"""
    try:
        if not coordinator.initialized:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        coordinator.set_aggregate_method(
            peer_id=request.peer_id,
            aggregate_method=request.aggregate_method,
            mu=request.mu
        )
        
        return StatusResponse(
            success=True,
            message=f"Aggregation method set to {request.aggregate_method}",
            data={
                "peer_id": request.peer_id,
                "aggregate_method": request.aggregate_method,
                "mu": request.mu
            }
        )
    
    except Exception as e:
        logger.error(f"Set aggregate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", response_model=StatusResponse)
async def get_status(peer_id: Optional[int] = None):
    """Get current system or peer status"""
    try:
        if not coordinator.initialized:
            logger.debug(f"[API] Status request - system not initialized")
            return StatusResponse(
                success=True,
                message="System not initialized",
                data={"initialized": False, "running": False}
            )
        
        status = coordinator.get_status(peer_id=peer_id)
        logger.debug(f"[API] Status: initialized={status.get('initialized')}, running={status.get('running')}, round={status.get('current_round')}")
        
        return StatusResponse(
            success=True,
            message="Status retrieved successfully",
            data=status
        )
    
    except Exception as e:
        logger.error(f"Get status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bandwidth", response_model=StatusResponse)
async def get_bandwidth(round_id: Optional[int] = None):
    """Get bandwidth statistics (per-round or cumulative)"""
    try:
        if not coordinator.initialized:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        bandwidth = coordinator.get_bandwidth(round_id=round_id)
        
        return StatusResponse(
            success=True,
            message="Bandwidth statistics retrieved",
            data=bandwidth
        )
    
    except Exception as e:
        logger.error(f"Get bandwidth failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics", response_model=StatusResponse)
async def get_metrics():
    """Get all training metrics and history"""
    try:
        if not coordinator.initialized:
            logger.debug(f"[API] Metrics request - system not initialized")
            raise HTTPException(status_code=400, detail="System not initialized")
        
        metrics = coordinator.get_metrics()
        logger.debug(f"[API] Metrics: rounds={len(metrics.get('global_loss', []))}, peers={len(metrics.get('peer_metrics', {}))}")
        
        return StatusResponse(
            success=True,
            message="Metrics retrieved successfully",
            data=metrics
        )
    
    except Exception as e:
        logger.error(f"Get metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs", response_model=StatusResponse)
async def get_logs(limit: int = 100):
    """Get recent log messages"""
    try:
        logs = coordinator.get_logs(limit=limit)
        
        return StatusResponse(
            success=True,
            message=f"Retrieved {len(logs)} log entries",
            data={"logs": logs}
        )
    
    except Exception as e:
        logger.error(f"Get logs failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset", response_model=StatusResponse)
async def reset_system():
    """Reset the entire system (stop threads, clear state)"""
    try:
        logger.info(f"[API] Reset request")
        coordinator.reset()
        logger.info(f"[API] System reset successfully")
        
        return StatusResponse(
            success=True,
            message="System reset successfully",
            data={"initialized": False, "running": False}
        )
    
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/topology", response_model=StatusResponse)
async def get_topology():
    """Get current topology information with peer metrics"""
    try:
        if not coordinator.initialized or not coordinator.topology:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        topology_info = coordinator.topology.get_topology_info()
        
        # Build topology dict (peer_id -> neighbors)
        topology = {}
        for peer_id in range(coordinator.topology.num_peers):
            topology[peer_id] = coordinator.topology.get_neighbors(peer_id)
        
        # Get peer metrics if available
        peer_metrics = {}
        if coordinator.peer_history:
            for peer_id, history in coordinator.peer_history.items():
                peer_metrics[peer_id] = {
                    'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                    'eval_loss': history['eval_loss'][-1] if history['eval_loss'] else None,
                    'eval_mse': history['eval_mse'][-1] if history['eval_mse'] else None,
                    'enabled': history['enabled'][-1] if history['enabled'] else True
                }
        
        return StatusResponse(
            success=True,
            message="Topology information retrieved",
            data={
                'topology': topology,
                'topology_info': topology_info,
                'peer_metrics': peer_metrics
            }
        )
    
    except Exception as e:
        logger.error(f"Get topology failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = API_HOST, port: int = API_PORT):
    """Run the FastAPI server"""
    logger.info(f"Starting DFL Tool API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
