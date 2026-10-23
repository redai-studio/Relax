# Distributed Checkpoint Service (DCS) - Architecture & Design

## Overview

The **Distributed Checkpoint Service (DCS)** is a high-performance distributed checkpoint engine designed for large-scale multi-GPU/multi-node model training. It provides:

- **Control Plane / Data Plane Separation**: Coordinator handles topology management; clients handle data transfer
- **Dynamic Role-Aware Networking**: Automatic peer discovery and topology updates
- **Dual Communication Backends**: Device-Direct (NCCL/GLOO) for intra-cluster, TCP for cross-cluster
- **Elastic Scaling & Resharding**: Support for dynamic group changes and tensor remapping
- **Production-Grade Fault Tolerance**: Heartbeat monitoring, automatic recovery, retry policies
- **Comprehensive Metrics**: Prometheus-compatible observability for latency, throughput, and errors

______________________________________________________________________

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DCS Architecture                           │
├──────────────────────┬──────────────────────────────────────────┤
│   Control Plane      │              Data Plane                  │
│                      │                                          │
│  ┌────────────────┐  │   ┌──────────────────────────────────┐   │
│  │  Coordinator   │◄─┼───┤  CheckpointEngineClient         │    │
│  │  (HTTP REST)   │  │   │                                  │   │
│  │                │  │   ├─ Role Registration              │    │
│  ├────────────────┤  │   ├─ Peer Discovery                 │    │
│  │ ┌────────────┐ │  │   ├─ Tensor Send/Recv              │     │
│  │ │ Topology   │ │  │   └─ Checkpoint Save/Load          │     │
│  │ │ Manager    │ │  │                                    │     │
│  │ └────────────┘ │  │   ┌──────────────────────────────────┐   │
│  │                │  │   │      Communication Backends      │   │
│  ├────────────────┤  │   │                                  │   │
│  │ ┌────────────┐ │  │   ├─ DeviceDirectBackend             │   │
│  │ │ Checkpoint │ │  │   │  (NCCL/GLOO)                     │   │
│  │ │ Version    │ │  │   │                                  │   │
│  │ │ Manager    │ │  │   ├─ CpuOffloadTcpBackend            │   │
│  │ └────────────┘ │  │   │  (TCP + Pinned Memory)           │   │
│  │                │  │   └──────────────────────────────────┘   │
│  └────────────────┘  │                                          │
└──────────────────────┴──────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  DCS Protocol   │
                    │ (Binary + JSON) │
                    └─────────────────┘
```

______________________________________________________________________

## Core Components

### 1. **Configuration** (`config.py`)

Defines tunable parameters for DCS deployment.

#### Key Classes:

- **`BackendType`**: Enum for communication backends (GLOO, NCCL, TCP)

- **`RoleInfo`**: Represents a node in the distributed system

  - `role_name`: Logical group (e.g., "actor", "rollout", "trainer")
  - `rank`: Process ID within the role
  - `ip`, `port`: Network address for P2P communication
  - `device_id`: GPU ID if applicable
  - `metadata`: Custom attributes (tensor parallelism, pipeline parallelism, etc.)

- **`DCSConfig`**: Main configuration class with settings for:

  - **Coordinator**: Host, port, heartbeat intervals
  - **Communication**: Backend type, TCP buffer sizes, tensor fusion threshold
  - **Storage**: Checkpoint directory, async I/O
  - **Fault Tolerance**: Max retries, retry delays
  - **Observability**: Metrics enablement, Prometheus port

- **`TopologyConfig`**: Defines role-to-role connections

  - `role_mappings`: E.g., `{"actor": "rollout"}` means actor_rank N connects to rollout_rank N

#### Example Configuration:

```python
config = DCSConfig(
    coordinator_host="0.0.0.0",
    coordinator_port=8000,
    backend_type=BackendType.NCCL,
    heartbeat_interval_seconds=5.0,
    heartbeat_timeout_seconds=30.0,
    checkpoint_dir="/checkpoints",
    tensor_fusion_threshold=1024*1024,  # 1MB
    enable_metrics=True,
)
```

______________________________________________________________________

### 2. **Protocol** (`protocol.py`)

Efficient binary protocol for tensor transmission over TCP.

#### Design:

```
Frame Format:
┌─────────────────┬─────────────┬──────────────┬──────────────┬───────────────┐
│ Magic (4B)      │ Version(4B) │ MetaLen (4B) │ Metadata(N)  │ Payload(M)    │
│ 0x44435301      │ 1           │ N            │ JSON         │ Raw Bytes     │
├─────────────────┴─────────────┴──────────────┴──────────────┴───────────────┤
│ Header (12B) │ Metadata (JSON) │ Payload (Binary)                           │
└──────────────┴─────────────────┴────────────────────────────────────────────┘
```

#### Key Classes:

- **`TensorMeta`**: Metadata for a single tensor

  - `name`: Tensor identifier
  - `dtype`: Data type string (e.g., "float32", "bfloat16")
  - `shape`: Tensor dimensions
  - `length`: Byte size in payload
  - `offset`: Byte offset in payload

- **`DCSMessage`**: In-memory message representation

  - `tensors`: Dict of tensor data
  - `metadata`: Optional extra metadata
  - `version`: Protocol version

- **`DCSProtocol`**: Static encoder/decoder

  - `encode(tensor_dict)`: Convert tensors to binary
  - `decode(data)`: Convert binary back to tensors
  - Supports torch.Tensor and numpy.ndarray
  - Handles bfloat16 conversion for numpy compatibility

- **`DCSProtocolSocket`**: Synchronous socket handler

  - `send()`: Send tensors over socket
  - `recv()`: Receive tensors from socket

- **`AsyncDCSProtocolSocket`**: Asynchronous socket handler

  - `async send()`: Async send
  - `async recv()`: Async receive

#### Advantages:

- **Compact**: Magic header + version for validation
- **Efficient**: Raw binary payload, minimal overhead
- **Flexible**: Metadata is extensible JSON
- **Streaming-friendly**: Can transmit large tensors in chunks

______________________________________________________________________

### 3. **Metrics** (`metrics.py`)

Production-grade observability with Prometheus export.

#### Metric Types:

**Histograms** (latency tracking):

- `dcs_save_latency_seconds`: Time to save checkpoint
- `dcs_load_latency_seconds`: Time to load checkpoint
- `dcs_send_latency_seconds`: Time to send tensors
- `dcs_recv_latency_seconds`: Time to receive tensors

**Counters** (monotonic):

- `dcs_bytes_sent_total`, `dcs_bytes_received_total`
- `dcs_bytes_saved_total`, `dcs_bytes_loaded_total`
- `dcs_*_operations_total`: Operation counts
- `dcs_errors_total`: Total errors

**Gauges** (point-in-time):

- `dcs_memory_buffer_usage_bytes`: Current buffer memory
- `dcs_active_connections`: Open connections
- `dcs_pending_operations`: In-flight operations

#### Key Classes:

- **`Histogram`**: Latency tracking with configurable buckets

  - `observe(value)`: Record a sample
  - `get_stats()`: Returns count, sum, avg, bucket distribution

- **`Counter`**: Monotonically increasing counter

  - `inc(value)`: Increment by value
  - Thread-safe with lock

- **`Gauge`**: Value that can go up/down

  - `set(value)`, `inc(value)`, `dec(value)`
  - Thread-safe

- **`MetricsCollector`**: Main collector

  - `record_save()`, `record_load()`, `record_send()`, `record_recv()`
  - `export_prometheus()`: Export in Prometheus text format
  - `get_all()`: Get all metrics as dict

#### Usage:

```python
metrics = MetricsCollector()
metrics.record_send(bytes_sent=1024*1024, duration=0.05)
print(metrics.export_prometheus())  # Prometheus format

# Global instance
from relax.distributed.checkpoint_service.metrics import get_metrics
metrics = get_metrics()
metrics.record_save(bytes_saved=5*1024*1024, duration=1.2)
```

______________________________________________________________________

### 4. **Communication Backends** (`backends/`)

Abstract interface with two concrete implementations.

#### Architecture:

```
CommBackend (Abstract)
├── DeviceDirectBackend (NCCL/GLOO)
│   └── For intra-cluster GPU-to-GPU or CPU communication
└── CpuOffloadTcpBackend (TCP)
    └── For cross-cluster or long-distance communication
```

#### Base Classes:

- **`CommHandle`**: Async operation handle

  - `request_id`: Unique operation ID
  - `is_complete`: Completion status
  - `result`: Operation result
  - `error`: Exception if failed
  - `wait()`: Blocking wait
  - `async wait_async()`: Async wait

- **`SendRequest`**: Point-to-point send descriptor

  - `tensor_dict`: Tensors to send
  - `dst_rank`: Destination rank
  - `async_op`: Blocking vs async flag
  - `metadata`: Extra data

- **`RecvRequest`**: Point-to-point receive descriptor

  - `src_rank`: Source rank
  - `tensor_names`: Expected tensor names
  - `async_op`: Blocking vs async flag

- **`CommBackend`** (ABC): Unified communication interface

  - `send()`: Point-to-point send
  - `recv()`: Point-to-point receive
  - `broadcast()`: One-to-all broadcast
  - `create_group()`: Create communication group
  - `register_peer()`: Register a peer node
  - `init_process_group()`: Initialize distributed communication

- **`TensorFusion`**: Optimizer for many small tensors

  - Concatenates multiple small tensors into one large buffer
  - Reduces protocol overhead
  - Configurable threshold (default 1MB)

#### 4.1 Device-Direct Backend (`device_direct.py`)

High-performance backend using PyTorch distributed (NCCL for GPU, GLOO for CPU).

**Features:**

- NCCL: GPU collective communication with optimal bandwidth
- GLOO: CPU-based fallback with async support
- CUDA stream integration for overlap with computation
- All-gather, broadcast, and point-to-point operations
- Async operations with completion handles

**Use Cases:**

- Multiple GPUs on same node (NVLink, PCIe)
- Multi-node GPU cluster with InfiniBand/Ethernet

**Limitations:**

- Requires all ranks initialized in process group
- Not ideal for cross-cluster latency

#### 4.2 CPU Offload TCP Backend (`cpu_offload.py`)

Flexible TCP-based backend for cross-cluster communication.

**Features:**

- Full async/await support (no blocking)
- CUDA pinned memory for efficient D2H transfers
- Connection pooling for reduced latency
- Tensor fusion optimization
- DCS protocol for efficient serialization

**Design Pipeline:**

```
GPU Memory
    ↓
Pinned Host Memory (async D2H)
    ↓
TCP Send Buffer
    ↓
Network
```

**Use Cases:**

- Cross-region/cross-cloud communication
- Long-distance checkpoint transfers
- CPU-only clusters

**Configuration:**

```python
backend = CpuOffloadTcpBackend(
    role_info=RoleInfo(...),
    config=DCSConfig(
        tcp_nodelay=True,
        tcp_buffer_size=65536,
        pinned_memory=True,
    )
)
```

______________________________________________________________________

### 5. **Client** (`client/engine.py`)

Data plane client for checkpoint operations.

#### Responsibilities:

- **Registration**: Register with coordinator, obtain rank
- **Peer Discovery**: Fetch topology, identify peers
- **Tensor Exchange**: Send/receive models with peers
- **Checkpoint I/O**: Save/load from storage
- **Heartbeat**: Signal health to coordinator

#### Key Classes:

- **`CheckpointEngineClient`**: Main client class
  - `coordinator_url`: Coordinator endpoint
  - `role_info`: Node metadata (role, rank, device, IP, port)
  - `backend_type`: Communication backend (NCCL/GLOO/TCP)

#### Key Methods:

- `async start()`: Initialize and register

  1. Create HTTP client
  2. Register with coordinator
  3. Initialize communication backend
  4. Start heartbeat task

- `async stop()`: Shutdown gracefully

  - Cancel heartbeat
  - Close backend
  - Close HTTP client

- `update_weights_for_actor_fwd_ref()`: Update model weights

  - Called from coordinator for actor forward reference
  - Synchronizes model updates across training and reference nodes

- `update_weights_for_rollout()`: Update rollout weights

  - Transfers trained weights to rollout workers

#### Properties:

```python
client.role          # Logical role name
client.rank          # Rank within role
client.world_size    # Total processes in role
client.node_id       # Unique identifier
client.is_registered # Registration status
client.backend       # Communication backend instance
```

#### Example Usage:

```python
import asyncio
from relax.distributed.checkpoint_service import CheckpointEngineClient, BackendType

async def main():
    client = CheckpointEngineClient(
        args=args,
        coordinator_url="http://localhost:8000",
        role="actor",
        rank=0,
        backend_type=BackendType.NCCL,
        device_id=0,
    )

    await client.start()

    # Register with coordinator
    print(f"Registered as {client.node_id}")

    # Later: update weights
    await client.update_weights_for_actor_fwd_ref(rollout_id=100)

    await client.stop()

asyncio.run(main())
```

______________________________________________________________________

### 6. **Coordinator** (`coordinator/`)

Control plane service for topology management.

#### Architecture:

```
DCSCoordinator (FastAPI + Ray Serve)
├── TopologyManager
│   ├── Node Registration
│   ├── Rank Assignment
│   ├── Peer Lookup
│   └── Heartbeat Monitoring
└── REST Endpoints
    ├── POST /register
    ├── GET /topology
    ├── GET /heartbeat
    ├── GET /get_model_update_group_ranks
    └── ...
```

#### 6.1 Coordinator Service (`service.py`)

FastAPI-based REST API for topology and checkpoint management.

**Endpoints:**

- `POST /register`: Register a new node

  - Input: `RegisterRequest` (role_name, rank, ip, port, device_id, metadata)
  - Output: `RegisterResponse` (status, rank, node_id)
  - Returns assigned global rank

- `GET /heartbeat`: Update node heartbeat

  - Parameters: `role`, `rank`
  - Output: `HeartbeatResponse` (status, timestamp)

- `GET /topology`: Get current topology

  - Parameters: `role_filter` (optional)
  - Output: `TopologyResponse` (all nodes, world_size)
  - Returns full role->rank mapping

- `GET /peer`: Get peer for a node

  - Parameters: `role`, `rank`, `peer_role`
  - Output: Peer's `RoleInfo`

- `GET /get_model_update_group_ranks`: Get communication groups for weight updates

  - Parameters: `role`, `rank`, `need_update_ref`
  - Output: `GroupRanksResponse` with group topology
  - Handles tensor parallelism resharding for model updates

- `DELETE /unregister`: Deregister a node

  - Parameters: `role`, `rank`

**API Models:**

```python
RegisterRequest:
  role_name: str
  rank: Optional[int]
  ip: Optional[str]
  port: Optional[int]
  device_id: Optional[int]
  metadata: Optional[Dict]

RegisterResponse:
  status: str
  message: str
  rank: int
  node_id: str

TopologyResponse:
  nodes: Dict[str, Dict[int, Dict]]
  world_size: int

GroupRanksResponse:
  global_rank: int
  group_ranks: List[int]
  world_size: int
  train_world_size: int
  target_world_size: int
  tp_all_gather_size: int
  train_pp_size: int
  master_address: str
  master_port: int
```

**Deployment Options:**

1. **Standalone**: Use FastAPI + uvicorn

   ```python
   coordinator = DCSCoordinator()
   app = coordinator.create_app()

   # uvicorn relax.distributed.checkpoint_service.coordinator.service:app --port 8000
   ```

2. **Ray Serve**: Integrated with Ray cluster

   ```python
   from ray import serve

   serve.start()
   serve.run(DCSCoordinator.bind())
   ```

#### 6.2 Topology Manager (`topology.py`)

In-memory topology database with lifecycle management.

**Features:**

- **Role Registration**: Assign ranks to nodes
- **Peer Lookup**: Find peer for role-role connection
- **Global Rank Mapping**: Logical to physical rank translation
- **Heartbeat Tracking**: Monitor node health
- **Dynamic Updates**: Support elastic scaling

**Key Classes:**

- **`TopologyNode`**: Node representation

  - `role_info`: Node metadata
  - `last_heartbeat`: Timestamp of last heartbeat
  - `is_alive`: Health status
  - `connections`: Set of peer node_ids

- **`TopologyManager`**: Topology database

  - `register(role_info)`: Add node and assign rank
  - `unregister(role, rank)`: Remove node
  - `heartbeat(role, rank)`: Update heartbeat
  - `get_node(role, rank)`: Get node info
  - `get_peer(role, rank, peer_role)`: Find peer
  - `get_role_nodes(role)`: Get all nodes in role
  - `get_world_size()`: Total nodes

**Example Usage:**

```python
manager = TopologyManager(
    config=TopologyConfig(role_mappings={"actor": "rollout"}),
    heartbeat_timeout=30.0
)

# Register nodes
manager.register(RoleInfo(role_name="actor", rank=0, ip="10.0.0.1", port=20000))
manager.register(RoleInfo(role_name="rollout", rank=0, ip="192.0.2.2", port=20001))

# Get peer
peer = manager.get_peer("actor", 0, "rollout")
print(f"Actor 0 should connect to Rollout 0 at {peer.address}")

# Heartbeat
manager.heartbeat("actor", 0)
```

______________________________________________________________________

## Data Flow

### Weight Update Flow

```
Actor (Training)
    ↓
    └─→ Coordinator
        ├─ Register
        ├─ Get model update groups
        └─ Get peer topology
    ↓
Actor FWD (Forward Reference)
    ↓
DeviceDirectBackend (NCCL AllGather)
    ↓
Reference (Reference Model)
    │
    └─→ Rollout (Generate Samples)
```

### Checkpoint Save Flow

```
Client
    ↓
GetMetrics (latency tracking)
    ↓
Tensor Fusion (if threshold met)
    ↓
DCSProtocol.encode()
    ↓
CommBackend.send() or StorageBackend.save()
    ↓
Metrics Collection
```

### Checkpoint Load Flow

```
Client
    ↓
Coordinator (get checkpoint version)
    ↓
StorageBackend.load() or CommBackend.recv()
    ↓
DCSProtocol.decode()
    ↓
Tensor Unfusion (if was fused)
    ↓
Metrics Collection
```

______________________________________________________________________

## Configuration Examples

### Single-Node Multi-GPU

```python
config = DCSConfig(
    backend_type=BackendType.NCCL,
    coordinator_host="127.0.0.1",
    coordinator_port=8000,
    comm_base_port=20000,
)

client = CheckpointEngineClient(
    coordinator_url="http://127.0.0.1:8000",
    role="actor",
    rank=0,
    backend_type=BackendType.NCCL,
    device_id=0,
)
```

### Multi-Node GPU Cluster

```python
config = DCSConfig(
    backend_type=BackendType.NCCL,
    coordinator_host="10.0.0.1",
    coordinator_port=8000,
    heartbeat_interval_seconds=5.0,
    heartbeat_timeout_seconds=30.0,
)

topology_config = TopologyConfig(
    role_mappings={
        "actor": "rollout",
        "rollout": "rm",
    }
)

coordinator = DCSCoordinator(config, topology_config)
```

### Cross-Region Distribution

```python
config = DCSConfig(
    backend_type=BackendType.TCP,  # TCP for long-distance
    tcp_nodelay=True,
    pinned_memory=True,
    tensor_fusion_threshold=10*1024*1024,  # 10MB for TCP
)

client = CheckpointEngineClient(
    coordinator_url="http://coordinator.example.com:8000",
    role="actor",
    rank=0,
    backend_type=BackendType.TCP,  # Override to TCP
    ip="10.0.0.1",
    port=20000,
)
```

______________________________________________________________________

## Performance Tuning

### Tensor Fusion

Fusion reduces overhead by combining small tensors:

```python
config.tensor_fusion_threshold = 1024 * 1024  # 1MB
# Only fuse if total tensors >= 1MB and count > 1
```

### Pinned Memory

Enables async GPU-to-CPU transfer:

```python
config.pinned_memory = True  # Default, recommended for GPU
```

### TCP Settings

For TCP backend:

```python
config.tcp_nodelay = True              # Disable Nagle's algorithm
config.tcp_buffer_size = 65536         # 64KB buffers
config.comm_base_port = 20000          # Base port
```

### Heartbeat

Adjust for network reliability:

```python
config.heartbeat_interval_seconds = 5.0   # Every 5 seconds
config.heartbeat_timeout_seconds = 30.0   # 30 second deadline
```

______________________________________________________________________

## Fault Tolerance

### Node Failure Detection

1. Client stops sending heartbeats
2. Coordinator marks node as dead (after heartbeat_timeout)
3. Topology is updated
4. Remaining nodes can continue with resharded topology

### Automatic Retry

```python
config.max_retries = 3
config.retry_delay_seconds = 1.0  # Exponential backoff
```

### Graceful Shutdown

```python
await client.stop()  # Clean shutdown
```

______________________________________________________________________

## Monitoring & Observability

### Metrics Export

```python
from relax.distributed.checkpoint_service.metrics import get_metrics

metrics = get_metrics()

# Get as dict
stats = metrics.get_all()
print(f"Total bytes sent: {stats['counters']['bytes_sent']}")
print(f"Avg send latency: {stats['latency']['send']['avg']:.3f}s")

# Export for Prometheus
prom_text = metrics.export_prometheus()
# Write to Prometheus endpoint
```

### Logging

All components use standard Python logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relax.distributed.checkpoint_service")

# Enable debug logs
logger.setLevel(logging.DEBUG)
```

______________________________________________________________________

## Advanced Topics

### Elastic Scaling

The system supports dynamic topology changes:

1. New node registers with coordinator
2. Coordinator assigns rank
3. Client calls `update_weights_for_actor_fwd_ref()` with new topology
4. Existing communication groups are updated

### Tensor Parallelism Resharding

The `get_model_update_group_ranks` endpoint handles:

- Training tensor parallelism (TP) size
- Forward reference TP size (different from training)
- All-gather groups for TP synchronization
- Automatic group formation based on parallelism configuration

### Custom Metadata

Store extra information on nodes:

```python
role_info = RoleInfo(
    role_name="actor",
    rank=0,
    metadata={
        "tp_size": 4,
        "pp_size": 2,
        "gpu_model": "A100",  # gitleaks:allow -- synthetic metadata example
        "memory_gb": 80,
    }
)

client = CheckpointEngineClient(
    ...,
    metadata=role_info.metadata,
)
```

______________________________________________________________________

## Troubleshooting

### Connection Issues

- Check firewall rules for P2P ports (base_port to base_port + max_ranks)
- Verify coordinator is reachable
- Check IP/port configuration in RoleInfo

### Heartbeat Timeouts

- Increase `heartbeat_timeout_seconds` for unstable networks
- Check network latency between coordinator and clients
- Monitor `dcs_errors_total` metrics

### Low Throughput

- Enable tensor fusion for many small tensors
- Use NCCL backend for same-node GPU communication
- Check network bandwidth with `dcs_bytes_sent_total / time`

### Memory Pressure

- Monitor `dcs_memory_buffer_usage_bytes` gauge
- Reduce `tcp_buffer_size` if memory constrained
- Disable `pinned_memory` on CPU-only systems

______________________________________________________________________

## Performance Characteristics

### DeviceDirectBackend (NCCL)

- **Latency**: Sub-microsecond for same-node
- **Throughput**: 1-8 TB/s depending on GPU interconnect
- **Best for**: Same-node multi-GPU or InfiniBand clusters

### DeviceDirectBackend (GLOO)

- **Latency**: 10-100 microseconds per operation
- **Throughput**: 1-10 GB/s on commodity networks
- **Best for**: CPU-based training or CPU fallback

### CpuOffloadTcpBackend

- **Latency**: 100 microseconds to milliseconds (network dependent)
- **Throughput**: Limited by network (typically 1-10 Gbps)
- **Best for**: Cross-region or WAN transfers

______________________________________________________________________

## Bibliography & References

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- Prometheus Metrics: https://prometheus.io/docs/
- DCS Protocol Design: \[Internal\]
