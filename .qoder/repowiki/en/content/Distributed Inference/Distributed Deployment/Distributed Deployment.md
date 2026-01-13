# Distributed Deployment

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md)
- [docs/serving/distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md)
- [examples/offline_inference/data_parallel.py](file://examples/offline_inference/data_parallel.py)
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py)
- [examples/online_serving/multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py)
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh)
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py)
- [vllm/distributed/communication_op.py](file://vllm/distributed/communication_op.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document explains distributed deployment patterns in vLLM with a focus on multi-node setups, process spawning, rank assignment, environment configuration, data parallel coordination, load balancing, and fault tolerance. It covers torchrun-based launching, integration with Kubernetes and Ray, and provides step-by-step production deployment guidance for cloud platforms.

## Project Structure
The repository organizes distributed deployment materials across:
- Documentation: Online serving and deployment guides for data parallel, troubleshooting, and Kubernetes.
- Examples: Standalone scripts for offline and online data parallel deployments, including torchrun-based launching and multi-instance coordination.
- Distributed runtime: Low-level primitives for process groups, stateless rendezvous, and communication operations.

```mermaid
graph TB
subgraph "Docs"
DP["Data Parallel Deployment<br/>docs/serving/data_parallel_deployment.md"]
TR["Distributed Troubleshooting<br/>docs/serving/distributed_troubleshooting.md"]
K8S["Kubernetes Deployment<br/>docs/deployment/k8s.md"]
end
subgraph "Examples"
TORCHRUN["Torchrun DP Example<br/>examples/offline_inference/torchrun_dp_example.py"]
DP_OFFLINE["Offline DP Launcher<br/>examples/offline_inference/data_parallel.py"]
DP_ONLINE["Online Multi-Instance DP<br/>examples/online_serving/multi_instance_data_parallel.py"]
RAY["Ray Multi-Node Script<br/>examples/online_serving/multi-node-serving.sh"]
end
subgraph "Runtime"
PS["Parallel State & Groups<br/>vllm/distributed/parallel_state.py"]
UTILS["Stateless PG & Barrier<br/>vllm/distributed/utils.py"]
COMM["TP Comm Ops<br/>vllm/distributed/communication_op.py"]
end
DP --- TORCHRUN
DP --- DP_OFFLINE
DP --- DP_ONLINE
DP --- RAY
DP --- K8S
DP --- TR
TORCHRUN --> PS
DP_OFFLINE --> PS
DP_ONLINE --> PS
RAY --> PS
PS --> UTILS
PS --> COMM
```

**Diagram sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [docs/serving/distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [examples/offline_inference/data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)
- [examples/online_serving/multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py#L1-L800)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L1-L546)
- [vllm/distributed/communication_op.py](file://vllm/distributed/communication_op.py#L1-L44)

**Section sources**
- [README.md](file://README.md#L65-L92)
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)

## Core Components
- Data Parallel Deployment: Configurable via CLI flags for DP size, TP size, and hybrid/external load balancing modes. Supports MoE with attention and expert parallelism coordination.
- Torchrun-based Launching: External launcher mode for DP with torchrun, enabling custom rank-specific prompt partitioning and inter-process communication via CPU/NCCL process groups.
- Multi-Node Coordination: Ray-based cluster bootstrap and DP rank placement; environment variables for inter-node communication; DP coordinator for idle detection and pause/resume.
- Kubernetes Integration: Pod specs for CPU/GPU deployments, probes, shared memory mounts, and health checks.
- Distributed Runtime Primitives: Process groups, stateless rendezvous, barriers, and tensor-parallel communication wrappers.

**Section sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py#L1-L800)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L1-L546)
- [vllm/distributed/communication_op.py](file://vllm/distributed/communication_op.py#L1-L44)

## Architecture Overview
The distributed serving stack comprises:
- API server(s): Single or multiple endpoints depending on load balancing mode.
- DP Engines: Per-rank engine processes communicating via ZMQ and process groups.
- Coordinator: Optional process to synchronize DP ranks and manage idle/pause cycles.
- Backplane: TCP rendezvous store and process groups for CPU/NCCL comms.
- Orchestration: Ray for multi-node placement and lifecycle; Kubernetes for pod scheduling and GPU drivers.

```mermaid
graph TB
subgraph "Head Node"
API["API Server(s)"]
COORD["DP Coordinator (optional)"]
end
subgraph "DP Ranks"
R0["Rank 0 Engine"]
R1["Rank 1 Engine"]
RN["Rank N Engine"]
end
subgraph "Backplane"
STORE["TCP Store (rendezvous)"]
PG_CPU["CPU Process Group (Gloo)"]
PG_DEV["Device Process Group (NCCL)"]
end
API --> R0
API --> R1
API --> RN
API -. ZMQ .-> COORD
COORD --> R0
COORD --> R1
COORD --> RN
R0 --> PG_CPU
R0 --> PG_DEV
R1 --> PG_CPU
R1 --> PG_DEV
RN --> PG_CPU
RN --> PG_DEV
PG_CPU --> STORE
PG_DEV --> STORE
```

**Diagram sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L420)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L367-L546)

## Detailed Component Analysis

### Data Parallel Deployment Modes
- Internal Load Balancing: Single API endpoint; API server balances requests across DP ranks. Supports scaling API servers internally.
- Hybrid Load Balancing: Per-node API servers queue to local DP ranks; upstream LB distributes traffic.
- External Load Balancing: Independent endpoints per DP rank; suitable for large-scale deployments and MoE DP+EP.

```mermaid
sequenceDiagram
participant Client as "Client"
participant API as "API Server"
participant LB as "Internal/Upstream LB"
participant R0 as "DP Rank 0"
participant R1 as "DP Rank 1"
participant RN as "DP Rank N"
Client->>LB : "HTTP request"
LB->>API : "Forward to endpoint"
API->>API : "Select DP rank (queue-aware)"
API->>R0 : "ZMQ request (if selected)"
API->>R1 : "ZMQ request (if selected)"
API->>RN : "ZMQ request (if selected)"
R0-->>API : "Response"
R1-->>API : "Response"
RN-->>API : "Response"
API-->>Client : "Aggregated response"
```

**Diagram sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L23-L134)

**Section sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)

### Torchrun-based Distributed Launching
- External launcher mode: Each torchrun process spawns a single worker/engine instance; rank-specific prompt slicing ensures balanced workload.
- Inter-process communication: Use CPU process group (Gloo) for control and NCCL device group for data.
- Environment seeds: Explicit seeding ensures deterministic sampling across ranks.

```mermaid
sequenceDiagram
participant User as "User"
participant TR as "torchrun"
participant LLM as "LLM Instance"
participant PG as "Process Groups"
participant W as "Worker/Engine"
User->>TR : "Launch with --nproc-per-node"
TR->>LLM : "Initialize with distributed_executor_backend='external_launcher'"
LLM->>PG : "Create CPU (Gloo) and device (NCCL) groups"
LLM->>W : "Spawn per-rank worker"
W->>W : "Partition prompts by DP rank"
W->>PG : "Optional : sync/control via CPU group"
W-->>LLM : "Generate outputs"
LLM-->>User : "Results"
```

**Diagram sources**
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L420)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L367-L546)

**Section sources**
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)

### Multi-Node Setup with Ray
- Bootstrap: Leader starts Ray head; workers connect with retry loop; cluster waits for expected size.
- Placement: DP ranks distributed across nodes; Ray allocates based on node resources.
- Environment: Configure host IP and communication variables at cluster creation time.

```mermaid
sequenceDiagram
participant Head as "Head Node"
participant Worker as "Worker Node"
participant Ray as "Ray Runtime"
Head->>Ray : "ray start --head"
Head->>Head : "Poll nodes until expected size"
Worker->>Ray : "ray start --address=<head> : <port> --block"
Worker-->>Ray : "Connected"
Head-->>Head : "Cluster ready"
```

**Diagram sources**
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [docs/serving/distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)

**Section sources**
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [docs/serving/distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)

### Kubernetes Deployment Patterns
- CPU/GPU deployments: Pod specs with PVC/Secrets, probes, and shared memory mounts.
- GPU scheduling: Resource requests/limits for vendor-specific GPU plugins.
- Health checks: Liveness/readiness probes against the /health endpoint.

```mermaid
flowchart TD
Start(["Create PVC/Secret"]) --> Deploy["Deploy vLLM Pod(s)"]
Deploy --> Probes["Configure Liveness/Readiness"]
Probes --> Mounts["Mount Shared Memory (/dev/shm)"]
Mounts --> Expose["Expose via Service"]
Expose --> Test["curl /v1/completions"]
Test --> End(["Ready"])
```

**Diagram sources**
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)

**Section sources**
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)

### Offline Data Parallel Launcher
- Multi-process spawn per DP rank with environment variables for rank, size, and master address/port.
- Prompt partitioning per rank; graceful exit with timeouts.

```mermaid
flowchart TD
A["Parse Args"] --> B["Set VLLM_* env vars"]
B --> C["Compute local/global DP ranks"]
C --> D["Spawn Process per DP rank"]
D --> E["Partition prompts per rank"]
E --> F["Run generate()"]
F --> G["Join with timeout"]
G --> H(["Exit code"])
```

**Diagram sources**
- [examples/offline_inference/data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)

**Section sources**
- [examples/offline_inference/data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)

### Online Multi-Instance Data Parallel Client
- Demonstrates selecting a specific DP rank for generation via AsyncLLMEngine.
- Background logging and stat aggregation.

```mermaid
sequenceDiagram
participant Client as "Client"
participant Engine as "AsyncLLMEngine"
participant Rank as "DP Rank 1"
Client->>Engine : "generate(prompt, data_parallel_rank=1)"
Engine->>Rank : "Dispatch request"
Rank-->>Engine : "Stream tokens"
Engine-->>Client : "Final output"
```

**Diagram sources**
- [examples/online_serving/multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)

**Section sources**
- [examples/online_serving/multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)

### Distributed Runtime Primitives
- GroupCoordinator: Manages CPU (Gloo) and device (NCCL) groups; provides all_reduce/all_gather/reduce_scatter/broadcast/send/recv APIs.
- StatelessProcessGroup: TCP-backed rendezvous store for forming process groups without global state pollution; robust barrier implementation.
- Communication Ops: Convenience wrappers for tensor-parallel collectives.

```mermaid
classDiagram
class GroupCoordinator {
+int rank
+int world_size
+int local_rank
+int rank_in_group
+ProcessGroup cpu_group
+ProcessGroup device_group
+all_reduce(tensor) Tensor
+all_gather(tensor, dim) Tensor
+reduce_scatter(tensor, dim) Tensor
+broadcast(tensor, src) Tensor
+send_object(obj, dst) void
+recv_object(src) Any
+broadcast_tensor_dict(dict, src) dict
}
class StatelessProcessGroup {
+int rank
+int world_size
+Store store
+socket socket
+send_obj(obj, dst) void
+recv_obj(src) Any
+broadcast_obj(obj, src) Any
+all_gather_obj(obj) list
+barrier(timeout) void
+create(host, port, rank, world_size, ...) StatelessProcessGroup
}
class CommunicationOps {
+tensor_model_parallel_all_reduce(tensor) Tensor
+tensor_model_parallel_all_gather(tensor, dim) Tensor
+tensor_model_parallel_reduce_scatter(tensor, dim) Tensor
+tensor_model_parallel_gather(tensor, dst, dim) Tensor|None
+broadcast_tensor_dict(dict, src) dict
}
GroupCoordinator --> StatelessProcessGroup : "uses store"
CommunicationOps --> GroupCoordinator : "uses TP group"
```

**Diagram sources**
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L800)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L143-L546)
- [vllm/distributed/communication_op.py](file://vllm/distributed/communication_op.py#L1-L44)

**Section sources**
- [vllm/distributed/parallel_state.py](file://vllm/distributed/parallel_state.py#L1-L800)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L1-L546)
- [vllm/distributed/communication_op.py](file://vllm/distributed/communication_op.py#L1-L44)

## Dependency Analysis
- CLI-driven DP orchestration depends on environment variables and process group initialization.
- torchrun external launcher relies on stateless rendezvous and CPU/NCCL groups for control/data exchange.
- Ray multi-node setup depends on consistent host IP and environment variables across nodes.
- Kubernetes deployments rely on GPU scheduling, shared memory, and health probes.

```mermaid
graph LR
DP_CLI["DP CLI Flags"] --> ENV["Environment Variables"]
ENV --> PG_INIT["Process Group Init"]
PG_INIT --> RUNTIME["Runtime Collectives"]
TORCHRUN["torchrun"] --> EXT_LAUNCH["External Launcher Mode"]
EXT_LAUNCH --> RUNTIME
RAY["Ray Cluster"] --> PLACEMENT["Node Placement"]
PLACEMENT --> RUNTIME
K8S["Kubernetes"] --> PODS["Pod Specs"]
PODS --> RUNTIME
```

**Diagram sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L367-L546)

**Section sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)
- [vllm/distributed/utils.py](file://vllm/distributed/utils.py#L1-L546)

## Performance Considerations
- Use internal or hybrid load balancing to reduce cross-node traffic and avoid API server bottlenecks.
- Enable chunked prefill and adjust max batched tokens for GPU memory efficiency in Kubernetes.
- Prefer stateless rendezvous and robust barriers to minimize startup synchronization overhead.
- For MoE models, align attention and expert layers across DP ranks and coordinate idle/pause cycles via the DP coordinator.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Inter-node GPU communication verification and environment propagation at cluster creation.
- “No available node types can fulfill resource request”: ensure consistent host IP across nodes.
- Ray observability and KubeRay troubleshooting for multi-node GPU issues.
- Kubernetes readiness/startup probe failures: increase thresholds to accommodate cold start.

**Section sources**
- [docs/serving/distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L384-L398)

## Conclusion
vLLM supports flexible distributed deployment patterns spanning internal/external load balancing, torchrun-based launching, Ray multi-node orchestration, and Kubernetes-native deployments. The runtime provides robust primitives for process groups, rendezvous, and collectives, enabling reliable data parallel inference across heterogeneous environments.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Step-by-Step Production Deployment (Cloud Platforms)
- Prepare cluster: Ensure GPU drivers, shared memory (/dev/shm), and network connectivity across nodes.
- Choose deployment mode:
  - Internal LB: Single API endpoint with built-in queue-aware balancing.
  - Hybrid LB: Per-node API servers behind an upstream load balancer.
  - External LB: Independent endpoints per DP rank with external telemetry-driven routing.
- Launch with torchrun (external launcher) for DP control and deterministic sampling.
- Integrate with Ray for multi-node placement and lifecycle management.
- Deploy on Kubernetes with GPU scheduling, PVC/Secrets, shared memory mounts, and health probes.

**Section sources**
- [docs/serving/data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L23-L134)
- [examples/offline_inference/torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [examples/online_serving/multi-node-serving.sh](file://examples/online_serving/multi-node-serving.sh#L1-L120)
- [docs/deployment/k8s.md](file://docs/deployment/k8s.md#L1-L398)