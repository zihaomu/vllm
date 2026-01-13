# Distributed Inference Examples

<cite>
**Referenced Files in This Document**
- [data_parallel.py](file://examples/offline_inference/data_parallel.py)
- [torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py)
- [disaggregated_prefill.py](file://examples/offline_inference/disaggregated_prefill.py)
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py)
- [run_cluster.sh](file://examples/online_serving/run_cluster.sh)
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md)
- [context_parallel_deployment.md](file://docs/serving/context_parallel_deployment.md)
- [parallel.py](file://vllm/config/parallel.py)
- [multiproc_executor.py](file://vllm/v1/executor/multiproc_executor.py)
- [parallel_state.py](file://vllm/distributed/parallel_state.py)
- [utils.py](file://vllm/v1/engine/utils.py)
- [perf.py](file://vllm/v1/metrics/perf.py)
- [benchmark_device_communicators.py](file://benchmarks/kernels/benchmark_device_communicators.py)
- [disagg_overhead_benchmark.sh](file://benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh)
- [distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md)
- [security.md](file://docs/usage/security.md)
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
This document provides practical, code-backed guidance for distributed inference using vLLM. It covers:
- Data parallelism for scaling throughput across independent batches
- Tensor and context parallelism for efficient model and KV cache distribution
- Disaggregated prefill architectures for separating prefill and decode workloads
- Multi-node deployment patterns, resource allocation strategies, and scaling considerations
- Configuration options, fault tolerance mechanisms, and performance optimization
- Practical examples for cluster setup, workload distribution, and monitoring
- Troubleshooting distributed deployments and optimizing inter-node communication

## Project Structure
The repository organizes distributed inference examples and documentation across:
- Examples for offline and online serving
- Serving documentation for deployment modes
- Engine internals for parallel configuration and execution
- Benchmarks for performance measurement and communication tuning

```mermaid
graph TB
subgraph "Examples"
DP["Offline DP<br/>examples/offline_inference/data_parallel.py"]
TR_DPE["Offline DP with torchrun<br/>examples/offline_inference/torchrun_dp_example.py"]
DISAGG["Disaggregated Prefill<br/>examples/offline_inference/disaggregated_prefill.py"]
MULTI_DP["Multi-instance DP<br/>examples/online_serving/multi_instance_data_parallel.py"]
CLUSTER["Cluster Launcher<br/>examples/online_serving/run_cluster.sh"]
end
subgraph "Docs"
DP_DEPLOY["Data Parallel Deployment<br/>docs/serving/data_parallel_deployment.md"]
CP_DEPLOY["Context Parallel Deployment<br/>docs/serving/context_parallel_deployment.md"]
DIST_TROUBLE["Distributed Troubleshooting<br/>docs/serving/distributed_troubleshooting.md"]
SEC["Security & Internode Comm<br/>docs/usage/security.md"]
end
subgraph "Engine Internals"
PARALLEL_CFG["Parallel Config<br/>vllm/config/parallel.py"]
EXECUTOR["Multiproc Executor<br/>vllm/v1/executor/multiproc_executor.py"]
PAR_STATE["Parallel State<br/>vllm/distributed/parallel_state.py"]
ENGINE_UTIL["Engine Resource Utils<br/>vllm/v1/engine/utils.py"]
PERF_METRICS["Perf Metrics<br/>vllm/v1/metrics/perf.py"]
end
subgraph "Benchmarks"
COMM_BENCH["Comm Benchmarks<br/>benchmarks/kernels/benchmark_device_communicators.py"]
DISAGG_BENCH["Disagg Overhead Benchmark<br/>benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh"]
end
DP --> DP_DEPLOY
TR_DPE --> PARALLEL_CFG
DISAGG --> PARALLEL_CFG
MULTI_DP --> EXECUTOR
CLUSTER --> DIST_TROUBLE
DP_DEPLOY --> PARALLEL_CFG
CP_DEPLOY --> PAR_STATE
PARALLEL_CFG --> EXECUTOR
EXECUTOR --> ENGINE_UTIL
PERF_METRICS --> COMM_BENCH
DISAGG_BENCH --> DISAGG
```

**Diagram sources**
- [data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)
- [torchrun_dp_example.py](file://examples/offline_inference/torchrun_dp_example.py#L1-L152)
- [disaggregated_prefill.py](file://examples/offline_inference/disaggregated_prefill.py#L1-L128)
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)
- [run_cluster.sh](file://examples/online_serving/run_cluster.sh#L1-L132)
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [context_parallel_deployment.md](file://docs/serving/context_parallel_deployment.md#L1-L48)
- [parallel.py](file://vllm/config/parallel.py#L588-L619)
- [multiproc_executor.py](file://vllm/v1/executor/multiproc_executor.py#L107-L125)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)
- [utils.py](file://vllm/v1/engine/utils.py#L438-L484)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)
- [benchmark_device_communicators.py](file://benchmarks/kernels/benchmark_device_communicators.py#L143-L341)
- [disagg_overhead_benchmark.sh](file://benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh#L93-L143)

**Section sources**
- [data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)

## Core Components
- Data Parallel Inference (offline and online)
  - Offline example demonstrates multi-node DP with explicit rank assignment and environment variables for DP coordination.
  - Online example shows multi-instance DP with explicit DP rank targeting and RPC configuration.
- Tensor and Context Parallelism
  - Context parallel deployment doc explains prefill and decode strategies and KV cache sharding trade-offs.
  - Engine parallel state initializes model-parallel groups for decode and prefill context parallel.
- Disaggregated Prefill
  - Example separates prefill and decode onto different GPUs with KV cache transfer via a connector.
  - Benchmark script measures overhead of disagg prefill implementation.
- Cluster Setup and Monitoring
  - Cluster launcher script provisions Ray head and worker nodes with host networking and GPU access.
  - Performance metrics and communication benchmarks quantify throughput and bandwidth usage.

**Section sources**
- [data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)
- [context_parallel_deployment.md](file://docs/serving/context_parallel_deployment.md#L1-L48)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)
- [disaggregated_prefill.py](file://examples/offline_inference/disaggregated_prefill.py#L1-L128)
- [disagg_overhead_benchmark.sh](file://benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh#L93-L143)
- [run_cluster.sh](file://examples/online_serving/run_cluster.sh#L1-L132)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)

## Architecture Overview
The distributed inference stack integrates:
- Command-line and programmatic configuration for DP/TP/PP/DCP
- Multiprocessing and executor backends for multi-node orchestration
- KV cache transfer connectors for disagg prefill
- Metrics and benchmarks for performance tuning

```mermaid
graph TB
subgraph "CLI/API"
VLLM_SERVE["vllm serve"]
ASYNC_ENGINE["AsyncLLMEngine"]
end
subgraph "Execution Backends"
MP["Multiprocessing Backend"]
RAY["Ray Backend"]
EXTL["External Launcher"]
end
subgraph "Parallel Groups"
TP["Tensor Parallel"]
DP["Data Parallel"]
PP["Pipeline Parallel"]
DCP["Decode Context Parallel"]
PCP["Prefill Context Parallel"]
end
subgraph "KV Transfer"
CONNECTOR["KV Connector (e.g., P2pNcclConnector)"]
end
subgraph "Monitoring"
METRICS["Perf Metrics"]
BENCH["Comm Benchmarks"]
end
VLLM_SERVE --> MP
VLLM_SERVE --> RAY
ASYNC_ENGINE --> MP
ASYNC_ENGINE --> EXTL
MP --> DP
MP --> TP
MP --> PP
DP --> DCP
DP --> PCP
TP --> CONNECTOR
DP --> CONNECTOR
PP --> CONNECTOR
METRICS --> BENCH
```

**Diagram sources**
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [context_parallel_deployment.md](file://docs/serving/context_parallel_deployment.md#L1-L48)
- [parallel.py](file://vllm/config/parallel.py#L588-L619)
- [multiproc_executor.py](file://vllm/v1/executor/multiproc_executor.py#L107-L125)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)
- [disaggregated_prefill.py](file://examples/offline_inference/disaggregated_prefill.py#L1-L128)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)
- [benchmark_device_communicators.py](file://benchmarks/kernels/benchmark_device_communicators.py#L143-L341)

## Detailed Component Analysis

### Data Parallel Inference (Offline)
This example demonstrates multi-node data parallel inference with:
- Explicit DP rank assignment and environment variables for DP master and local/global ranks
- Per-rank prompt partitioning and independent sampling parameters
- Multi-processing launch with configurable timeout and GPU visibility

```mermaid
sequenceDiagram
participant Node0 as "Node 0"
participant Node1 as "Node 1"
participant DP0 as "DP Rank 0"
participant DP1 as "DP Rank 1"
Node0->>DP0 : "Launch DP rank 0"
Node1->>DP1 : "Launch DP rank 1"
DP0->>DP0 : "Set DP env vars"
DP1->>DP1 : "Set DP env vars"
DP0->>DP0 : "Partition prompts"
DP1->>DP1 : "Partition prompts"
DP0->>DP0 : "Initialize LLM (TP-sized workers)"
DP1->>DP1 : "Initialize LLM (TP-sized workers)"
DP0->>DP0 : "Generate on subset of prompts"
DP1->>DP1 : "Generate on subset of prompts"
DP0-->>Node0 : "Results"
DP1-->>Node1 : "Results"
```

**Diagram sources**
- [data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)

**Section sources**
- [data_parallel.py](file://examples/offline_inference/data_parallel.py#L1-L269)

### Data Parallel Inference (Online Multi-Instance)
This example targets specific DP ranks on separate instances and coordinates via RPC:
- Uses AsyncLLMEngine with explicit data_parallel_rank selection
- Demonstrates headless mode and RPC address/port configuration
- Background logging via aggregated stat logger

```mermaid
sequenceDiagram
participant Client as "Client"
participant Engine as "AsyncLLMEngine"
participant Rank1 as "DP Rank 1 Instance"
Client->>Engine : "generate(prompt, data_parallel_rank=1)"
Engine->>Rank1 : "Dispatch to DP rank 1"
Rank1->>Rank1 : "Execute generate"
Rank1-->>Engine : "Stream outputs"
Engine-->>Client : "Final output"
```

**Diagram sources**
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)

**Section sources**
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)

### Tensor and Context Parallelism
Context parallel deployment doc outlines:
- Prefill strategies: partial query/full KV and partial query/partial KV with ring-style communication
- Decode strategies: KV cache sharding along heads and token dimension, reducing duplication with DCP
- Practical guidance for choosing TP and DCP sizes based on model KV heads and GPU counts

```mermaid
flowchart TD
Start(["Start"]) --> Prefill["Prefill Context Parallel"]
Prefill --> Strategy{"Strategy"}
Strategy --> |Partial Query Full KV| Gather["Gather KV across GPUs"]
Strategy --> |Partial Query Partial KV| Ring["Ring-style KV exchange"]
Gather --> Compute["Compute attention per GPU chunk"]
Ring --> Compute
Compute --> Decode["Decode Context Parallel"]
Decode --> Shard["Shard KV cache by heads or tokens"]
Shard --> ReduceDup["Reduce KV duplication with DCP"]
ReduceDup --> End(["End"])
```

**Diagram sources**
- [context_parallel_deployment.md](file://docs/serving/context_parallel_deployment.md#L1-L48)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)

**Section sources**
- [context_parallel_deployment.md](file://docs/serving/context_parallel_deployment.md#L1-L48)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)

### Disaggregated Prefill Architecture
The example separates prefill and decode across GPUs with KV cache transfer:
- Prefill node sets up KV transfer config as producer
- Decode node consumes KV cache and resumes generation
- Coordination via event synchronization and connector role/rank

```mermaid
sequenceDiagram
participant Prefill as "Prefill Node"
participant Producer as "KV Producer"
participant Consumer as "KV Consumer"
participant Decode as "Decode Node"
Prefill->>Producer : "Initialize KV producer (rank 0)"
Decode->>Consumer : "Initialize KV consumer (rank 1)"
Prefill->>Producer : "Generate prefill"
Producer-->>Consumer : "Transfer KV cache"
Consumer->>Decode : "Receive KV cache"
Decode->>Decode : "Resume decode with cached KV"
Decode-->>Consumer : "Complete decode"
Consumer-->>Producer : "Acknowledge completion"
```

**Diagram sources**
- [disaggregated_prefill.py](file://examples/offline_inference/disaggregated_prefill.py#L1-L128)

**Section sources**
- [disaggregated_prefill.py](file://examples/offline_inference/disaggregated_prefill.py#L1-L128)
- [disagg_overhead_benchmark.sh](file://benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh#L93-L143)

### Multi-Node Deployment Patterns and Resource Allocation
- Internal load balancing: single API server exposing a single endpoint; DP ranks distributed across nodes with RPC and HTTP ports configured per node.
- Hybrid load balancing: per-node API servers queuing to colocated DP ranks; upstream load balancer distributes requests.
- External load balancing: independent deployments per DP rank with separate endpoints; external router balances traffic.
- Ray backend: simplified multi-node launch with automatic placement; pack strategies for large DP sizes spanning nodes.

```mermaid
flowchart TD
A["Internal LB"] --> B["Single API server"]
A --> C["DP ranks across nodes"]
D["Hybrid LB"] --> E["Per-node API servers"]
D --> F["Upstream LB to local ranks"]
G["External LB"] --> H["Independent DP rank endpoints"]
I["Ray Backend"] --> J["Auto placement across nodes"]
J --> K["Pack strategies for large DP"]
```

**Diagram sources**
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [utils.py](file://vllm/v1/engine/utils.py#L438-L484)

**Section sources**
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [utils.py](file://vllm/v1/engine/utils.py#L438-L484)

### Configuration Options and Fault Tolerance
- Distributed executor backend selection and constraints for multi-node scenarios
- DP rank and RPC configuration for multi-node deployments
- Security considerations for inter-node communications (environment variables and isolation)
- Troubleshooting guidance for GPU communication verification and Ray observability

```mermaid
flowchart TD
Start(["Start"]) --> Backend["Select distributed executor backend"]
Backend --> Constraints{"Multi-node constraints?"}
Constraints --> |Yes| Validate["Validate backend and world_size"]
Constraints --> |No| Proceed["Proceed with configuration"]
Validate --> Proceed
Proceed --> DPConf["Configure DP ranks and RPC"]
DPConf --> Security["Apply security env vars"]
Security --> Run["Run distributed deployment"]
Run --> Monitor["Monitor via metrics and logs"]
Monitor --> Troubleshoot["Troubleshoot with docs and scripts"]
```

**Diagram sources**
- [parallel.py](file://vllm/config/parallel.py#L588-L619)
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [security.md](file://docs/usage/security.md#L1-L42)
- [distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)

**Section sources**
- [parallel.py](file://vllm/config/parallel.py#L588-L619)
- [data_parallel_deployment.md](file://docs/serving/data_parallel_deployment.md#L1-L134)
- [security.md](file://docs/usage/security.md#L1-L42)
- [distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)

## Dependency Analysis
The following diagram highlights key dependencies among components involved in distributed inference:

```mermaid
graph TB
PAR_CFG["vllm/config/parallel.py"]
EXEC["vllm/v1/executor/multiproc_executor.py"]
STATE["vllm/distributed/parallel_state.py"]
UTILS["vllm/v1/engine/utils.py"]
PERF["vllm/v1/metrics/perf.py"]
COMM_BENCH["benchmarks/kernels/benchmark_device_communicators.py"]
PAR_CFG --> EXEC
EXEC --> UTILS
STATE --> EXEC
PERF --> COMM_BENCH
```

**Diagram sources**
- [parallel.py](file://vllm/config/parallel.py#L588-L619)
- [multiproc_executor.py](file://vllm/v1/executor/multiproc_executor.py#L107-L125)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)
- [utils.py](file://vllm/v1/engine/utils.py#L438-L484)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)
- [benchmark_device_communicators.py](file://benchmarks/kernels/benchmark_device_communicators.py#L143-L341)

**Section sources**
- [parallel.py](file://vllm/config/parallel.py#L588-L619)
- [multiproc_executor.py](file://vllm/v1/executor/multiproc_executor.py#L107-L125)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1353-L1380)
- [utils.py](file://vllm/v1/engine/utils.py#L438-L484)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)
- [benchmark_device_communicators.py](file://benchmarks/kernels/benchmark_device_communicators.py#L143-L341)

## Performance Considerations
- Inter-node communication optimization
  - Use communication benchmarks to compare device communicators and select optimal backends for your hardware.
  - Tune environment variables for network interfaces and transport layers to improve bandwidth and reduce latency.
- Throughput and latency metrics
  - Track TFLOPs and GB/s metrics to assess utilization and identify bottlenecks.
  - Monitor service level objectives (SLOs) such as TTFT, TPOT, and end-to-end latency for SLA-driven tuning.
- KV cache transfer overhead
  - Measure disagg prefill overhead and tune connector settings and batching to minimize cross-node transfer costs.

**Section sources**
- [benchmark_device_communicators.py](file://benchmarks/kernels/benchmark_device_communicators.py#L143-L341)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)
- [disagg_overhead_benchmark.sh](file://benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh#L93-L143)

## Troubleshooting Guide
Common issues and remedies:
- Inter-node GPU communication verification
  - After launching Ray clusters, verify GPU-to-GPU communication; set environment variables during cluster creation to propagate to all nodes.
- Node IP selection problems
  - Ensure consistent IP addresses across vLLM and Ray; use VLLM_HOST_IP to align addresses and inspect node IPs with Ray status commands.
- Ray observability and debugging
  - Leverage Ray’s observability tools for monitoring, debugging, and multi-node GPU troubleshooting.

**Section sources**
- [distributed_troubleshooting.md](file://docs/serving/distributed_troubleshooting.md#L1-L17)
- [run_cluster.sh](file://examples/online_serving/run_cluster.sh#L1-L132)

## Conclusion
This document mapped distributed inference patterns in vLLM to concrete examples and internals:
- Data parallel scaling with internal, hybrid, and external load balancing
- Tensor and context parallel strategies for efficient prefill and decode
- Disaggregated prefill with KV cache transfer
- Multi-node deployment, resource allocation, and monitoring
- Performance tuning via metrics and communication benchmarks
- Troubleshooting with documented scripts and Ray observability

Adopt the examples and configurations that match your workload characteristics and infrastructure constraints.

## Appendices
- Practical cluster setup
  - Use the cluster launcher script to provision head and worker nodes with host networking and GPU access.
- Workload distribution
  - Choose internal, hybrid, or external load balancing based on scale and operational preferences.
- Monitoring
  - Enable aggregated logging and track TFLOPs/throughput metrics; leverage Ray observability for diagnostics.

**Section sources**
- [run_cluster.sh](file://examples/online_serving/run_cluster.sh#L1-L132)
- [multi_instance_data_parallel.py](file://examples/online_serving/multi_instance_data_parallel.py#L1-L88)
- [perf.py](file://vllm/v1/metrics/perf.py#L1177-L1213)