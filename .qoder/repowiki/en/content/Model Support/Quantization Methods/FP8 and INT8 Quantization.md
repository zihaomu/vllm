# FP8 and INT8 Quantization

<cite>
**Referenced Files in This Document**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu)
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu)
- [bench_fp8_gemm.py](file://benchmarks/kernels/bench_fp8_gemm.py)
- [quark.py](file://vllm/model_executor/layers/quantization/quark/quark.py)
- [fused_batched_moe.py](file://vllm/model_executor/layers/fused_moe/fused_batched_moe.py)
- [test_attention.py](file://tests/kernels/attention/test_attention.py)
- [quant_utils.py](file://tests/kernels/quant_utils.py)
- [CMakeLists.txt](file://CMakeLists.txt)
- [utils.cmake](file://cmake/utils.cmake)
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
This document explains FP8 and INT8 quantization in vLLM, focusing on:
- FP8 quantization: dynamic and static per-tensor and per-token schemes, W8A8 support, and hardware-specific optimizations for Ada/Hopper architectures.
- INT8 quantization: W8A8 block-wise compression, per-channel scaling, and performance benefits.
- Implementation of quantized GEMM operations, attention kernels, and mixture-of-expert (MoE) layers.
- Configuration examples, performance benchmarks, accuracy trade-offs, hardware compatibility, memory bandwidth improvements, and integration with PagedAttention.

## Project Structure
The quantization implementation spans Python layers and CUDA/Triton kernels:
- Python input quantization and utilities for FP8/INT8 group shapes and scales.
- CUDA kernels for FP8 and INT8 quantization and scaled matmul.
- Triton kernels for block-wise INT8 GEMM and per-token group quantization.
- Benchmarks comparing FP8 vs full-precision GEMM.
- Quark quantization scheme selection and capability checks.
- MoE and attention integration points.

```mermaid
graph TB
subgraph "Python Layers"
IQF["input_quant_fp8.py<br/>QuantFP8"]
QU["quant_utils.py<br/>GroupShape, QuantKey"]
IU["int8_utils.py<br/>Per-token/group INT8, Block INT8 GEMM"]
QC["quark.py<br/>Scheme detection & selection"]
end
subgraph "CUDA Kernels"
CFP8["common.cu<br/>FP8 per-tensor/token dyn/static"]
CI8["scaled_quant.cu<br/>INT8 per-tensor/token dyn/static"]
end
subgraph "Triton Kernels"
TI8["int8_utils.py<br/>_w8a8_block_int8_matmul"]
TPG["int8_utils.py<br/>per_token_group_quant_int8"]
end
subgraph "Integration"
MOE["fused_batched_moe.py<br/>MoE block-wise scales"]
ATT["test_attention.py<br/>PagedAttention integration"]
end
IQF --> CFP8
IQF --> TPG
QU --> IQF
IU --> CI8
IU --> TI8
QC --> IQF
QC --> IU
MOE --> IU
ATT --> CFP8
```

**Diagram sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L1-L203)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L1-L248)
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py#L1-L132)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L1-L475)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)
- [fused_batched_moe.py](file://vllm/model_executor/layers/fused_moe/fused_batched_moe.py#L192-L233)
- [test_attention.py](file://tests/kernels/attention/test_attention.py#L245-L280)

**Section sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L1-L203)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L1-L248)
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py#L1-L132)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L1-L475)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)
- [fused_batched_moe.py](file://vllm/model_executor/layers/fused_moe/fused_batched_moe.py#L192-L233)
- [test_attention.py](file://tests/kernels/attention/test_attention.py#L245-L280)

## Core Components
- FP8 input quantization:
  - Supports static/dynamic, per-tensor/per-token/group quantization.
  - Uses native PyTorch fallback and CUDA kernels via a custom op.
  - Includes ROCm aiter FP8 quantization path.
- INT8 utilities:
  - Per-token/group quantization for activations.
  - Block-wise INT8 GEMM with configurable block shapes and device-specific tuning.
  - Scaled INT8 quantization kernels for per-tensor and dynamic per-token modes.
- Quantization keys and group shapes:
  - GroupShape.PER_TENSOR, GroupShape.PER_TOKEN, and arbitrary block shapes.
  - QuantKey captures dtype, scale descriptors, and symmetry.
- Quark scheme detection:
  - Validates and selects FP8 W8A8, static tensor W8A8, and OCP MX formats.
  - Enforces GPU capability thresholds for FP8 support.

**Section sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L1-L203)
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py#L1-L132)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L1-L475)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)
- [quark.py](file://vllm/model_executor/layers/quantization/quark/quark.py#L221-L322)

## Architecture Overview
FP8 and INT8 quantization integrates through:
- Python quantization layers invoking CUDA/Triton kernels.
- Custom ops for scaled FP8 quantization and per-token/group INT8 quantization.
- MoE and attention modules consuming quantized inputs and scales.
- Device capability checks and architecture-specific kernel builds.

```mermaid
sequenceDiagram
participant User as "User Code"
participant Quant as "QuantFP8 (Python)"
participant Ops as "Custom Ops (CUDA/Triton)"
participant Kern as "CUDA/Triton Kernels"
User->>Quant : "forward(x, scale, scale_ub)"
alt "ROCm + aiter FP8"
Quant->>Ops : "per_tensor_quant / per_token_quant"
Ops->>Kern : "HIP kernels"
else "CUDA"
Quant->>Ops : "scaled_fp8_quant(...)"
Ops->>Kern : "FP8 quant kernels"
end
Kern-->>Quant : "quantized x, scales"
Quant-->>User : "returns (x_q, scales)"
```

**Diagram sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L67-L125)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L136-L248)

## Detailed Component Analysis

### FP8 Quantization: Dynamic and Static Schemes
- Per-tensor static:
  - Uses a single scale tensor; invoked via a custom op with a constant scale.
- Per-token dynamic:
  - Computes token-wise absmax and clamps to a minimum scaling factor; returns per-token scales.
- Group quantization:
  - Divides the hidden dimension into groups and computes per-group scales; supports column-major scale layouts.
- ROCm aiter path:
  - Uses ROCm aiter ops for per-tensor and per-token FP8 quant when conditions are met.

```mermaid
flowchart TD
Start(["FP8 Forward Entry"]) --> CheckGroup["Is group quantization?"]
CheckGroup --> |Yes| GroupPath["Compute per-group absmax<br/>Clamp and normalize<br/>Return scales"]
CheckGroup --> |No| CheckStatic["Static or Dynamic?"]
CheckStatic --> |Static| UseConstScale["Use provided scale"]
CheckStatic --> |Dynamic| ComputeScale["Compute per-token absmax<br/>Clamp to min scaling factor"]
UseConstScale --> Quantize["Scale + clamp + cast to FP8"]
ComputeScale --> Quantize
GroupPath --> Quantize
Quantize --> End(["Return (x_q, scales)"])
```

**Diagram sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L67-L167)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L90-L132)

**Section sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L1-L203)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L1-L248)
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py#L1-L132)

### INT8 Quantization: W8A8 Block-wise Compression
- Per-token/group quantization:
  - Per-token absmax-based scaling; optional epsilon to avoid division by zero.
  - CUDA path uses a custom op; Triton path provides portable kernel.
- Block-wise INT8 GEMM:
  - Supports arbitrary block shapes [N, K] with per-block weight scales and per-token activation scales.
  - Kernel grid selection uses device-specific configurations; falls back to defaults if missing.
- Scaled INT8 quantization:
  - Per-tensor static and dynamic per-token modes with saturation and rounding tailored to device.

```mermaid
flowchart TD
AStart(["INT8 Block-wise Matmul"]) --> QAct["Quantize activations<br/>per-token/group"]
QAct --> QW["Load weight + per-block scales"]
QW --> Config["Lookup device config<br/>or use defaults"]
Config --> Launch["_w8a8_block_int8_matmul launch"]
Launch --> Dot["Dot-product + scale accumulation"]
Dot --> Store["Store C with output dtype"]
Store --> AEnd(["Return C"])
```

**Diagram sources**
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L386-L475)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)

**Section sources**
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L1-L475)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)

### Quantized GEMM Operations
- FP8 GEMM:
  - Benchmarks compare BF16 baseline against FP8 variants with tensor/channel weight schemes and token/tensor activation schemes.
  - Demonstrates enabling per-token activation quantization and static activation scales.
- INT8 GEMM:
  - Block-wise GEMM with configurable block sizes and device-specific tuning.

```mermaid
sequenceDiagram
participant Bench as "bench_fp8_gemm.py"
participant Op as "scaled_fp8_quant"
participant MM as "cutlass_scaled_mm"
Bench->>Op : "Quantize A/B with selected scheme"
Op-->>Bench : "Return (Aq, Bq) and scales"
Bench->>MM : "Call scaled_mm(Aq, Bq, scale_A, scale_B, dtype)"
MM-->>Bench : "GEMM result"
```

**Diagram sources**
- [bench_fp8_gemm.py](file://benchmarks/kernels/bench_fp8_gemm.py#L1-L160)

**Section sources**
- [bench_fp8_gemm.py](file://benchmarks/kernels/bench_fp8_gemm.py#L1-L160)

### Attention Kernels and PagedAttention Integration
- PagedAttention v2:
  - Accepts KV cache scales and supports partitioned computation for long sequences.
  - Integrates with FP8 activation/weight quantization via the underlying scaled matmul path.
- FP8 attention:
  - FP8 quantization kernels compute per-token scales and clamp to safe ranges.

```mermaid
sequenceDiagram
participant Attn as "PagedAttention v2"
participant Kern as "FP8 scaled matmul kernels"
participant Cache as "KV Cache (possibly FP8)"
Attn->>Kern : "Compute attention scores with scales"
Kern->>Cache : "Read K/V with per-tensor/channel scales"
Cache-->>Kern : "Scaled KV tiles"
Kern-->>Attn : "Context + reductions"
Attn-->>Caller : "Output with logits and partition stats"
```

**Diagram sources**
- [test_attention.py](file://tests/kernels/attention/test_attention.py#L245-L280)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L136-L248)

**Section sources**
- [test_attention.py](file://tests/kernels/attention/test_attention.py#L245-L280)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L1-L248)

### MoE Layers with Block-wise Quantization
- Fused batched MoE:
  - Consumes per-activation token quantization and per-weight block scales.
  - Uses block sizes for group-wise matmul and stores results masked by expert selection.

```mermaid
sequenceDiagram
participant MoE as "FusedBatchedMoE"
participant Kern as "Block INT8 GEMM"
participant Act as "Activation quantization"
MoE->>Act : "Quantize intermediate per-token/group"
Act-->>MoE : "Quantized activation + scales"
MoE->>Kern : "Run block-wise matmul with per-block weight scales"
Kern-->>MoE : "Expert outputs"
MoE-->>Caller : "Final fused MoE output"
```

**Diagram sources**
- [fused_batched_moe.py](file://vllm/model_executor/layers/fused_moe/fused_batched_moe.py#L192-L233)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L386-L475)

**Section sources**
- [fused_batched_moe.py](file://vllm/model_executor/layers/fused_moe/fused_batched_moe.py#L192-L233)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L386-L475)

### Quark Scheme Detection and Hardware Compatibility
- FP8 W8A8:
  - Requires static weights with per-tensor or per-channel schemes and per-tensor activation for dynamic input.
- Static tensor W8A8:
  - Requires static int8 weights (per-tensor or per-channel) and per-tensor int8 activation; symmetric weight quantization.
- OCP MX:
  - Per-group quantization with group size 32 and e8m0 scale format for FP4/F6 dtypes.
- Capability checks:
  - Minimum GPU compute capability enforced for FP8 support.

```mermaid
flowchart TD
SStart(["Detect Quark Scheme"]) --> CheckW["Check weight dtype/scheme"]
SStart --> CheckA["Check activation dtype/scheme"]
CheckW --> FP8W8A8{"FP8 W8A8?"}
CheckW --> StaticW8A8{"Static Tensor W8A8?"}
CheckW --> OCPMX{"OCP MX?"}
FP8W8A8 --> |Yes| CapCheck["Check GPU capability >= min"]
StaticW8A8 --> |Yes| CapCheck
OCPMX --> |Yes| CapCheck
CapCheck --> SEnd(["Scheme selected"])
```

**Diagram sources**
- [quark.py](file://vllm/model_executor/layers/quantization/quark/quark.py#L221-L322)

**Section sources**
- [quark.py](file://vllm/model_executor/layers/quantization/quark/quark.py#L221-L322)

## Dependency Analysis
- GroupShape and QuantKey define quantization metadata used across Python and tests.
- CUDA kernels depend on device capability and build flags; CMake sets target architectures and optional kernel builds.
- Triton configs are discovered per device and shape; missing configs fall back to defaults.

```mermaid
graph LR
QU["quant_utils.py<br/>GroupShape, QuantKey"] --> IQF["input_quant_fp8.py"]
IQF --> CFP8["common.cu"]
IU["int8_utils.py"] --> CI8["scaled_quant.cu"]
IU --> TI8["int8_utils.py<br/>Triton kernels"]
CMake["CMakeLists.txt"] --> Build["Target archs & kernel builds"]
Utils["utils.cmake"] --> Build
```

**Diagram sources**
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py#L1-L132)
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L1-L203)
- [common.cu](file://csrc/quantization/w8a8/fp8/common.cu#L1-L248)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L1-L475)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)
- [CMakeLists.txt](file://CMakeLists.txt#L900-L940)
- [utils.cmake](file://cmake/utils.cmake#L399-L425)

**Section sources**
- [quant_utils.py](file://vllm/model_executor/layers/quantization/utils/quant_utils.py#L1-L132)
- [CMakeLists.txt](file://CMakeLists.txt#L900-L940)
- [utils.cmake](file://cmake/utils.cmake#L399-L425)

## Performance Considerations
- FP8 GEMM benchmarks:
  - Compare BF16 baseline with FP8 variants using per-tensor and per-channel weight schemes and per-token or per-tensor activation schemes.
  - Demonstrates enabling per-token activation quantization and static activation scales.
- INT8 block-wise GEMM:
  - Device-specific configuration lookup improves occupancy and reduces overhead.
  - Block sizes must align with kernel constraints; defaults ensure fallback behavior.
- Memory bandwidth:
  - FP8 reduces activation bandwidth; INT8 reduces weight storage and increases arithmetic intensity.
- Attention and MoE:
  - PagedAttention partitions computation; FP8 activation quantization reduces KV cache bandwidth.
  - Block-wise INT8 GEMM accelerates expert projections in MoE layers.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- FP8 dynamic per-token quantization:
  - Ensure contiguous tensors and optional upper-bound scale is provided when needed.
  - On ROCm, verify aiter FP8 enablement and contiguity constraints.
- INT8 per-token/group quantization:
  - Group size must divide the hidden dimension; ensure tensors are contiguous.
  - For Triton kernels, confirm device capability and that rounding modes match expectations.
- Block-wise INT8 GEMM:
  - Verify block shapes and stride assumptions; ensure per-token and per-block scale layouts match kernel expectations.
- Quark scheme selection:
  - If FP8 W8A8 is unsupported, check GPU compute capability; ensure symmetric weight quantization and correct activation schemes.

**Section sources**
- [input_quant_fp8.py](file://vllm/model_executor/layers/quantization/input_quant_fp8.py#L67-L125)
- [int8_utils.py](file://vllm/model_executor/layers/quantization/utils/int8_utils.py#L191-L256)
- [scaled_quant.cu](file://csrc/quantization/w8a8/int8/scaled_quant.cu#L1-L328)
- [quark.py](file://vllm/model_executor/layers/quantization/quark/quark.py#L221-L322)

## Conclusion
vLLM’s FP8 and INT8 quantization stack provides flexible, high-performance inference:
- FP8 supports dynamic/static per-tensor/per-token and group quantization with ROCm aiter acceleration.
- INT8 enables W8A8 block-wise GEMM with per-token activation and per-block weight scaling.
- Integration points include attention (PagedAttention) and MoE layers, with device-aware kernel builds and configuration tuning.
- Benchmarks and scheme detection help users select optimal configurations for accuracy and throughput.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Configuration Examples
- FP8 W8A8 (dynamic activation):
  - Weight: static per-tensor or per-channel FP8.
  - Activation: per-tensor FP8 dynamic.
- FP8 W8A8 (static activation):
  - Weight: static per-tensor or per-channel FP8.
  - Activation: static per-tensor FP8.
- INT8 W8A8 (static):
  - Weight: static int8 per-tensor or per-channel (symmetric).
  - Activation: static int8 per-tensor (symmetric/asymmetric supported).
- INT8 W8A8 block-wise:
  - Weight: int8 per 128x128 blocks (example).
  - Activation: per-token/group quantization.

[No sources needed since this section provides general guidance]

### Accuracy and Benchmarks
- Reference FP8 dynamic per-tensor quantization:
  - Reproduces kernel-level operations to match CUDA kernel outputs precisely.
- FP8 GEMM benchmark:
  - Compares BF16 vs FP8 across model shapes and TP sizes; toggles per-token activation quantization and static activation scales.

**Section sources**
- [quant_utils.py](file://tests/kernels/quant_utils.py#L66-L101)
- [bench_fp8_gemm.py](file://benchmarks/kernels/bench_fp8_gemm.py#L1-L160)

### Hardware Compatibility and Architecture Builds
- Minimum GPU capability checks for FP8 schemes.
- CMake sets target architectures and conditionally builds kernels for specific architectures (e.g., Hopper).
- utils.cmake filters supported architectures and applies flags.

**Section sources**
- [quark.py](file://vllm/model_executor/layers/quantization/quark/quark.py#L205-L220)
- [CMakeLists.txt](file://CMakeLists.txt#L900-L940)
- [utils.cmake](file://cmake/utils.cmake#L399-L425)