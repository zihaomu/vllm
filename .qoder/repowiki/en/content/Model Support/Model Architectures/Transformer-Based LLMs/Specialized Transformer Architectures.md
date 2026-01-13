# Specialized Transformer Architectures

<cite>
**Referenced Files in This Document**
- [mamba.py](file://vllm/model_executor/models/mamba.py)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py)
- [jamba.py](file://vllm/model_executor/models/jamba.py)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py)
- [nemotron.py](file://vllm/model_executor/models/nemotron.py)
- [olmo.py](file://vllm/model_executor/models/olmo.py)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu)
- [test_mamba_ssm.py](file://tests/kernels/mamba/test_mamba_ssm.py)
- [jamba_tool_parser.py](file://tests/tool_parsers/test_jamba_tool_parser.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#trrobleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document explains specialized transformer architectures implemented in vLLM, focusing on Mamba (State Space Models), Mamba2, hybrid Jamba (Mamba-Transformer), Nemotron, OLMo, and Phi-series models. It details how these models depart from traditional attention mechanisms, how State Space Model computations and selective scan operations are implemented, and how hybrid Mamba-Transformer layers are structured. Practical guidance is included for loading specialized models, leveraging unique attention/backends, and optimizing memory and compute performance.

## Project Structure
The specialized architectures are implemented across model definitions, layer mixins, and CUDA kernels:
- Model backbones: Mamba, Mamba2, Jamba, Zamba2, Nemotron, OLMo, Phi-series
- Mamba-specific layers and operators: Mamba mixer, selective scan, convolution
- CUDA kernels for selective scan and related primitives
- Tests validating selective scan correctness and hybrid behavior

```mermaid
graph TB
subgraph "Models"
MAMBA["MambaModel<br/>mamba.py"]
MAMBA2["Mamba2Model<br/>mamba2.py"]
JAMBA["JambaForCausalLM<br/>jamba.py"]
ZAMBA2["Zamba2 Hybrid<br/>zamba2.py"]
NEMO["NemotronForCausalLM<br/>nemotron.py"]
OLMO["OlmoForCausalLM<br/>olmo.py"]
end
subgraph "Mamba Layers"
MIXER["MambaMixer<br/>mamba_mixer.py"]
MIXER2["MambaMixer2<br/>mamba_mixer2.py"]
end
subgraph "Kernels"
SCAN["Selective Scan CUDA<br/>selective_scan_fwd.cu"]
end
MAMBA --> MIXER
MAMBA2 --> MIXER2
JAMBA --> MIXER
ZAMBA2 --> MIXER
MIXER --> SCAN
MIXER2 --> SCAN
```

**Diagram sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)

## Core Components
- Mamba backbone: Implements a pure Mamba stack with selective scan and convolution, supporting prefix caching and state shape/dtype calculation helpers.
- Mamba2 backbone: Extends Mamba with gated RMS norm and grouped attention-like structures, enabling hybrid MoE/attention pathways.
- Jamba hybrid: Alternates attention and Mamba blocks, with shared transformer and Mamba pathways and MoE routing.
- Zamba2 hybrid: Combines a shared transformer pathway with a Mamba pathway, adding transformer output to Mamba input.
- Nemotron: Standard transformer with LayerNorm variant, squared ReLU activation, and SwiGLU replacement.
- OLMo: Transformer with attention and SiLU-Mul MLP, using fused projections and rotary embeddings.
- Mamba mixers: Provide selective scan and convolution operations, parameter initialization, and KV-cache state management.
- Selective scan CUDA: Optimized kernel implementing the core SSM recurrence with variable-length and state caching.

Key implementation highlights:
- State Space Model (SSM) computation via selective scan with A, B, C, D parameters and optional gating.
- Convolutional preprocessing using causal convolution prior to selective scan.
- Hybrid designs combine linear-time Mamba with attention/MoE blocks.

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L1-L500)
- [olmo.py](file://vllm/model_executor/models/olmo.py#L1-L413)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)

## Architecture Overview
The specialized architectures replace or complement attention with linear-time selective scan and convolution. Mamba and Mamba2 operate on sequences via a recurrent SSM that depends on the current token and internal state. Jamba and Zamba2 integrate attention/MoE alongside Mamba for richer representation learning.

```mermaid
graph TB
subgraph "Mamba Family"
MAMBA["MambaModel<br/>pure SSM"]
MAMBA2["Mamba2Model<br/>SSM + gated norms + groups"]
end
subgraph "Hybrid Families"
JAMBA["JambaForCausalLM<br/>alternating attention + Mamba"]
ZAMBA2["Zamba2 Hybrid<br/>shared transformer + Mamba"]
end
subgraph "Traditional Transformers"
NEMO["NemotronForCausalLM<br/>standard attention"]
OLMO["OlmoForCausalLM<br/>standard attention"]
end
MAMBA --> |"Selective Scan"| MAMBA
MAMBA2 --> |"Selective Scan + Gating"| MAMBA2
JAMBA --> |"Attention + Mamba blocks"| JAMBA
ZAMBA2 --> |"Transformer + Mamba pathway"| ZAMBA2
NEMO --> |"Attention + MLP"| NEMO
OLMO --> |"Attention + MLP"| OLMO
```

**Diagram sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L1-L500)
- [olmo.py](file://vllm/model_executor/models/olmo.py#L1-L413)

## Detailed Component Analysis

### Mamba (State Space Models)
Mamba replaces attention with a selective scan over tokens. The model applies a convolution to input features, then runs an SSM recurrence to produce contextualized states, optionally gated and projected.

```mermaid
classDiagram
class MambaModel {
+int vocab_size
+forward(input_ids, positions, ...)
+load_weights(weights)
}
class MambaDecoderLayer {
+forward(hidden_states, residual)
}
class MambaMixer {
+conv1d
+in_proj
+x_proj
+dt_proj
+out_proj
+dt_layernorm
+b_layernorm
+c_layernorm
+_ssm_transform(x)
}
MambaModel --> MambaDecoderLayer : "stacks"
MambaDecoderLayer --> MambaMixer : "uses"
```

- Convolution: causal convolution applied to intermediate activations before selective scan.
- Selective scan: computes recurrence y_t = f(y_{t-1}, x_t, parameters) using A, B, C, D and optional z-gating.
- State handling: per-head/group state maintained across tokens; supports initial states and caching.

Practical notes:
- State dtype/shape helpers exposed for efficient cache allocation.
- Weight loading handles special names (e.g., A_log to A) and pipeline parallel missing parameters.

**Diagram sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)

### Mamba2
Mamba2 extends Mamba with gated RMS norm and grouped structures. It integrates selective state updates with attention-like grouping and MoE routing in hybrid layers.

```mermaid
classDiagram
class Mamba2Model {
+forward(...)
}
class Mamba2DecoderLayer {
+forward(hidden_states, residual)
}
class MambaMixer2 {
+causal_conv1d
+selective_state_update(...)
+rms_norm_gated(...)
}
Mamba2Model --> Mamba2DecoderLayer
Mamba2DecoderLayer --> MambaMixer2
```

- Gated RMS norm: applies silu(gate) before normalization; supports group-wise reductions and tensor-parallel variants.
- Selective state update: optimized update kernel for incremental SSM computation.
- Hybrid integration: Jamba-style MoE and attention can be combined with Mamba2 blocks.

**Diagram sources**
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)

**Section sources**
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)

### Hybrid Jamba (Mamba-Transformer)
Jamba alternates attention and Mamba blocks, with MoE routing and optional attention-only layers.

```mermaid
sequenceDiagram
participant J as "JambaForCausalLM"
participant L as "JambaMambaDecoderLayer"
participant A as "JambaAttentionDecoderLayer"
participant M as "MambaMixer"
participant S as "Selectivity"
J->>L : forward(hidden_states, positions)
alt block type == "mamba"
L->>M : mixer(hidden_states, output)
M->>S : selective_scan(conv_out, A,B,C,D,...)
S-->>M : contextualized_states
M-->>L : output
else block type == "attention"
L->>A : self_attention(...)
A-->>L : attn_output
end
L-->>J : hidden_states, residual
```

- Layer selection: configured per-layer type (attention vs mamba).
- MoE routing: optional mixture-of-experts in attention or Mamba feed-forward.
- Hybrid residual: attention output can be routed through Mamba path in downstream layers.

**Diagram sources**
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)

**Section sources**
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)

### Zamba2 Hybrid
Zamba2 combines a shared transformer pathway with a Mamba pathway, adding transformer output to Mamba input.

```mermaid
flowchart TD
Start(["Input hidden states"]) --> LN["Input LayerNorm"]
LN --> Add["Add shared transformer output"]
Add --> MambaPath["Mamba Decoder Layer"]
MambaPath --> Residual["Residual connection"]
Residual --> End(["Output"])
```

- Shared transformer: processes input independently and projects to hidden size.
- Addition: adds transformer output to Mamba input before selective scan.
- Final residual: residual connection after Mamba.

**Diagram sources**
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)

**Section sources**
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)

### Nemotron
Nemotron follows a standard transformer stack with a LayerNorm variant and squared ReLU activation.

```mermaid
classDiagram
class NemotronModel {
+forward(...)
}
class NemotronDecoderLayer {
+forward(positions, hidden_states, residual)
}
class NemotronAttention {
+forward(positions, hidden_states)
}
class NemotronMLP {
+forward(x)
}
NemotronModel --> NemotronDecoderLayer
NemotronDecoderLayer --> NemotronAttention
NemotronDecoderLayer --> NemotronMLP
```

- Norm variant: LayerNorm with a constant added to weights.
- MLP: single up-projection with squared ReLU activation and down-projection.
- Attention: QKV projection, rotary embeddings, scaled attention.

**Diagram sources**
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L1-L500)

**Section sources**
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L1-L500)

### OLMo
OLMo implements a standard transformer with attention and SiLU-Mul MLP.

```mermaid
classDiagram
class OlmoModel {
+forward(...)
}
class OlmoDecoderLayer {
+forward(positions, hidden_states)
}
class OlmoAttention {
+forward(positions, hidden_states)
}
class OlmoMLP {
+forward(x)
}
OlmoModel --> OlmoDecoderLayer
OlmoDecoderLayer --> OlmoAttention
OlmoDecoderLayer --> OlmoMLP
```

- Attention: QKV fused projection, rotary embeddings, scaled attention.
- MLP: gate+up fused projection, SiLU-Mul activation, down-projection.

**Diagram sources**
- [olmo.py](file://vllm/model_executor/models/olmo.py#L1-L413)

**Section sources**
- [olmo.py](file://vllm/model_executor/models/olmo.py#L1-L413)

### Selective Scan Implementation and Kernels
The core SSM operation is implemented in CUDA with careful memory access patterns and shared memory usage.

```mermaid
flowchart TD
A["Inputs u, delta, A, B, C, D, z"] --> B["Preprocess: delta + bias, softplus if requested"]
B --> C["Load chunks of u and delta"]
C --> D["Compute exp2(delta * A) and delta*u"]
D --> E["Block inclusive scan over items"]
E --> F["Accumulate state and output contribution"]
F --> G{"Has initial state?"}
G --> |Yes| H["Load previous state from cache"]
G --> |No| I["Start with identity prefix"]
H --> E
I --> E
F --> J["Write final state to cache if enabled"]
F --> K["Apply gating z if present"]
K --> L["Write output"]
```

- Variable-length support: query_start_loc enables ragged sequences.
- State caching: APC mode writes/reads cached states per slot.
- Launch configuration: kernel chooses thread/block/stride parameters based on sequence length.

**Diagram sources**
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)

**Section sources**
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)
- [test_mamba_ssm.py](file://tests/kernels/mamba/test_mamba_ssm.py#L91-L678)

### Practical Model Loading Examples
- Mamba: The model loads weights with special handling for A_log to A and pipeline-parallel missing parameters.
- Jamba: Uses a mapper to normalize Hugging Face weight keys and stacks QKV gates appropriately.
- Nemotron/OLMo: Standard attention stacks with packed module mappings for QKV and gate/up projections.

Loading guidance:
- Use the model’s built-in load_weights method to handle sharding, missing layers, and packed projections.
- For hybrid models, ensure attention and MoE weights are mapped correctly.

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L171-L188)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L457-L569)
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L368-L423)
- [olmo.py](file://vllm/model_executor/models/olmo.py#L303-L339)

### Memory Optimization Strategies
- Prefix caching: Mamba backbones expose state dtype/shape calculators to allocate minimal cache sizes.
- Tensor parallel sharding: Linear projections and RMS norms are partitioned across TP ranks.
- Mixed-precision: Selective scan supports half/bfloat16 inputs with float32 accumulation for stability.
- Grouped attention in Mamba2: Reduces cross-head communication and can improve throughput.

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L238-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L188-L289)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)

## Dependency Analysis
The specialized models depend on:
- Mamba mixers for selective scan and convolution
- Attention layers for hybrid architectures
- CUDA kernels for efficient SSM computation
- Utility modules for state shape/dtype calculation and weight loading

```mermaid
graph LR
MAMBA["mamba.py"] --> MIXER["mamba_mixer.py"]
MAMBA2["mamba2.py"] --> MIXER2["mamba_mixer2.py"]
JAMBA["jamba.py"] --> MIXER
ZAMBA2["zamba2.py"] --> MIXER
MIXER --> SCAN["selective_scan_fwd.cu"]
MIXER2 --> SCAN
NEMO["nemotron.py"] --> ATT["Attention"]
OLMO["olmo.py"] --> ATT
```

**Diagram sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L1-L500)
- [olmo.py](file://vllm/model_executor/models/olmo.py#L1-L413)

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L1-L277)
- [mamba2.py](file://vllm/model_executor/models/mamba2.py#L1-L289)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L1-L610)
- [zamba2.py](file://vllm/model_executor/models/zamba2.py#L541-L649)
- [mamba_mixer.py](file://vllm/model_executor/layers/mamba/mamba_mixer.py#L1-L200)
- [mamba_mixer2.py](file://vllm/model_executor/layers/mamba/mamba_mixer2.py#L1-L200)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L1-L791)
- [nemotron.py](file://vllm/model_executor/models/nemotron.py#L1-L500)
- [olmo.py](file://vllm/model_executor/models/olmo.py#L1-L413)

## Performance Considerations
- Memory footprint:
  - Mamba: state size grows linearly with sequence length; caching reduces recomputation.
  - Attention: quadratic in sequence length; Mamba offers linear-time alternatives.
- Compute efficiency:
  - Selective scan kernel is optimized for shared memory and block scans; launch configs adapt to sequence length.
  - Mamba2 gated norms enable fused computation patterns.
- Scalability:
  - Tensor parallel sharding of projections and RMS norms scales across devices.
  - Hybrid Jamba layers can route tokens through MoE to balance compute.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
Common issues and resolutions:
- Incorrect state dtype/shape: Verify state dtype/shape calculations for Mamba backbones; ensure cache_config matches model dtype.
- Weight loading mismatches: Use model-specific load_weights and mapper for Jamba; ensure packed module mappings for attention/MLP.
- Selective scan errors:
  - Validate input layouts and strides (contiguous last dimension).
  - Confirm dstate limits and variable-B/C shapes for ragged sequences.
  - Check pad_slot_id and cache indices for APC mode.

Validation resources:
- Reference selective scan behavior against test implementations that split sequences and compare outputs.

**Section sources**
- [mamba.py](file://vllm/model_executor/models/mamba.py#L238-L277)
- [jamba.py](file://vllm/model_executor/models/jamba.py#L457-L569)
- [selective_scan_fwd.cu](file://csrc/mamba/mamba_ssm/selective_scan_fwd.cu#L619-L791)
- [test_mamba_ssm.py](file://tests/kernels/mamba/test_mamba_ssm.py#L91-L678)

## Conclusion
vLLM implements specialized transformer architectures that move beyond attention to linear-time selective scan (Mamba, Mamba2), while also supporting hybrid designs (Jamba, Zamba2) and traditional transformer baselines (Nemotron, OLMo). The CUDA selective scan kernel and hybrid layer designs enable scalable, memory-efficient inference. Proper model loading, state caching, and mixed-precision configurations are essential for optimal performance.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices
- Tool parser tests for Jamba demonstrate hybrid tool-use parsing patterns in hybrid architectures.

**Section sources**
- [jamba_tool_parser.py](file://tests/tool_parsers/test_jamba_tool_parser.py)