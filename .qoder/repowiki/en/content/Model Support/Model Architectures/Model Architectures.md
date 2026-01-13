# Model Architectures

<cite>
**Referenced Files in This Document**
- [registry.py](file://vllm/model_executor/models/registry.py)
- [llama.py](file://vllm/model_executor/models/llama.py)
- [mixtral.py](file://vllm/model_executor/models/mixtral.py)
- [qwen2.py](file://vllm/model_executor/models/qwen2.py)
- [gemma.py](file://vllm/model_executor/models/gemma.py)
- [deepseek_v2.py](file://vllm/model_executor/models/deepseek_v2.py)
- [deepseek_mtp.py](file://vllm/model_executor/models/deepseek_mtp.py)
- [whisper.py](file://vllm/model_executor/models/whisper.py)
- [internvl.py](file://vllm/model_executor/models/internvl.py)
- [gemma3n_mm.py](file://vllm/model_executor/models/gemma3n_mm.py)
- [hyperclovax_vision.py](file://vllm/model_executor/models/hyperclovax_vision.py)
- [qwen3_omni_moe_thinker.py](file://vllm/model_executor/models/qwen3_omni_moe_thinker.py)
- [parameter.py](file://vllm/model_executor/parameter.py)
- [bitsandbytes_loader.py](file://vllm/model_executor/model_loader/bitsandbytes_loader.py)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py)
- [modular_kernel.py](file://vllm/model_executor/layers/fused_moe/modular_kernel.py)
- [selector.py](file://vllm/attention/selector.py)
- [cuda.py](file://vllm/platforms/cuda.py)
- [parallel.py](file://vllm/config/parallel.py)
- [parallel_state.py](file://vllm/distributed/parallel_state.py)
- [benchmark_moe.py](file://benchmarks/kernels/benchmark_moe.py)
- [only_thinker.py](file://examples/offline_inference/qwen2_5_omni/only_thinker.py)
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
This document explains vLLM’s model architectures ecosystem with a focus on how models are registered, discovered, and executed. It covers:
- The model registry and discovery mechanism
- Supported transformer-based LLM families (e.g., Llama, Mistral, Gemma, Qwen)
- Mixture-of-Experts (MoE) models (e.g., Mixtral, DeepSeek MoE, Qwen MoE)
- Multimodal models (vision-language, audio, and video)
- Model loading, parameter initialization, and memory management
- Practical guidance for registering custom models and leveraging architecture-specific optimizations
- How attention backends, quantization, and distributed strategies relate to model architectures

## Project Structure
At a high level, vLLM organizes model implementations under a dedicated models package and integrates them via a registry. The registry maps Hugging Face-style architectures to vLLM’s internal model classes and capabilities. Supporting modules handle attention backend selection, quantization, distributed parallelism, and multimodal processing.

```mermaid
graph TB
Registry["Model Registry<br/>registry.py"] --> Impl["Model Implementations<br/>models/*.py"]
Impl --> Llama["Llama<br/>llama.py"]
Impl --> Mixtral["Mixtral<br/>mixtral.py"]
Impl --> Qwen2["Qwen2<br/>qwen2.py"]
Impl --> Gemma["Gemma/Gemma2/Gemma3<br/>gemma.py"]
Impl --> DeepSeek["DeepSeek MoE<br/>deepseek_v2.py"]
Impl --> Whisper["Whisper<br/>whisper.py"]
Impl --> InternVL["InternVL<br/>internvl.py"]
Impl --> Gemma3nMM["Gemma3n Multimodal<br/>gemma3n_mm.py"]
Impl --> HyperClova["HyperClova Vision<br/>hyperclovax_vision.py"]
Impl --> Qwen3Omni["Qwen3 Omni MoE Thinker<br/>qwen3_omni_moe_thinker.py"]
Registry --> Discovery["Discovery & Lazy Import<br/>registry.py"]
Impl --> Params["Parameter Init & Weight Loader<br/>parameter.py"]
Impl --> Quant["Quantization Backends<br/>auto_round.py, bitblas.py"]
Impl --> Dist["Distributed Parallelism<br/>parallel_state.py"]
Impl --> Attn["Attention Backend Selection<br/>selector.py + cuda.py"]
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L450-L503)
- [llama.py](file://vllm/model_executor/models/llama.py#L1-L200)
- [mixtral.py](file://vllm/model_executor/models/mixtral.py#L1-L200)
- [qwen2.py](file://vllm/model_executor/models/qwen2.py#L1-L200)
- [gemma.py](file://vllm/model_executor/models/gemma.py#L204-L399)
- [deepseek_v2.py](file://vllm/model_executor/models/deepseek_v2.py#L226-L1359)
- [deepseek_mtp.py](file://vllm/model_executor/models/deepseek_mtp.py#L183-L213)
- [whisper.py](file://vllm/model_executor/models/whisper.py#L1-L200)
- [internvl.py](file://vllm/model_executor/models/internvl.py#L1331-L1364)
- [gemma3n_mm.py](file://vllm/model_executor/models/gemma3n_mm.py#L644-L664)
- [hyperclovax_vision.py](file://vllm/model_executor/models/hyperclovax_vision.py#L745-L780)
- [qwen3_omni_moe_thinker.py](file://vllm/model_executor/models/qwen3_omni_moe_thinker.py#L1539-L1563)
- [parameter.py](file://vllm/model_executor/parameter.py#L433-L555)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [selector.py](file://vllm/attention/selector.py#L1-L146)
- [cuda.py](file://vllm/platforms/cuda.py#L287-L380)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1254-L1577)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L450-L503)

## Core Components
- Model Registry and Discovery
  - Central registry maps Hugging Face architecture names to vLLM model implementations and capabilities. It supports both eager and lazy model class resolution, with caching to avoid repeated inspection.
  - External models can be registered via a module:class string or a direct class reference.

- Attention Backend Selection
  - vLLM selects attention backends based on device capability, dtype, KV cache dtype, block size, and model characteristics. This ensures optimal performance per architecture and hardware.

- Quantization Backends
  - Multiple quantization backends are integrated (e.g., AutoRound, BitBLAS). Backend selection considers platform, CPU/XPU vs GPU, and packing formats.

- Distributed Parallelism
  - vLLM supports tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), and data parallelism (DP). Parallel groups are initialized and managed centrally.

- Multimodal Support
  - Models integrate multimodal inputs (images, videos, audio) with specialized embedding and attention layers. Registry entries enumerate supported multimodal architectures.

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L743)
- [selector.py](file://vllm/attention/selector.py#L1-L146)
- [cuda.py](file://vllm/platforms/cuda.py#L287-L380)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1254-L1577)

## Architecture Overview
The model architecture ecosystem is driven by the registry and layered around:
- Architecture-to-implementation mapping
- Capability introspection (text generation, pooling, multimodal, transcription, etc.)
- Lazy model class loading and caching
- Attention backend selection tailored to head size, dtype, and cache configuration
- Quantization backends chosen per platform and model needs
- Distributed parallelism orchestration

```mermaid
sequenceDiagram
participant User as "User"
participant Registry as "Model Registry<br/>registry.py"
participant Platform as "Platform/CUDA<br/>cuda.py"
participant Selector as "Attention Selector<br/>selector.py"
participant Impl as "Model Implementation<br/>models/*.py"
User->>Registry : Resolve architecture name
Registry->>Registry : Lookup mapping and capabilities
Registry->>Impl : Lazy import model class (if needed)
Impl-->>Registry : Model class ready
Registry-->>User : Model info and class
User->>Platform : Request attention backend
Platform->>Selector : get_attn_backend(...)
Selector-->>Platform : Backend class path
Platform-->>User : Backend selected
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L743)
- [selector.py](file://vllm/attention/selector.py#L1-L146)
- [cuda.py](file://vllm/platforms/cuda.py#L287-L380)

## Detailed Component Analysis

### Model Registry and Registration
- Registry maps Hugging Face architecture names to vLLM implementations and capability flags (text generation, pooling, multimodal, transcription, etc.).
- Supports external registration via:
  - A string in module:class form for lazy import (avoiding CUDA initialization in subprocess)
  - A direct PyTorch Module subclass
- Inspects model classes in a subprocess to avoid CUDA initialization and caches results for reuse.

```mermaid
classDiagram
class _ModelRegistry {
+get_supported_archs() Set
+register_model(model_arch, model_cls) void
-_raise_for_unsupported(architectures) void
}
class _BaseRegisteredModel {
<<abstract>>
+inspect_model_cls() _ModelInfo
+load_model_cls() type
}
class _RegisteredModel {
+interfaces _ModelInfo
+model_cls type
+inspect_model_cls() _ModelInfo
+load_model_cls() type
}
class _LazyRegisteredModel {
+module_name str
+class_name str
+inspect_model_cls() _ModelInfo
+load_model_cls() type
}
class _ModelInfo {
+architecture str
+is_text_generation_model bool
+is_pooling_model bool
+attn_type str
+supports_multimodal bool
+supports_transcription bool
+...
}
_ModelRegistry --> _BaseRegisteredModel : "stores"
_BaseRegisteredModel <|-- _RegisteredModel
_BaseRegisteredModel <|-- _LazyRegisteredModel
_RegisteredModel --> _ModelInfo : "wraps"
_LazyRegisteredModel --> _ModelInfo : "inspects"
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L743)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L743)

### Transformer-Based LLMs
- Llama family
  - Implements attention with QKV projection, rotary embeddings, and RMSNorm. Supports tensor parallelism and pipeline parallelism.
  - Example references:
    - [LlamaAttention](file://vllm/model_executor/models/llama.py#L115-L200)
    - [LlamaMLP](file://vllm/model_executor/models/llama.py#L72-L113)

- Qwen2 family
  - Similar decoder-only structure with fused SiLU-Multiply and RMSNorm. Supports QK normalization and dual-chunk attention configurations.
  - Example references:
    - [Qwen2Attention](file://vllm/model_executor/models/qwen2.py#L112-L200)
    - [Qwen2MLP](file://vllm/model_executor/models/qwen2.py#L75-L110)

- Gemma family
  - Decoder-only with RMSNorm and fused SiLU-Multiply. Integrates with quantization and LoRA support.
  - Example references:
    - [GemmaDecoderLayer](file://vllm/model_executor/models/gemma.py#L204-L236)
    - [GemmaForCausalLM](file://vllm/model_executor/models/gemma.py#L367-L399)

```mermaid
flowchart TD
Start(["Load Llama/Qwen2/Gemma"]) --> CheckTP["Check TP world size"]
CheckTP --> SplitHeads["Split heads across TP ranks"]
SplitHeads --> QKV["QKV Projection (Parallel)"]
QKV --> RoPE["Rotary Embedding"]
RoPE --> Attn["Attention"]
Attn --> FFN["MLP (Gate+Up/Down)"]
FFN --> Norms["RMSNorm + Residual"]
Norms --> NextLayer["Repeat for N layers"]
NextLayer --> End(["Output Hidden States"])
```

**Diagram sources**
- [llama.py](file://vllm/model_executor/models/llama.py#L115-L200)
- [qwen2.py](file://vllm/model_executor/models/qwen2.py#L112-L200)
- [gemma.py](file://vllm/model_executor/models/gemma.py#L204-L236)

**Section sources**
- [llama.py](file://vllm/model_executor/models/llama.py#L72-L200)
- [qwen2.py](file://vllm/model_executor/models/qwen2.py#L75-L200)
- [gemma.py](file://vllm/model_executor/models/gemma.py#L204-L399)

### Mixture-of-Experts (MoE) Models
- Mixtral
  - Uses a fused MoE kernel with expert parallelism and optional redundant experts for load balancing. Gate and experts are configured per vLLM’s parallel settings.
  - Example references:
    - [MixtralMoE](file://vllm/model_executor/models/mixtral.py#L74-L153)

- DeepSeek MoE
  - DeepseekV2/DeepseekV3 MoE with routed scaling factor and expert groups. Supports expert parallelism and sequence-parallel MoE.
  - Example references:
    - [DeepseekV2MoE](file://vllm/model_executor/models/deepseek_v2.py#L233-L250)
    - [DeepseekV2MixtureOfExperts](file://vllm/model_executor/models/deepseek_v2.py#L1336-L1359)
    - [DeepSeekMTP](file://vllm/model_executor/models/deepseek_mtp.py#L183-L213)

- Qwen MoE
  - Qwen2Moe, Qwen3Moe, Qwen3Next, and Qwen3Omni MoE variants. Benchmark utilities enumerate expert counts and top-K routing parameters.
  - Example references:
    - [benchmark_moe.py](file://benchmarks/kernels/benchmark_moe.py#L578-L612)

```mermaid
sequenceDiagram
participant User as "User"
participant Mixtral as "MixtralMoE<br/>mixtral.py"
participant EP as "Expert Parallel Group"
participant FusedMoE as "FusedMoE Kernel"
User->>Mixtral : Forward hidden_states
Mixtral->>EP : Compute router logits (replicated)
EP-->>Mixtral : Top-k experts per token
Mixtral->>FusedMoE : FusedMoE(experts, router_logits)
FusedMoE-->>Mixtral : Local expert outputs
Mixtral-->>User : Reduced outputs
```

**Diagram sources**
- [mixtral.py](file://vllm/model_executor/models/mixtral.py#L74-L153)

**Section sources**
- [mixtral.py](file://vllm/model_executor/models/mixtral.py#L74-L153)
- [deepseek_v2.py](file://vllm/model_executor/models/deepseek_v2.py#L233-L250)
- [deepseek_mtp.py](file://vllm/model_executor/models/deepseek_mtp.py#L183-L213)
- [benchmark_moe.py](file://benchmarks/kernels/benchmark_moe.py#L578-L612)

### Multimodal Models
- Vision-Language Models
  - LLaVA, Qwen-VL, InternVL, and others integrate visual encoders and language decoders. They embed images/videos and feed tokens into the language model.
  - Example references:
    - [InternVL multimodal embedding](file://vllm/model_executor/models/internvl.py#L1331-L1364)
    - [Gemma3n multimodal embedding](file://vllm/model_executor/models/gemma3n_mm.py#L644-L664)
    - [HyperClova multimodal forward](file://vllm/model_executor/models/hyperclovax_vision.py#L745-L780)

- Audio Models
  - Whisper encoder-decoder architecture with cross-attention and positional embeddings for speech-to-text.
  - Example references:
    - [Whisper encoder attention](file://vllm/model_executor/models/whisper.py#L144-L170)
    - [Whisper positional embedding](file://vllm/model_executor/models/whisper.py#L172-L178)

- Video Understanding
  - Qwen2_5 Omni and Qwen3 Omni MoE Thinker support mixed modalities including images, audio, and videos, with position ID handling across modalities.
  - Example references:
    - [Qwen2_5 Omni multimodal inputs](file://examples/offline_inference/qwen2_5_omni/only_thinker.py#L34-L73)
    - [Qwen3 Omni MoE Thinker position IDs](file://vllm/model_executor/models/qwen3_omni_moe_thinker.py#L1539-L1563)

```mermaid
flowchart TD
Parse["Parse Multimodal Inputs"] --> Img["Image Processor"]
Parse --> Vid["Video Processor"]
Parse --> Aud["Audio Processor"]
Img --> Embeds["Per-modality Embeddings"]
Vid --> Embeds
Aud --> Embeds
Embeds --> Concat["Concatenate Tokens"]
Concat --> LLM["Language Model Forward"]
LLM --> Output["Text Generation / Captioning"]
```

**Diagram sources**
- [internvl.py](file://vllm/model_executor/models/internvl.py#L1331-L1364)
- [gemma3n_mm.py](file://vllm/model_executor/models/gemma3n_mm.py#L644-L664)
- [whisper.py](file://vllm/model_executor/models/whisper.py#L144-L178)
- [only_thinker.py](file://examples/offline_inference/qwen2_5_omni/only_thinker.py#L34-L73)
- [qwen3_omni_moe_thinker.py](file://vllm/model_executor/models/qwen3_omni_moe_thinker.py#L1539-L1563)

**Section sources**
- [internvl.py](file://vllm/model_executor/models/internvl.py#L1331-L1364)
- [gemma3n_mm.py](file://vllm/model_executor/models/gemma3n_mm.py#L644-L664)
- [whisper.py](file://vllm/model_executor/models/whisper.py#L144-L178)
- [only_thinker.py](file://examples/offline_inference/qwen2_5_omni/only_thinker.py#L34-L73)
- [qwen3_omni_moe_thinker.py](file://vllm/model_executor/models/qwen3_omni_moe_thinker.py#L1539-L1563)

### Model Loading Pipeline, Parameter Initialization, and Memory Management
- Parameter Initialization
  - vLLM uses a parameter abstraction to manage weight partitions and loading. Partitioned parameters ensure correctness when weights are split across ranks or devices.
  - Example references:
    - [PartitionedModelWeightParameter](file://vllm/model_executor/parameter.py#L433-L555)

- Weight Loading and Quantization
  - BitsAndBytes loader enforces constraints for pre-quantized models under tensor parallelism. Quantization backends (AutoRound, BitBLAS) adapt to platform and packing formats.
  - Example references:
    - [BitsAndBytes loader constraints](file://vllm/model_executor/model_loader/bitsandbytes_loader.py#L554-L572)
    - [AutoRound backend selection](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
    - [BitBLAS version checks](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)

- Memory Management
  - Attention backends and KV cache layouts are selected to match hardware capabilities and model characteristics. This affects memory footprint and throughput.

```mermaid
sequenceDiagram
participant Engine as "Engine"
participant Loader as "Weight Loader<br/>bitsandbytes_loader.py"
participant Params as "Parameters<br/>parameter.py"
participant Quant as "Quantization<br/>auto_round.py / bitblas.py"
Engine->>Loader : Initialize with model and config
Loader->>Params : Map and load partitions
Params-->>Loader : Partitioned parameters
Loader->>Quant : Apply quantization backend
Quant-->>Loader : Quantized parameters
Loader-->>Engine : Ready-to-use weights
```

**Diagram sources**
- [bitsandbytes_loader.py](file://vllm/model_executor/model_loader/bitsandbytes_loader.py#L554-L572)
- [parameter.py](file://vllm/model_executor/parameter.py#L433-L555)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)

**Section sources**
- [parameter.py](file://vllm/model_executor/parameter.py#L433-L555)
- [bitsandbytes_loader.py](file://vllm/model_executor/model_loader/bitsandbytes_loader.py#L554-L572)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)

### Practical Examples
- Registering a Custom Model
  - Use the registry’s register_model API with either a module:class string (lazy) or a direct class. This enables discovery and capability introspection.
  - Reference: [register_model](file://vllm/model_executor/models/registry.py#L753-L799)

- Implementing a Custom Model
  - Follow the pattern of existing models: define attention, MLP, norms, and embedding layers; expose packed module mappings if applicable; integrate with quantization and parallelism helpers.
  - References:
    - [LlamaAttention](file://vllm/model_executor/models/llama.py#L115-L200)
    - [Qwen2Attention](file://vllm/model_executor/models/qwen2.py#L112-L200)
    - [MixtralMoE](file://vllm/model_executor/models/mixtral.py#L74-L153)

- Architecture-Specific Optimizations
  - Enable fused kernels for MoE (e.g., FusedMoE) and select attention backends appropriate for head size and dtype.
  - References:
    - [FusedMoE modular kernel interface](file://vllm/model_executor/layers/fused_moe/modular_kernel.py#L192-L208)
    - [Attention backend selection](file://vllm/attention/selector.py#L1-L146)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L753-L799)
- [llama.py](file://vllm/model_executor/models/llama.py#L115-L200)
- [qwen2.py](file://vllm/model_executor/models/qwen2.py#L112-L200)
- [mixtral.py](file://vllm/model_executor/models/mixtral.py#L74-L153)
- [modular_kernel.py](file://vllm/model_executor/layers/fused_moe/modular_kernel.py#L192-L208)
- [selector.py](file://vllm/attention/selector.py#L1-L146)

## Dependency Analysis
- Registry to Implementations
  - The registry aggregates architecture mappings and exposes capability flags. Implementations depend on attention, linear layers, and quantization abstractions.
- Attention Backends
  - Platform-specific selection depends on device capability and configuration. Backends may require specific KV cache layouts.
- Quantization Backends
  - Selection varies by platform (CPU/XPU vs GPU) and packing format. Some backends impose constraints (e.g., BitsAndBytes with TP).
- Distributed Strategies
  - Parallel groups (TP, PP, EP, DP) are initialized centrally and influence model layer behavior (e.g., head partitioning, expert distribution).

```mermaid
graph LR
Registry["registry.py"] --> Impl["models/*.py"]
Impl --> AttnSel["attention/selector.py"]
AttnSel --> CUDA["platforms/cuda.py"]
Impl --> QuantSel["quantization/backends"]
Impl --> Dist["distributed/parallel_state.py"]
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L450-L503)
- [selector.py](file://vllm/attention/selector.py#L1-L146)
- [cuda.py](file://vllm/platforms/cuda.py#L287-L380)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1254-L1577)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L450-L503)
- [selector.py](file://vllm/attention/selector.py#L1-L146)
- [cuda.py](file://vllm/platforms/cuda.py#L287-L380)
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L1254-L1577)

## Performance Considerations
- Attention Backend Selection
  - Choose backends aligned with head size, dtype, and KV cache configuration. Some backends require specific KV cache layouts.
  - Reference: [get_attn_backend](file://vllm/attention/selector.py#L46-L118)

- Quantization Compatibility
  - On CPU/XPU or with specific packing formats, AutoRound and BitBLAS backends apply different optimizations. Pre-quantized BitsAndBytes with TP is not supported; consider pipeline parallelism instead.
  - References:
    - [AutoRound backend selection](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
    - [BitBLAS version checks](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
    - [BitsAndBytes TP constraint](file://vllm/model_executor/model_loader/bitsandbytes_loader.py#L554-L572)

- Distributed Inference Strategies
  - Scale across nodes using tensor and pipeline parallelism. Ensure sufficient GPU memory and verify backend logs indicating KV cache sizing and concurrency.
  - Reference: [parallelism scaling guide](file://docs/serving/parallelism_scaling.md#L1-L19)

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Unsupported Attention Backend
  - If the selected backend is invalid for the configuration, platform selection raises an error. Verify head size, dtype, and KV cache dtype.
  - Reference: [platform backend validation](file://vllm/platforms/cuda.py#L287-L358)

- BitsAndBytes Prequantization with TP
  - Prequantized BitsAndBytes models do not support tensor parallelism. Switch to pipeline parallelism or disable prequantization.
  - Reference: [loader constraint](file://vllm/model_executor/model_loader/bitsandbytes_loader.py#L554-L572)

- Model Not Found in Registry
  - Ensure the architecture name matches the registry mapping. For custom models, register via module:class string or direct class.
  - Reference: [register_model](file://vllm/model_executor/models/registry.py#L753-L799)

**Section sources**
- [cuda.py](file://vllm/platforms/cuda.py#L287-L358)
- [bitsandbytes_loader.py](file://vllm/model_executor/model_loader/bitsandbytes_loader.py#L554-L572)
- [registry.py](file://vllm/model_executor/models/registry.py#L753-L799)

## Conclusion
vLLM’s model architectures ecosystem combines a robust registry, architecture-specific implementations, and flexible attention, quantization, and distributed strategies. By leveraging the registry for discovery, selecting appropriate attention backends, applying compatible quantization, and orchestrating distributed parallelism, users can efficiently deploy a wide range of transformer-based LLMs, MoE models, and multimodal systems.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices
- Appendix A: Supported Architectures Overview
  - The registry consolidates supported architectures across text generation, embedding, cross-encoder, multimodal, and speculative decoding domains. Refer to the registry constants for the authoritative list.
  - Reference: [VLLM_MODELS aggregation](file://vllm/model_executor/models/registry.py#L495-L503)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L495-L503)