# Custom Model Registration

<cite>
**Referenced Files in This Document**
- [registry.py](file://vllm/model_executor/models/registry.py)
- [interfaces.py](file://vllm/model_executor/models/interfaces.py)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py)
- [selector.py](file://vllm/attention/selector.py)
- [test_registry.py](file://tests/models/test_registry.py)
- [registry.py (tests)](file://tests/models/registry.py)
- [__init__.py](file://vllm/model_executor/models/__init__.py)
- [model.py](file://vllm/config/model.py)
- [platform_interface.py](file://vllm/platforms/interface.py)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py)
- [README.md](file://examples/offline_inference/basic/README.md)
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
This document explains how to register custom model architectures into the vLLM ecosystem. It covers the model registry system, model interface requirements, capability flags, feature detection, inspection and lazy-loading mechanisms, caching strategies for model metadata, and how registration affects attention backend selection, quantization compatibility, and distributed inference. Practical examples demonstrate integrating custom transformer models, multimodal architectures, and specialized inference models.

## Project Structure
The model registration and capability detection system spans several modules:
- Registry and capability inspection: vllm/model_executor/models/registry.py
- Capability flags and protocol interfaces: vllm/model_executor/models/interfaces.py and interfaces_base.py
- Attention backend selection: vllm/attention/selector.py
- Platform capability verification: vllm/platforms/interface.py
- Quantization integration points: vllm/model_executor/layers/quantization/*
- Model configuration integration: vllm/config/model.py
- Tests and examples: tests/models/* and examples/*

```mermaid
graph TB
subgraph "Model Registry"
R["registry.py<br/>_ModelRegistry, _RegisteredModel,<br/>_LazyRegisteredModel, _ModelInfo"]
end
subgraph "Interfaces"
I["interfaces.py<br/>Capabilities: multimodal, LoRA, PP, transcription,<br/>attention-free, hybrid, MoE, prefix-caching"]
IB["interfaces_base.py<br/>Base protocols: VllmModel, TextGeneration, Pooling"]
end
subgraph "Platform & Attention"
P["platform_interface.py<br/>Device capability checks"]
A["selector.py<br/>Attention backend selection"]
end
subgraph "Quantization"
Q1["kv_cache.py<br/>KV cache scale creation"]
Q2["auto_round.py<br/>Quant method selection"]
Q3["gptq_bitblas.py<br/>BitBLAS GPTQ checks"]
Q4["bitblas.py<br/>BitBLAS availability checks"]
end
subgraph "Config"
C["model.py<br/>ModelConfig integration,<br/>runner/pooling conversion"]
end
subgraph "Tests & Examples"
T["tests/models/test_registry.py"]
TE["tests/models/registry.py"]
E["examples/offline_inference/basic/README.md"]
end
R --> I
R --> IB
R --> P
R --> A
R --> C
R --> Q1
R --> Q2
R --> Q3
R --> Q4
T --> R
TE --> R
E --> R
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L748)
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L44-L198)
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [platform_interface.py](file://vllm/platforms/interface.py#L279-L323)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [model.py](file://vllm/config/model.py#L494-L519)
- [test_registry.py](file://tests/models/test_registry.py#L30-L121)
- [registry.py (tests)](file://tests/models/registry.py#L982-L1025)
- [README.md](file://examples/offline_inference/basic/README.md#L1-L81)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L748)
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L44-L198)
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [platform_interface.py](file://vllm/platforms/interface.py#L279-L323)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [model.py](file://vllm/config/model.py#L494-L519)
- [test_registry.py](file://tests/models/test_registry.py#L30-L121)
- [registry.py (tests)](file://tests/models/registry.py#L982-L1025)
- [README.md](file://examples/offline_inference/basic/README.md#L1-L81)

## Core Components
- Model registry and registration API:
  - _ModelRegistry: stores model architectures and their registered handlers (_RegisteredModel or _LazyRegisteredModel).
  - _RegisteredModel: wraps an already-imported model class and its _ModelInfo.
  - _LazyRegisteredModel: enables lazy import via module:class string, with inspection and caching.
  - _ModelInfo: encapsulates capability flags and metadata derived from a model class.
- Capability detection:
  - Interfaces in interfaces.py define capability flags (multimodal, LoRA, PP, transcription, attention-free, hybrid, MoE, prefix-caching, cross-encoding, no-ops).
  - Base interfaces in interfaces_base.py define minimal model contract and decorators for pooling/attention type defaults.
- Inspection and lazy loading:
  - LazyRegisteredModel inspects a model class in a subprocess to avoid CUDA initialization in the parent process.
  - Caching of _ModelInfo persists across runs keyed by module hash.
- Attention backend selection:
  - selector.get_attn_backend chooses the appropriate backend based on dtype, head size, KV cache dtype, and platform capabilities.
- Platform capability verification:
  - Platforms verify device capability compatibility before loading a model architecture.
- Quantization compatibility:
  - Quantization backends select appropriate methods based on platform and packing format.
- Model configuration integration:
  - ModelConfig resolves architecture and runner/pooling conversions, invoking registry inspection early.

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L748)
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L44-L198)
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [platform_interface.py](file://vllm/platforms/interface.py#L279-L323)
- [model.py](file://vllm/config/model.py#L494-L519)

## Architecture Overview
The registration and runtime flow integrates model discovery, capability inspection, backend selection, and platform validation.

```mermaid
sequenceDiagram
participant User as "User Code"
participant Config as "ModelConfig"
participant Registry as "_ModelRegistry"
participant Inspect as "_LazyRegisteredModel.inspect_model_cls"
participant SubProc as "Subprocess"
participant Platform as "current_platform"
participant BackendSel as "get_attn_backend"
User->>Config : Initialize with model architecture(s)
Config->>Registry : inspect_model_cls(architectures, self)
Registry->>Inspect : Load and inspect model class
Inspect->>SubProc : Run inspection in isolated process
SubProc-->>Inspect : _ModelInfo (capabilities)
Inspect-->>Registry : _ModelInfo
Registry->>Platform : verify_model_arch(model_arch)
Platform-->>Registry : OK or error
Registry-->>Config : Resolved model_info, normalized arch
Config->>BackendSel : Select attention backend
BackendSel-->>Config : Backend class
Config-->>User : Ready to run inference
```

**Diagram sources**
- [model.py](file://vllm/config/model.py#L494-L519)
- [registry.py](file://vllm/model_executor/models/registry.py#L745-L936)
- [registry.py](file://vllm/model_executor/models/registry.py#L676-L743)
- [platform_interface.py](file://vllm/platforms/interface.py#L279-L323)
- [selector.py](file://vllm/attention/selector.py#L46-L118)

## Detailed Component Analysis

### Model Registry and Registration API
- Registration accepts either:
  - A direct PyTorch module class (subclass of nn.Module).
  - A string "<module>:<class>" for lazy import.
- Overwrites are warned; supports normalization via architecture defaults.
- Provides inspection and loading helpers with LRU caching and subprocess isolation.

```mermaid
classDiagram
class _BaseRegisteredModel {
<<abstract>>
+inspect_model_cls() _ModelInfo
+load_model_cls() type[nn.Module]
}
class _RegisteredModel {
+interfaces _ModelInfo
+model_cls type[nn.Module]
+from_model_cls(model_cls) _RegisteredModel
+inspect_model_cls() _ModelInfo
+load_model_cls() type[nn.Module]
}
class _LazyRegisteredModel {
+module_name string
+class_name string
+inspect_model_cls() _ModelInfo
+load_model_cls() type[nn.Module]
-_load_modelinfo_from_cache(hash) _ModelInfo|None
-_save_modelinfo_to_cache(mi, hash) void
}
class _ModelRegistry {
+models dict[str, _BaseRegisteredModel]
+register_model(model_arch, model_cls) void
+get_supported_archs() set[str]
+inspect_model_cls(archs, model_config) (_ModelInfo, str)
}
class _ModelInfo {
+architecture string
+is_text_generation_model bool
+is_pooling_model bool
+attn_type AttnTypeStr
+default_pooling_type PoolingTypeStr
+supports_multimodal bool
+supports_pp bool
+is_attention_free bool
+is_hybrid bool
+has_inner_state bool
+supports_mamba_prefix_caching bool
+supports_transcription bool
+supports_transcription_only bool
+from_model_cls(model) _ModelInfo
}
_BaseRegisteredModel <|-- _RegisteredModel
_BaseRegisteredModel <|-- _LazyRegisteredModel
_ModelRegistry --> _BaseRegisteredModel : "stores"
_ModelInfo <.. _RegisteredModel : "constructed from"
_ModelInfo <.. _LazyRegisteredModel : "constructed from"
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L748)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L745-L936)
- [registry.py](file://vllm/model_executor/models/registry.py#L576-L748)

### Capability Flags and Feature Detection
- Capability flags are detected via protocol checks and class attributes:
  - Multimodal: supports_multimodal, supports_multimodal_raw_input_only, supports_encoder_tp_data
  - Pipeline parallel: supports_pp with forward signature acceptance of intermediate_tensors
  - LoRA: supports_lora with required mappings
  - Transcription: supports_transcription and supports_transcription_only
  - Attention characteristics: is_attention_free, is_hybrid, supports_mamba_prefix_caching
  - Others: has_inner_state, has_noops, supports_cross_encoding
- Decorators and helpers set defaults for pooling and attention types.

```mermaid
flowchart TD
Start(["Inspect model class"]) --> Detect["Detect capability flags<br/>via protocols and attributes"]
Detect --> Flags{"Flags present?"}
Flags --> |Yes| BuildInfo["Build _ModelInfo with flags"]
Flags --> |No| Warn["Log warnings for missing attributes"]
BuildInfo --> Done(["Return _ModelInfo"])
Warn --> Done
```

**Diagram sources**
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L111-L198)

**Section sources**
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L111-L198)

### Model Inspection, Lazy Loading, and Caching
- LazyRegisteredModel:
  - Uses module hash to locate cache file.
  - Loads cached _ModelInfo if hash matches; otherwise inspects in a subprocess.
  - Saves _ModelInfo to cache after successful inspection.
- LRU caching:
  - _try_inspect_model_cls and _try_load_model_cls cache results to avoid repeated inspection/loading.
- Subprocess isolation:
  - Prevents CUDA initialization in the parent process during inspection.

```mermaid
sequenceDiagram
participant Reg as "_ModelRegistry"
participant LR as "_LazyRegisteredModel"
participant FS as "Cache Filesystem"
participant Proc as "Subprocess"
Reg->>LR : inspect_model_cls()
LR->>FS : Read cache by module hash
alt Cache hit
FS-->>LR : _ModelInfo
LR-->>Reg : _ModelInfo
else Cache miss
LR->>Proc : Run inspection
Proc-->>LR : _ModelInfo
LR->>FS : Write cache
LR-->>Reg : _ModelInfo
end
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L622-L711)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L622-L711)
- [registry.py](file://vllm/model_executor/models/registry.py#L718-L743)

### Attention Backend Selection and Model Registration
- Attention backend selection depends on dtype, head size, KV cache dtype, block size, and attention type.
- Platform determines backend class and may adjust KV cache layout.
- Registration indirectly influences backend choice via model’s attn_type and attention-free/hybrid flags.

```mermaid
flowchart TD
A["ModelConfig resolved arch"] --> B["Get attn_type from _ModelInfo"]
B --> C["get_attn_backend(head_size, dtype, kv_cache_dtype,<br/>block_size, use_mla, has_sink, use_sparse,<br/>use_mm_prefix, attn_type)"]
C --> D["Platform.get_attn_backend_cls(...)"]
D --> E["Resolve backend class"]
E --> F["Adjust KV cache layout if required"]
```

**Diagram sources**
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L575)

**Section sources**
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L575)

### Quantization Compatibility and Model Registration
- Quantization backends choose methods based on platform and packing format.
- KV cache scales are initialized for attention layers to support quantized KV caches.
- BitBLAS availability and minimum versions are validated before use.

```mermaid
flowchart TD
QStart["QuantizationConfig"] --> QSel["Select quant method by platform/backend/format"]
QSel --> KVC["Create KV cache scales on attention layers"]
KVC --> QRun["Apply quantized kernels during forward"]
```

**Diagram sources**
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)

**Section sources**
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)

### Distributed Inference Capabilities
- Pipeline parallel support is indicated by supports_pp and verified by checking forward signature for intermediate_tensors.
- Tests validate PP support for specific architectures and ensure CUDA is not initialized during inspection.

```mermaid
flowchart TD
PPStart["supports_pp(model)"] --> CheckAttrs["Check class has make_empty_intermediate_tensors"]
CheckAttrs --> CheckForward["Check forward signature includes intermediate_tensors"]
CheckForward --> PPRes{"Both satisfied?"}
PPRes --> |Yes| PPEnable["Pipeline parallel enabled"]
PPRes --> |No| PPWarn["Warning: mismatched attributes/signature"]
```

**Diagram sources**
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L440-L510)
- [test_registry.py](file://tests/models/test_registry.py#L83-L110)

**Section sources**
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L440-L510)
- [test_registry.py](file://tests/models/test_registry.py#L83-L110)

### Practical Examples

#### Example 1: Registering a Custom Transformer Model
Steps:
- Define a PyTorch nn.Module subclass implementing the vLLM model contract (init with vllm_config, embed_input_ids, forward with input_ids and positions).
- Optionally implement capability flags (e.g., supports_multimodal, supports_pp).
- Register the model via the registry API with a model_arch string and either the class or "<module>:<class>" string.

References:
- [__init__.py](file://vllm/model_executor/models/__init__.py#L1-L44)
- [registry.py](file://vllm/model_executor/models/registry.py#L753-L798)

#### Example 2: Integrating a Multimodal Architecture
Steps:
- Implement SupportsMultiModal and related flags.
- Provide get_language_model and embed_input_ids to merge text and multimodal embeddings.
- Register the model and rely on multimodal limits and processors.

References:
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [registry.py](file://vllm/model_executor/models/registry.py#L753-L798)

#### Example 3: Specialized Inference (Pooling/Sequence Classification)
Steps:
- Implement VllmModelForPooling and set default_pooling_type and attn_type.
- Register the model; ModelConfig will convert it appropriately for pooling runners.

References:
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L145-L198)
- [model.py](file://vllm/config/model.py#L494-L519)

#### Example 4: Using Quantized Models
Steps:
- Choose a quantization backend compatible with your platform.
- Ensure KV cache scales are initialized for attention layers if needed.
- Validate BitBLAS version if using BitBLAS-backed quantization.

References:
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)
- [README.md](file://examples/offline_inference/basic/README.md#L53-L81)

## Dependency Analysis
- Coupling:
  - Registry depends on interfaces for capability detection and on platform for device capability checks.
  - Attention backend selection depends on platform-provided backend classes and configuration.
  - Quantization backends depend on platform and optional libraries (e.g., BitBLAS).
- Cohesion:
  - Capability flags are centralized in interfaces.py and interfaces_base.py.
  - Lazy loading and caching are encapsulated in _LazyRegisteredModel.
- External dependencies:
  - Subprocess-based inspection avoids CUDA initialization in the parent process.
  - Platform abstraction ensures backend selection adapts to device capabilities.

```mermaid
graph TB
Reg["registry.py"] --> IF["interfaces.py"]
Reg --> IFB["interfaces_base.py"]
Reg --> Plat["platform_interface.py"]
Reg --> Cfg["model.py"]
Reg --> AttSel["selector.py"]
AttSel --> Plat
Reg --> QAR["auto_round.py"]
Reg --> QKC["kv_cache.py"]
Reg --> QB["bitblas.py"]
Reg --> QG["gptq_bitblas.py"]
```

**Diagram sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L748)
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L44-L198)
- [platform_interface.py](file://vllm/platforms/interface.py#L279-L323)
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)
- [model.py](file://vllm/config/model.py#L494-L519)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L527-L748)
- [interfaces.py](file://vllm/model_executor/models/interfaces.py#L72-L215)
- [interfaces_base.py](file://vllm/model_executor/models/interfaces_base.py#L44-L198)
- [platform_interface.py](file://vllm/platforms/interface.py#L279-L323)
- [selector.py](file://vllm/attention/selector.py#L46-L118)
- [auto_round.py](file://vllm/model_executor/layers/quantization/auto_round.py#L438-L454)
- [kv_cache.py](file://vllm/model_executor/layers/quantization/kv_cache.py#L27-L53)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)
- [model.py](file://vllm/config/model.py#L494-L519)

## Performance Considerations
- Lazy loading and caching reduce repeated inspection costs and prevent CUDA initialization overhead in the parent process.
- LRU caching of inspection and loading results minimizes redundant work.
- Attention backend selection is cached to avoid repeated resolution.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
Common issues and debugging techniques:
- Unsupported or previously supported architectures:
  - The registry raises explicit errors for unsupported or deprecated architectures, guiding users to older versions or updates.
- CUDA initialization during import:
  - LazyRegisteredModel inspects models in a subprocess to avoid CUDA initialization in the parent process.
- Pipeline parallel mismatches:
  - Tests validate supports_pp and forward signature; warnings are logged when attributes/signature mismatch.
- Quantization backend errors:
  - BitBLAS version checks and platform-specific backends surface clear errors when incompatible.

References:
- [registry.py](file://vllm/model_executor/models/registry.py#L799-L822)
- [registry.py](file://vllm/model_executor/models/registry.py#L676-L711)
- [test_registry.py](file://tests/models/test_registry.py#L83-L110)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)

**Section sources**
- [registry.py](file://vllm/model_executor/models/registry.py#L799-L822)
- [registry.py](file://vllm/model_executor/models/registry.py#L676-L711)
- [test_registry.py](file://tests/models/test_registry.py#L83-L110)
- [bitblas.py](file://vllm/model_executor/layers/quantization/bitblas.py#L45-L84)
- [gptq_bitblas.py](file://vllm/model_executor/layers/quantization/gptq_bitblas.py#L80-L107)

## Conclusion
vLLM’s model registration system provides a robust framework for integrating custom architectures. By adhering to capability flags and interfaces, leveraging lazy loading and caching, and aligning with platform and attention backend selection, developers can seamlessly add new models. Quantization compatibility and distributed inference features are naturally supported through the registry’s inspection and platform abstractions.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Best Practices for Model Integration
- Implement required model interfaces and capability flags explicitly.
- Use lazy registration (<module>:<class>) to avoid CUDA initialization during import.
- Keep model inspection lightweight; rely on cached _ModelInfo.
- Validate platform compatibility before loading.
- Ensure attention type and pooling defaults are set appropriately for your model.

[No sources needed since this section provides general guidance]