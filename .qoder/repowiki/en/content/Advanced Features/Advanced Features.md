# Advanced Features

<cite>
**Referenced Files in This Document**
- [speculative.py](file://vllm/config/speculative.py)
- [spec_decode.md](file://docs/features/spec_decode.md)
- [prefix_caching.md](file://docs/design/prefix_caching.md)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py)
- [lora.md](file://docs/features/lora.md)
- [lora_model.py](file://vllm/lora/lora_model.py)
- [request.py](file://vllm/lora/request.py)
- [benchmark_prefix_caching.py](file://benchmarks/benchmark_prefix_caching.py)
- [benchmark_ngram_proposer.py](file://benchmarks/benchmark_ngram_proposer.py)
- [benchmark_lora.py](file://benchmarks/kernels/benchmark_lora.py)
- [multilora_inference.py](file://examples/offline_inference/multilora_inference.py)
- [prefix_caching.py](file://examples/offline_inference/prefix_caching.py)
- [automatic_prefix_caching.py](file://examples/offline_inference/automatic_prefix_caching.py)
- [spec_decode.py](file://examples/offline_inference/spec_decode.py)
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
This document covers vLLM’s advanced features: speculative decoding, prefix caching, and LoRA adapters. It explains implementation details, configuration options, performance characteristics, memory optimization strategies, and practical integration patterns. It also highlights trade-offs, optimal use cases, and troubleshooting guidance grounded in the repository’s documentation and source files.

## Project Structure
The advanced features span configuration models, attention kernels, documentation, examples, and benchmarks:
- Speculative decoding: configuration model, CLI usage, and end-to-end examples
- Prefix caching: design documentation, attention kernels for prefix prefill, and examples
- LoRA adapters: feature docs, model loader, request model, and examples

```mermaid
graph TB
subgraph "Speculative Decoding"
SD_Config["SpeculativeConfig<br/>(vllm/config/speculative.py)"]
SD_Docs["Speculative Docs<br/>(docs/features/spec_decode.md)"]
SD_Examples["Examples<br/>(examples/offline_inference/spec_decode.py)"]
end
subgraph "Prefix Caching"
PC_Design["Design Doc<br/>(docs/design/prefix_caching.md)"]
PC_Kernel["Prefix Prefill Kernel<br/>(vllm/attention/ops/prefix_prefill.py)"]
PC_Bench["Benchmark<br/>(benchmarks/benchmark_prefix_caching.py)"]
PC_Examples["Examples<br/>(examples/offline_inference/prefix_caching.py,<br/>examples/offline_inference/automatic_prefix_caching.py)"]
end
subgraph "LoRA Adapters"
LA_Docs["LoRA Docs<br/>(docs/features/lora.md)"]
LA_Model["LoRAModel Loader<br/>(vllm/lora/lora_model.py)"]
LA_Request["LoRARequest<br/>(vllm/lora/request.py)"]
LA_Bench["Benchmark<br/>(benchmarks/kernels/benchmark_lora.py)"]
LA_Examples["Examples<br/>(examples/offline_inference/multilora_inference.py)"]
end
SD_Config --> SD_Docs
SD_Config --> SD_Examples
PC_Design --> PC_Kernel
PC_Design --> PC_Bench
PC_Design --> PC_Examples
LA_Docs --> LA_Model
LA_Docs --> LA_Request
LA_Docs --> LA_Bench
LA_Docs --> LA_Examples
```

**Diagram sources**
- [speculative.py](file://vllm/config/speculative.py#L1-L645)
- [spec_decode.md](file://docs/features/spec_decode.md#L1-L333)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L1-L232)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L1-L815)
- [lora.md](file://docs/features/lora.md#L1-L370)
- [lora_model.py](file://vllm/lora/lora_model.py#L1-L247)
- [request.py](file://vllm/lora/request.py#L1-L96)

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L1-L645)
- [spec_decode.md](file://docs/features/spec_decode.md#L1-L333)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L1-L232)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L1-L815)
- [lora.md](file://docs/features/lora.md#L1-L370)
- [lora_model.py](file://vllm/lora/lora_model.py#L1-L247)
- [request.py](file://vllm/lora/request.py#L1-L96)

## Core Components
- SpeculativeConfig: central configuration for speculative decoding, including method selection, draft model parameters, and advanced controls
- Prefix caching design: hashing scheme, block lifecycle, and cache isolation
- LoRA model loader and request model: adapter loading, validation, and runtime request representation

Key implementation anchors:
- SpeculativeConfig fields and validators define supported methods, draft model constraints, and runtime safeguards
- Prefix caching design documents hashing, eviction, and multi-modal handling
- LoRA loader validates modules and constructs adapter weights

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L52-L167)
- [speculative.py](file://vllm/config/speculative.py#L235-L462)
- [speculative.py](file://vllm/config/speculative.py#L464-L645)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L1-L232)
- [lora_model.py](file://vllm/lora/lora_model.py#L66-L111)
- [lora_model.py](file://vllm/lora/lora_model.py#L112-L247)
- [request.py](file://vllm/lora/request.py#L9-L96)

## Architecture Overview
The advanced features integrate with the engine and attention subsystems:
- Speculative decoding: configuration drives draft model instantiation and method-specific logic; attention kernels adapt to prefix-only prefill
- Prefix caching: KV cache manager computes block hashes and manages allocation/eviction; attention kernels support efficient prefix prefill
- LoRA adapters: model loader parses and validates adapter tensors; request model carries adapter identity and path

```mermaid
graph TB
Engine["Engine"]
SD["SpeculativeConfig"]
PC["Prefix Caching Manager"]
LA["LoRA Manager"]
Attn["Attention Ops"]
Engine --> SD
Engine --> PC
Engine --> LA
PC --> Attn
SD --> Attn
```

**Diagram sources**
- [speculative.py](file://vllm/config/speculative.py#L52-L167)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L97-L131)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L618-L815)
- [lora_model.py](file://vllm/lora/lora_model.py#L66-L111)

## Detailed Component Analysis

### Speculative Decoding
Speculative decoding accelerates memory-bound inference by generating candidate tokens ahead of acceptance. vLLM supports multiple proposal strategies and draft models.

Implementation highlights:
- Configuration model defines method selection, draft model parameters, and advanced toggles (e.g., disabling by batch size, padded drafter batching)
- Method detection supports n-gram, suffix decoding, Medusa, MLP speculator, EAGLE, and MTP variants
- Validation ensures compatibility with tensor parallel sizes and model types
- Suffix decoding requires an external dependency and exposes tunable parameters for tree depth, cache size, and token probability thresholds

Performance and trade-offs:
- Inter-token latency improvements depend on dataset and sampling parameters; ongoing optimization is noted in the documentation
- Pipeline parallelism is currently incompatible with speculative decoding
- Draft models often run without tensor parallelism to achieve speedups despite smaller size constraints

Practical configuration examples:
- Offline and online usage with various methods and draft models
- Suffix decoding with dynamic speculation and recommended defaults
- EAGLE-based speculators with method-specific constraints

```mermaid
classDiagram
class SpeculativeConfig {
+bool enforce_eager
+int num_speculative_tokens
+str model
+str method
+int draft_tensor_parallel_size
+str quantization
+int max_model_len
+str revision
+int disable_by_batch_size
+bool disable_padded_drafter_batch
+int prompt_lookup_max
+int prompt_lookup_min
+str speculative_token_tree
+int suffix_decoding_max_tree_depth
+int suffix_decoding_max_cached_requests
+float suffix_decoding_max_spec_factor
+float suffix_decoding_min_token_prob
+compute_hash() str
+use_eagle() bool
}
```

**Diagram sources**
- [speculative.py](file://vllm/config/speculative.py#L52-L167)
- [speculative.py](file://vllm/config/speculative.py#L168-L234)
- [speculative.py](file://vllm/config/speculative.py#L235-L462)
- [speculative.py](file://vllm/config/speculative.py#L464-L645)

```mermaid
sequenceDiagram
participant User as "User"
participant CLI as "vLLM CLI"
participant Engine as "Engine"
participant SD as "SpeculativeConfig"
participant Attn as "Attention Ops"
User->>CLI : "--speculative_config ..."
CLI->>Engine : Initialize with SpeculativeConfig
Engine->>SD : Validate and normalize
Engine->>Attn : Use prefix prefill kernels for proposal phase
Attn-->>Engine : Proposal outputs
Engine-->>CLI : Final accepted tokens
```

**Diagram sources**
- [spec_decode.md](file://docs/features/spec_decode.md#L1-L172)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L618-L815)
- [speculative.py](file://vllm/config/speculative.py#L235-L462)

Configuration and usage references:
- Offline and online examples for n-gram, suffix decoding, MLP speculators, and EAGLE
- Compatibility notes and limitations

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L52-L167)
- [speculative.py](file://vllm/config/speculative.py#L235-L462)
- [speculative.py](file://vllm/config/speculative.py#L464-L645)
- [spec_decode.md](file://docs/features/spec_decode.md#L1-L333)
- [spec_decode.py](file://examples/offline_inference/spec_decode.py#L1-L200)

### Prefix Caching
Prefix caching reduces redundant prompt computations by reusing KV-cache blocks across requests with shared prefixes. vLLM uses a hash-based scheme to uniquely identify blocks and supports multi-modal inputs and cache isolation.

Key mechanisms:
- Hash composition includes parent hash, block tokens, and extras (e.g., LoRA IDs, multi-modality hashes, cache salts)
- Block lifecycle: allocate, touch (when reused), append, cache when full, free, and LRU eviction
- Attention kernels support prefix-only prefill to accelerate cached segments

Memory optimization and throughput:
- Full blocks are cached; duplicates are resolved upon freeing
- LRU eviction prioritizes blocks least likely to be reused
- Multi-modal hashing ensures correctness across heterogeneous inputs

```mermaid
flowchart TD
Start(["New Request"]) --> Hash["Compute prefix hash<br/>and lookup cached blocks"]
Hash --> HasCache{"Cache hit blocks?"}
HasCache --> |Yes| Touch["Touch hits (inc refcnt, remove from free)"]
HasCache --> |No| AllocNew["Allocate new blocks from free queue"]
Touch --> Slots["Append tokens to existing and new blocks"]
AllocNew --> Slots
Slots --> Full{"Any full block?"}
Full --> |Yes| Cache["Add to cache blocks"]
Full --> |No| Continue["Continue decoding"]
Cache --> Continue
Continue --> Done(["Request finished"])
Done --> Free["Free blocks (reverse order)<br/>and evict LRU cached blocks if needed"]
```

**Diagram sources**
- [prefix_caching.md](file://docs/design/prefix_caching.md#L132-L205)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L206-L232)

```mermaid
sequenceDiagram
participant Req as "Request"
participant KV as "KV Cache Manager"
participant Kern as "Prefix Prefill Kernel"
Req->>KV : get_computed_blocks()
KV-->>Req : Cached block IDs
Req->>KV : allocate_slots()
KV-->>Req : Block table with hits/tail
Req->>Kern : context_attention_fwd(prefix-only)
Kern-->>Req : Cached segment attention
Req->>KV : free on completion
KV-->>Req : Evict LRU cached blocks if needed
```

**Diagram sources**
- [prefix_caching.md](file://docs/design/prefix_caching.md#L132-L205)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L618-L815)

Practical examples:
- Manual and automatic prefix caching usage
- Benchmark scripts for measuring caching effects

**Section sources**
- [prefix_caching.md](file://docs/design/prefix_caching.md#L1-L232)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L1-L815)
- [prefix_caching.py](file://examples/offline_inference/prefix_caching.py#L1-L200)
- [automatic_prefix_caching.py](file://examples/offline_inference/automatic_prefix_caching.py#L1-L200)
- [benchmark_prefix_caching.py](file://benchmarks/benchmark_prefix_caching.py#L1-L200)

### LoRA Adapters
LoRA enables parameter-efficient fine-tuning and dynamic adapter switching. vLLM supports single and multi-adapter serving, runtime loading/unloading, and multimodal default adapters.

Implementation:
- LoRAModel loads and validates adapter tensors, mapping PEFT keys to model modules
- LoRARequest carries adapter identity, path, and optional base model linkage
- Feature docs describe server-side configuration, runtime updates, and default multimodal adapters

Performance and memory tuning:
- max_lora_rank should match the maximum rank among adapters to avoid unnecessary allocations
- Tensorizer support reduces I/O overhead for large adapters

```mermaid
classDiagram
class LoRAModel {
+int id
+int rank
+dict~str, LoRALayerWeights~ loras
+clone(lora_model_id) LoRAModel
+get_lora(module_name) LoRALayerWeights
+check_lora_name(name) bool
+from_lora_tensors(...)
+from_local_checkpoint(...)
}
class LoRARequest {
+string lora_name
+int lora_int_id
+string lora_path
+string lora_local_path
+int long_lora_max_len
+string base_model_name
+dict tensorizer_config_dict
+__eq__, __hash__
}
```

**Diagram sources**
- [lora_model.py](file://vllm/lora/lora_model.py#L25-L111)
- [lora_model.py](file://vllm/lora/lora_model.py#L112-L247)
- [request.py](file://vllm/lora/request.py#L9-L96)

Operational patterns:
- Offline generation with per-request adapters
- Server-side adapter modules and runtime load/unload APIs
- Default multimodal adapters for automatic application

**Section sources**
- [lora_model.py](file://vllm/lora/lora_model.py#L66-L111)
- [lora_model.py](file://vllm/lora/lora_model.py#L112-L247)
- [request.py](file://vllm/lora/request.py#L9-L96)
- [lora.md](file://docs/features/lora.md#L1-L370)
- [multilora_inference.py](file://examples/offline_inference/multilora_inference.py#L1-L200)
- [benchmark_lora.py](file://benchmarks/kernels/benchmark_lora.py#L1-L200)

## Dependency Analysis
- Speculative decoding depends on attention kernels for prefix prefill and draft model configuration
- Prefix caching relies on KV cache manager and attention kernels for efficient reuse
- LoRA depends on model loader and request model for adapter resolution and validation

```mermaid
graph TB
SD["SpeculativeConfig"] --> Attn["prefix_prefill.py"]
PC["prefix_caching.md"] --> Attn
PC --> Attn
LA["lora_model.py"] --> LAReq["request.py"]
```

**Diagram sources**
- [speculative.py](file://vllm/config/speculative.py#L235-L462)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L97-L131)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L618-L815)
- [lora_model.py](file://vllm/lora/lora_model.py#L66-L111)
- [request.py](file://vllm/lora/request.py#L9-L96)

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L235-L462)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L97-L131)
- [prefix_prefill.py](file://vllm/attention/ops/prefix_prefill.py#L618-L815)
- [lora_model.py](file://vllm/lora/lora_model.py#L66-L111)
- [request.py](file://vllm/lora/request.py#L9-L96)

## Performance Considerations
- Speculative decoding
  - Inter-token latency gains vary by workload; pipeline parallelism is incompatible
  - Draft model TP size constraints and method-specific validations affect throughput
- Prefix caching
  - Hashing overhead is low; focus on block reuse and LRU eviction policies
  - Multi-modal hashing adds deterministic cache differentiation
- LoRA
  - Rank selection impacts memory footprint; align max_lora_rank with deployed adapters
  - Tensorizer can reduce load times for large adapters

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Speculative decoding
  - Ensure method and draft model compatibility; validate tensor parallel sizes
  - For suffix decoding, confirm the external dependency is installed
  - Disable speculation under high backlog using the batch-size threshold option
- Prefix caching
  - Verify block hashing and multi-modal inputs; use cache salts for isolation
  - Monitor eviction behavior and adjust block size or cache capacity
- LoRA
  - Validate target modules and ranks; ensure vocabulary consistency for embedding LoRA
  - Use runtime load/unload APIs carefully and confirm adapter paths

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L464-L645)
- [prefix_caching.md](file://docs/design/prefix_caching.md#L81-L105)
- [lora_model.py](file://vllm/lora/lora_model.py#L148-L207)

## Conclusion
vLLM’s advanced features—speculative decoding, prefix caching, and LoRA adapters—offer substantial performance and operational benefits. Speculative decoding accelerates inference with multiple proposal strategies; prefix caching minimizes redundant computations through robust hashing and efficient kernels; LoRA enables flexible, parameter-efficient adaptation with dynamic management. Proper configuration, awareness of trade-offs, and alignment with workload characteristics are essential for optimal outcomes.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices
- Practical examples and benchmarks are available under examples and benchmarks directories for hands-on experimentation.

**Section sources**
- [spec_decode.py](file://examples/offline_inference/spec_decode.py#L1-L200)
- [prefix_caching.py](file://examples/offline_inference/prefix_caching.py#L1-L200)
- [automatic_prefix_caching.py](file://examples/offline_inference/automatic_prefix_caching.py#L1-L200)
- [multilora_inference.py](file://examples/offline_inference/multilora_inference.py#L1-L200)
- [benchmark_prefix_caching.py](file://benchmarks/benchmark_prefix_caching.py#L1-L200)
- [benchmark_ngram_proposer.py](file://benchmarks/benchmark_ngram_proposer.py#L1-L200)
- [benchmark_lora.py](file://benchmarks/kernels/benchmark_lora.py#L1-L200)