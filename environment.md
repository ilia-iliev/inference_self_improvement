# RL Environment: Inference Throughput Optimization

## Overview

Given a dual-3090 system with Gemma 4 2B pre-downloaded, the agent must maximize inference throughput without sacrificing output quality. The agent's solution is scored against a vLLM baseline on a held-out prompt set, with greedy exact-match verification to ensure no quality degradation.

## Hardware

- 2x NVIDIA RTX 3090 (48GB VRAM total)
- Gemma 4 2B (pre-downloaded)

## Prompt Data

Two prompt sets, drawn from the same distribution:

1. **Development set** (`/data/dev_prompts.jsonl`, 100 prompts) — provided to the agent for testing and profiling. Reference greedy outputs included in `/data/dev_reference.jsonl`.
2. **Held-out eval set** (`/judge/eval_prompts.jsonl`, 100 prompts) — only accessed by the judge at scoring time. Prevents reward hacking via cached/hardcoded outputs.

Prompt distribution:
- Mixed input context lengths (short and long)
- Mixed tasks (Q&A, summarization, code generation, reasoning)
- Fully deterministic (no randomness in prompts)
- Includes modalities like audio and images
- Source: curated from ShareGPT or OpenAssistant (TBD)

## Agent Task

### Prompt

> You have access to a system with dual NVIDIA RTX 3090 GPUs and a pre-downloaded Gemma 4 2B model at `/models/gemma4-2b/`.
>
> Your task is to maximize inference throughput on this hardware. You will be evaluated on how quickly you can generate greedy completions for a set of prompts, compared to an optimized vLLM baseline.
>
> **Constraints:**
> - All outputs must be greedy-decoded and token-for-token identical to single-request HuggingFace greedy reference outputs. Any deviation results in a score of 0.
> - You may use any approach: custom kernels, batching strategies, parallelism, torch.compile, CUDA graphs, etc.
> - vLLM is pre-installed on the system.
>
> **Provided resources:**
> - Development prompts: `/data/dev_prompts.jsonl`
> - Reference greedy outputs for dev prompts: `/data/dev_reference.jsonl`
> - Model weights: `/models/gemma4-2b/`
>
> **Deliverable:**
> When you are done, create a script `/solution/run.py` that:
> 1. Reads a JSONL file path as its first command-line argument
> 2. Each line of the JSONL contains `{"prompt": "..."}` 
> 3. Generates greedy completions (max 256 tokens each) for all prompts
> 4. Writes results to `/solution/outputs.jsonl`, each line containing `{"prompt": "...", "completion": "..."}`
>
> Your solution will be evaluated on a held-out prompt set not available to you during development.

## Judge

### Procedure

1. Generate reference outputs for the held-out eval set using single-request HuggingFace greedy decoding (slow but canonical).
2. Run the vLLM baseline on the eval set. Record wall-clock time.
3. Run the agent's `/solution/run.py` on the eval set. Record wall-clock time.
4. Verify every output is token-for-token identical to the HuggingFace reference.
5. Compute score.

### Scoring

```
if any output does not match reference:
    score = 0
elif agent_time >= vllm_time:
    score = 0
else:
    score = vllm_time / agent_time   # continuous, >1.0 means faster than vLLM
```

Score is continuous. A score of 0 means the agent failed (either wrong outputs or slower than baseline). A score of 1.5 means the agent is 1.5x faster than vLLM.

### Baseline (vLLM)

vLLM serves Gemma 4 2B on the available GPUs with default settings (no speculative decoding, as no smaller draft model exists). Processes all eval prompts via the standard API. This is already a strong baseline: PagedAttention, CUDA graphs, continuous batching.

## Why This Is Hard

- The baseline (vLLM) is already highly optimized for this exact use case.
- Greedy exact match rules out lossy optimizations (quantization, approximate attention, pruning).
- 2B model is small enough that compute isn't the bottleneck — memory bandwidth and scheduling dominate.
- Dual GPUs add complexity: tensor parallelism likely hurts for a 2B model (communication overhead > compute savings), so the agent must find smarter strategies.
- Held-out eval set prevents memorization or prompt-specific tricks.

## Possible Agent Strategies

- Dual replicas (one per GPU) with intelligent routing by prompt length
- Dynamic batching to minimize padding
- Custom Triton kernels tuned for Gemma 4's architecture
- torch.compile with optimal backend selection
- CUDA graph capture for fixed-shape workloads
- KV cache memory management to maximize batch size
- Hybrid approaches combining multiple techniques
