# strata-gemma

A Gemma 4 inference engine for one NVIDIA GPU, forked from [Strata](https://github.com/Niko1221/Strata) and rebuilt
around Gemma 4 26B-A4B (MoE) with vision. Strata runs one model (Qwen3.8-Flash-Next) with RAM offload and its own
GDN / sparse-attention kernels. This fork drops all of that and keeps Strata's way of working: a fixed model, its own
CUDA kernels and graphs instead of a general graph runtime, and every kernel checked against llama.cpp.

Built for one workload: hCaptcha-style visual puzzles (1–3 images, a short JSON answer under a schema). It speaks the
part of llama-server's HTTP API that such a client uses, so the client switches by URL.

## What is in it

| Part | Where | Notes |
|---|---|---|
| Weights | `src/gemma/model.cpp` | A llama.cpp GGUF (`gemma4`) in one device arena, still in its GGUF encoding. Hyperparameters come from the metadata, so a pruned file (fewer experts) or a mixed-quant file loads as is. |
| Prompt path | `src/gemma/engine.cpp` (`layer_big`) | Quantized products through llama.cpp's MMQ kernels (compiled in, with ggml-cuda's host symbols stubbed, as in Strata). MoE: (token, slot) pairs sorted by expert, one MMQ launch per projection. Attention: cuBLAS f16 GEMMs and a masked softmax. Images are non-causal on the sliding layers, as in llama.cpp. |
| Decode path | `src/gemma/engine.cpp` (`layer_small`), `src/gemma/kernels.cu` | Up to 8 rows, one CUDA graph per row count. native_mmvq (Strata's transcription of ggml's MMVQ) for the projections. Expert-indexed MMVQ (`native_mmvq_id`), and expert-grouped for verify windows (an expert several tokens chose is read once). Flash-decoding attention. Fused norm / residual / q8_1-quantize kernels: about 14 launches a layer. |
| Speculative decoding | `src/gemma/mtp.cpp` | Google's Gemma 4 assistant (MTP drafter, `gemma4-assistant` GGUF). A 4-layer model that reads the target's KV cache. Drafts run as one captured graph; the target verifies a window of up to 8 rows. Greedy, so the output is the target's own. |
| Vision | `src/gemma/vision.cu`, `src/gemma/vision_gemm.cuh`, `src/gemma/image.cpp` | The `gemma4v` encoder natively. Matrices in BF16 via cuBLAS; GEMMs of at most 13e9 multiply-adds take the cuBLASLt algorithm timed fastest for their shape at load (bigger ones keep cuBLAS's default: under the A4000's power cap it is the faster one, `tools/gemma/vis_gemm_bench.cu`). Attention in an own FlashAttention-2 kernel (mma.sync, head_dim 72 padded to 80); llama.cpp falls back to a generic tile kernel there. Resize is bit-for-bit mtmd's Pillow-compatible bicubic. |
| Server | `src/gemma/server_main.cpp` | `/completion` (raw prompt + `multimodal_data`, `json_schema` or GBNF `grammar`), `/props`, `/health`. Bearer key. Tokenizer, grammar sampler and `json_schema_to_grammar` from a CPU-only libllama. |

Strata files still used: `include/strata/artifact/gguf_reader.hpp`, `src/kernels/cuda/native_mmvq.cu` (+ the expert
kernels added here), `iq_kernels.cu`, `dequant_bf16.cu`, and the MMQ glue (`src/gemma/mmq.cu`,
`src/gemma/ggml_cuda_host.cu`, adapted from `src/prefill/moe_mmq.cu`). Everything Qwen-specific is gone; it is in git
history and in the upstream repository.

## Build

```bash
cmake -B build -DSTRATA_LLAMA_DIR=/path/to/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

`STRATA_LLAMA_DIR` is a llama.cpp checkout. It was tested with 3cf03257 (2026-09-20), the commit Strata pins. It is
built CPU-only; ggml-cuda is never linked.

## Run

```bash
build/strata-gemma-server -m gemma-4-26B-e96-q4_0.gguf --mmproj gemma-4-26B-it-mmproj.gguf \
    --mtp mtp-gemma-4-26B-A4B-it.gguf --draft 3 --ctx 2048 --host 0.0.0.0 --port 8092 --api-key-file key.txt
```

Tools:

- `strata-gemma-parity -m M --ref R [--split N] [--bench]` replays a llama.cpp logits dump made by
  `tools/gemma/ref_logits` and times verify windows.
- `strata-gemma-vision --mmproj M --image I --ref E` compares image embeddings with a llama.cpp dump made by
  `tools/gemma/vis_bench --dump`.

## Measured (RTX A4000 16 GB, Gemma 4 26B-A4B pruned to 96 experts, QAT q4_0)

**Parity with llama.cpp:**

- Same greedy tokens on text prompts (46/47 steps).
- The logit differences are the size of llama.cpp's own flash-attention vs non-flash-attention difference.
- Image embeddings: cosine 0.998–0.9995 against mtmd.

**Per-page latency on real puzzle pages** (server side; the same model, the same pages):

| | llama.cpp (llama-server) | strata-gemma |
|---|---|---|
| binary (1 image, ~380 tokens) | 0.93 s | 0.36 s |
| area (1 image, ~440 tokens) | 1.5 s | 0.55 s |
| drag (3 images, ~1060 tokens) | 2.3 s | 1.0 s |

Where the time went, and what changed:

- **Image encoder:** 106 ms against ~430 ms in llama-server.
- **Prompt:** ~3,500 tokens/s.
- **Decode:** 9.4 ms a token, ~7.8 ms a token with MTP (67–87% of drafts accepted).
- **Accuracy:** the same as llama.cpp on 120 held-out tasks (same keep counts on binary and area, 4/13 vs 2/13 on drag).
