Harness for the moe-glue changes (branch `moe-glue`): the decode step around and including the experts, and the
prompt chunk's glue around the expert GEMMs, old vs new on the same inputs, compared **bitwise**, then timed.

- `baseline/` is a byte-for-byte copy of `src/gemma/kernels.cu`, `src/kernels/cuda/native_mmvq.cu` and their headers
  at gemma 8c7651c. `old_k.cu` / `old_mmvq.cu` compile them into the namespaces `strata::gemma::old_k` and
  `strata::old_kernels` (a macro renames the namespace), so the old kernels run in the same process as the new ones.
  `glue_bench.cu`'s `old_layer_small` is the old `Engine::layer_small` verbatim; the new step calls
  `strata::gemma::decode_layer`, the function the engine itself calls.
- Real weights of one sliding and one global layer are read from the GGUF (`common.hpp`; 32 or 64 experts kept,
  the global layer then shares the sliding layer's experts, to stay under ~400 MiB of VRAM); activations are synthetic.

```
./build.sh                                                       # nvcc, ~1 min (on the VPS under the build lock)
M=/root/models/c26-drag-q8dense.gguf
flock /root/sg-tools/gpu.lock ./glue_bench $M decode check 1 64  # one decode step (30 layers), bitwise: x, the head's
                                                                 # input, its q8_1, both KV caches (GLUE_SEED, GLUE_POS)
./prof.sh dec $M decode nsys 1 32 40                             # old / new / new+fork step graphs under nsys
flock /root/sg-tools/gpu.lock ./glue_bench $M mmvq 0             # each product alone (+ the merged q/k/v, gate/up)
flock /root/sg-tools/gpu.lock ./glue_bench $M router 3000 0      # the decode router on the real 96-row router, + ties
flock /root/sg-tools/gpu.lock ./glue_bench $M prompt 692 masses_l0.txt   # prompt glue (or 1787 / uniform), bitwise
./prof.sh p692 $M prompt 692 masses_l0.txt prof                  # the same as graphs under nsys
```

`mmq_order` (`./build.sh mmq_order`, links the engine build's `libstrata_mmq.a`) runs the old expert sort twice on the
same ids - its atomics place most rows differently each time - and the real MMQ gate_up of one layer on both orders:
0 of 5536 (692 rows) and 0 of 14296 (1787 rows) (token, slot) rows differ, 6 trials each, so the prompt path's
results do not depend on the sort's order (the old code is run-to-run deterministic there).

`masses_l*.txt` (from `routing_masses.py` and `/root/sg-tools/routing-26b.json`) give the skewed per-expert routing:
logits = log(mass) + Gumbel noise. `prof.sh` runs `glue_bench` under `nsys --cuda-graph-trace=node` and
`nsys_graphs.py` prints per graph the replay span and kernel time (min / p25 / median; `KERNELS=--kernels` per kernel).

**Timing on the shared card.** Production keeps the A4000 busy: event-timed loops come out ~2x and noisy. The
numbers that count are CUPTI kernel durations from nsys, replays interleaved old/new one at a time, taken as the
minimum (the replays no production time slice hit).
