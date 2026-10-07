#!/bin/bash
# tools/gemma/glue/build.sh - build glue_bench (run from anywhere; on the VPS under the shared build lock).
set -e
cd "$(dirname "$0")"
R=../../..
export PATH=/usr/local/cuda/bin:$PATH
F="-O3 -arch=sm_86 -use_fast_math -extended-lambda -std=c++17 -lineinfo"
mkdir -p obj
nvcc $F -I. -Ibaseline/include -I$R/include -c old_k.cu -o obj/old_k.o &
nvcc $F -I. -Ibaseline/include -I$R/include -c old_mmvq.cu -o obj/old_mmvq.o &
nvcc $F -I. -I$R/include -I$R/third_party/ggml -c $R/src/gemma/kernels.cu -o obj/kernels.o &
nvcc $F -I. -I$R/include -I$R/third_party/ggml -c $R/src/kernels/cuda/native_mmvq.cu -o obj/native_mmvq.o &
wait
nvcc $F -I. -I$R/include -I$R/third_party/ggml -c $R/src/kernels/cuda/iq_kernels.cu -o obj/iq_kernels.o &
EXTRA=""
for f in $R/src/gemma/decode.cpp; do
  if [ -f "$f" ]; then nvcc $F -I. -I$R/include -x cu -c "$f" -o obj/decode.o & EXTRA="$EXTRA obj/decode.o"; fi
done
nvcc $F -I. -I$R/include -c glue_bench.cu -o obj/glue_bench.o &
wait
nvcc $F obj/glue_bench.o obj/old_k.o obj/old_mmvq.o obj/kernels.o obj/native_mmvq.o obj/iq_kernels.o $EXTRA -o glue_bench
echo built glue_bench
