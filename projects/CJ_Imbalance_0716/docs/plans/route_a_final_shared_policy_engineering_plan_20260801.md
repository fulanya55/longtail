# Final shared-policy engineering benchmark plan

## Scope

Benchmark the complete frozen inference path, excluding disk I/O and model
loading: Focal-36 backbone, H1 head, RawSaw, ISRJ periodic expert and fusion,
MFT matched-filter guard, VFT Ridge-STFT expert and fusion, calibration, and
thresholding.

Run on both CUDA and CPU for batch sizes 1 and 64 with two warmups and ten timed
repeats.  Report median, p95, milliseconds per sample, throughput, peak CUDA
memory, learned parameter counts, tree-node counts, artifact bytes, and
profiler-supported neural FLOPs.  FFT/tree operations that lack a standard FLOP
definition must be identified rather than silently omitted.

## Preregistered engineering gates

1. CUDA batch-64 complete throughput is at least 250 samples/s.
2. CPU batch-64 complete throughput is at least 20 samples/s.
3. CUDA peak allocated memory is at most 2 GiB.
4. Complete CUDA latency is at most 10 times the H1+RawSaw baseline latency.
5. Non-tree learned-scalar increment is at most 5% of the neural baseline.
6. Total frozen artifact size for one seed is at most 25 MiB.

Engineering failure narrows the deployment claim; it does not authorize model
or threshold changes on development-v12.

