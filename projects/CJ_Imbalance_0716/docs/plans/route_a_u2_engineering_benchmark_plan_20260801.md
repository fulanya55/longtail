# Route A U2 engineering benchmark plan

Benchmark frozen seed 2039 on 512 development-v3 early samples, batch size 64, after two warmups.
Report U0 and U2 end-to-end latency, component latency, throughput, peak GPU memory, checkpoint and
expert storage, neural parameter counts, and profiler-supported neural FLOPs. Run on RTX 5080 and
CPU. Data loading is excluded; calibration and thresholds are included in end-to-end postprocessing.
The FullAlpha task vector is a single merged inference checkpoint, not an inference ensemble. Keep
test-v3 sealed and unused.
