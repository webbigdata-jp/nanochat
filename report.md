# nanochat training report

Generated: 2025-10-16 16:25:22

## Environment

### Git Information
- Branch: master
- Commit: d4a77fb (dirty)
- Message: fix token eval2

### Hardware
- Platform: Linux
- CPUs: 64 cores (64 logical)
- Memory: 2015.6 GB
- GPUs: 8x NVIDIA H100 80GB HBM3
- GPU Memory: 633.5 GB total
- CUDA Version: 12.8
- Hourly Rate: $24.00/hour

### Software
- Python: 3.11.9
- PyTorch: 2.9.0+cu128


### Bloat
- Characters: 382,832
- Lines: 9,485
- Files: 57
- Tokens (approx): 95,708
- Dependencies (uv.lock lines): 2,004

Run started: 2025-10-16 16:25:24

---

## Tokenizer evaluation
timestamp: 2025-10-16 16:25:26

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 705 | 2.58 | -74.5% |
| korean | 893 | 745 | 1.20 | 729 | 1.22 | +2.1% |
| code | 1259 | 576 | 2.19 | 708 | 1.78 | -22.9% |
| math | 1834 | 936 | 1.96 | 1063 | 1.73 | -13.6% |
| science | 1112 | 260 | 4.28 | 455 | 2.44 | -75.0% |
| japanese | 3618 | 2056 | 1.76 | 630 | 5.74 | +69.4% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 705 | 2.58 | -82.2% |
| korean | 893 | 364 | 2.45 | 729 | 1.22 | -100.3% |
| code | 1259 | 309 | 4.07 | 708 | 1.78 | -129.1% |
| math | 1834 | 832 | 2.20 | 1063 | 1.73 | -27.8% |
| science | 1112 | 249 | 4.47 | 455 | 2.44 | -82.7% |
| japanese | 3618 | 1458 | 2.48 | 630 | 5.74 | +56.8% |


## Summary

- Characters: 382,832
- Lines: 9,485
- Files: 57
- Tokens (approx): 95,708
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|

Total wall clock time: 0h0m
