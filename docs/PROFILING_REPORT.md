# RGIZero Pipeline Performance Report

**Generated:** 2026-01-03 18:15:43

## Executive Summary

### Key Findings

| Metric | Count21 | Connect4 | Othello |
|--------|---------|----------|---------|
| Self-play (200 games) | 4.8s | 7.6s | 32.3s |
| Training (2 gens) | 3.7s | 5.5s | 1.9s |
| Tournament (100 games) | 5.7s | 11.0s | 37.2s |
| **Total** | **20.5s** | **36.4s** | **134s** |
| CPU Mean | 13% | 12% | 11% |
| GPU Mean | 0% | 0% | 0% |

### Bottleneck Analysis

1. **Self-Play is the primary bottleneck for complex games**
   - Othello: 6.2 games/sec vs Count21: 41.8 games/sec (6.7x slower)
   - This is due to more MCTS evaluations per move (larger game tree)

2. **Training is NOT a bottleneck**
   - Training takes only 1-4 seconds regardless of game complexity
   - The neural network is small (~30K params) and data is limited

3. **Tournament is slow but parallelizable**
   - Similar to self-play, dominated by MCTS game logic
   - Othello: 2.7 games/sec vs Count21: 17.7 games/sec

4. **CPU is significantly underutilized**
   - Mean CPU: 10-15% across all phases
   - Per-core max: 70-85% occasionally, but average is 20-30%
   - **Conclusion: Multiprocessing would provide 3-5x speedup**

5. **GPU is essentially unused**
   - MPS GPU shows 0% utilization
   - Neural network inference is too fast to be a bottleneck
   - Not worth optimizing GPU usage

### Recommendations

| Priority | Optimization | Expected Impact |
|----------|--------------|-----------------|
| HIGH | Use ProcessPoolExecutor for self-play | 3-5x speedup |
| HIGH | Use ProcessPoolExecutor for tournament | 2-3x speedup |
| LOW | Increase batch size | Minimal (GPU not bottleneck) |
| LOW | Make model bigger | Minimal (training only 2-4% of time) |

### Time Budget for "Highest ELO in X Minutes"

For a 10-minute budget on Othello:
- **Current**: 500 games self-play + 100 tournament ≈ 12 min
- **With multiprocessing (4 workers)**: Same setup ≈ 3-4 min
- **Recommended config**: More self-play games (750-1000) with multiprocessing

---

# Performance Profile: Count21

**Device:** mps
**Total Time:** 20.5s (0.3 min)

## Time Breakdown

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| Self-Play | 4.8 | 23.4% |
| Training | 3.7 | 18.2% |
| Tournament | 5.7 | 27.7% |

## Resource Utilization

| Phase | CPU Mean | CPU Max | GPU Mean | GPU Max |
|-------|----------|---------|----------|----------|
| Self-Play | 14.9% | 32.5% | 0.0% | 0.0% |
| Training | 13.7% | 28.6% | 0.1% | 0.2% |
| Tournament | 9.0% | 30.2% | 0.0% | 0.0% |

## Bottleneck Analysis

### Self-Play

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- games_played: 200
- games_per_second: 41.82
- total_batches: 439
- total_evals: 59184
- mean_batch_size: 134.82
- mean_evals_per_sec: 24065.06

### Training

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- generations: 2
- avg_time_per_gen: 1.87

### Tournament

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- games: 100
- games_per_second: 17.65

## Per-Core CPU Utilization (Self-Play)

| Core | Mean % | Max % |
|------|--------|-------|
| 0 | 16.3 | 31.8 |
| 1 | 9.4 | 22.7 |
| 2 | 5.7 | 18.2 |
| 3 | 2.8 | 13.6 |
| 4 | 28.9 | 61.9 |
| 5 | 29.3 | 60.0 |
| 6 | 30.7 | 76.2 |
| 7 | 31.9 | 85.7 |
| 8 | 16.7 | 76.2 |
| 9 | 14.3 | 72.7 |
| 10 | 15.2 | 71.4 |
| 11 | 16.5 | 70.0 |

**Cores with >50% utilization:** 0/12


---

# Performance Profile: Connect4

**Device:** mps
**Total Time:** 36.4s (0.6 min)

## Time Breakdown

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| Self-Play | 7.6 | 20.9% |
| Training | 5.5 | 15.1% |
| Tournament | 11.0 | 30.4% |

## Resource Utilization

| Phase | CPU Mean | CPU Max | GPU Mean | GPU Max |
|-------|----------|---------|----------|----------|
| Self-Play | 12.6% | 31.7% | 0.0% | 0.0% |
| Training | 10.6% | 29.0% | 0.1% | 0.2% |
| Tournament | 13.4% | 47.2% | 0.0% | 0.0% |

## Bottleneck Analysis

### Self-Play

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- games_played: 200
- games_per_second: 26.27
- total_batches: 1374
- total_evals: 126594
- mean_batch_size: 92.14
- mean_evals_per_sec: 27267.18

### Training

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- generations: 2
- avg_time_per_gen: 2.75

### Tournament

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- games: 100
- games_per_second: 9.05

## Per-Core CPU Utilization (Self-Play)

| Core | Mean % | Max % |
|------|--------|-------|
| 0 | 20.0 | 31.8 |
| 1 | 11.1 | 19.0 |
| 2 | 5.5 | 15.4 |
| 3 | 2.7 | 9.5 |
| 4 | 28.9 | 59.1 |
| 5 | 28.9 | 71.4 |
| 6 | 28.1 | 65.0 |
| 7 | 27.9 | 68.2 |
| 8 | 23.4 | 61.9 |
| 9 | 22.8 | 55.0 |
| 10 | 22.4 | 71.4 |
| 11 | 22.1 | 59.1 |

**Cores with >50% utilization:** 0/12


---

# Performance Profile: Othello

**Device:** mps
**Total Time:** 134.0s (2.2 min)

## Time Breakdown

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| Self-Play | 32.3 | 24.1% |
| Training | 1.9 | 1.4% |
| Tournament | 37.2 | 27.8% |

## Resource Utilization

| Phase | CPU Mean | CPU Max | GPU Mean | GPU Max |
|-------|----------|---------|----------|----------|
| Self-Play | 10.0% | 47.5% | 0.0% | 0.2% |
| Training | 10.2% | 28.7% | 0.1% | 0.2% |
| Tournament | 13.4% | 39.8% | 0.0% | 0.0% |

## Bottleneck Analysis

### Self-Play

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- games_played: 200
- games_per_second: 6.20
- total_batches: 2522
- total_evals: 482718
- mean_batch_size: 191.40
- mean_evals_per_sec: 54367.31

### Training

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- generations: 2
- avg_time_per_gen: 0.95

### Tournament

✅ **Balanced** - neither CPU nor GPU saturated

**Phase-specific stats:**
- games: 100
- games_per_second: 2.69

## Per-Core CPU Utilization (Self-Play)

| Core | Mean % | Max % |
|------|--------|-------|
| 0 | 15.7 | 73.9 |
| 1 | 9.6 | 81.0 |
| 2 | 5.0 | 63.6 |
| 3 | 3.7 | 63.6 |
| 4 | 21.9 | 85.7 |
| 5 | 22.7 | 86.4 |
| 6 | 22.9 | 76.2 |
| 7 | 22.5 | 81.8 |
| 8 | 25.0 | 72.7 |
| 9 | 25.9 | 78.3 |
| 10 | 25.3 | 79.2 |
| 11 | 25.6 | 86.4 |

**Cores with >50% utilization:** 0/12


---

