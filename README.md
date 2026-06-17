# Parallel Cuckoo Search with OpenMP: Benchmark Optimization and Fuel Cell Parameter Estimation

A parallel implementation of the **Cuckoo Search Algorithm (CSA)** in C++ using OpenMP, applied to both classical optimization benchmark functions and fuel cell parameter estimation.

## Overview

Cuckoo Search is a bio-inspired metaheuristic based on the nest parasitism behavior of certain cuckoo species. Each candidate solution ("nest") is updated through **Lévy flights**, which combine small local steps with occasional long-distance jumps, providing an effective balance between exploration and exploitation.

The algorithm is particularly well-suited for continuous optimization problems due to its ability to escape local optima and efficiently explore complex search spaces.

---

## Benchmark Functions

| Function           | Domain          | Known Minimum |
| ------------------ | --------------- | ------------- |
| Sinusoidal         | [-2, 2]²        | 0.1           |
| Michalewicz        | [0, 4]²         | ≈ -1.801      |
| Rosenbrock         | [0, 4]²         | 0.0           |
| Six-Hump Camelback | [-3,3] × [-2,2] | ≈ -1.0316     |

These benchmark functions provide a diverse set of optimization challenges, including multimodality, narrow valleys, and highly non-convex landscapes.

---

## Application: Fuel Cell Parameter Estimation

The algorithm is used to calibrate five parameters of a PEM fuel cell voltage model by minimizing three different error metrics:

* **SSE** — Sum of Squared Errors
* **SAE** — Sum of Absolute Errors
* **MAE** — Median Absolute Error

This application demonstrates the effectiveness of Cuckoo Search on a real-world engineering optimization problem.

---

## Compilation

```bash id="6sq9yl"
# With OpenMP (recommended)
g++ -O3 -fopenmp -o cuckoo cuckoo_search.cpp

# Without OpenMP
g++ -O3 -o cuckoo cuckoo_search.cpp
```

---

## Usage

```bash id="zvl4mb"
./cuckoo [function] [threads]

./cuckoo all 4          # Benchmarks + speedup analysis + fuel cell estimation
./cuckoo benchmark 4    # Benchmark functions only
./cuckoo speedup 4      # Scalability analysis (1, 2, 4, 8 threads)
./cuckoo fuelcell 4     # Fuel cell parameter estimation
./cuckoo rosenbrock 4   # Single benchmark function
./cuckoo anim 4         # Generate data for optimization animation
```

---

## Speedup Analysis

The implementation includes a scalability mode that measures execution time using 1, 2, 4, and 8 threads while computing speedup and parallel efficiency.

```text id="c0k9sj"
Threads   Time(s)     Speedup    Efficiency
1         12.3420      1.000      100.0%
2          6.4210      1.922       96.1%
4          3.3150      3.723       93.1%
8          1.9840      6.222       77.8%
```

This analysis provides insight into the scalability of the population-based parallelization strategy implemented with OpenMP.

---

## Algorithm Parameters

| Parameter  | Description                          | Default |
| ---------- | ------------------------------------ | ------- |
| `pop_size` | Number of nests (population size)    | 4000    |
| `max_iter` | Maximum number of iterations         | 500     |
| `pa`       | Fraction of abandoned nests          | 0.25    |
| `alpha`    | Lévy flight step-size scaling factor | 0.01    |
| `beta`     | Lévy distribution exponent           | 1.5     |

---

## Parallelization Strategy

The algorithm leverages **OpenMP** to parallelize population-level operations, allowing multiple nests to be evaluated and updated simultaneously.

Key parallelized components include:

* Candidate solution evaluation
* Lévy flight generation
* Nest replacement and discovery operations
* Benchmark and scalability experiments

This approach significantly reduces execution time while maintaining solution quality.

---

## Technologies

* C++17
* OpenMP (population-level parallelism)
* Mantegna's algorithm for efficient Lévy flight sampling

---

## Research Context

This project investigates the performance of a parallel Cuckoo Search implementation on both synthetic benchmark functions and an engineering parameter estimation problem. Particular emphasis is placed on:

* Metaheuristic optimization
* Parallel computing with OpenMP
* Scalability and speedup analysis
* Continuous optimization in engineering applications
* Bio-inspired search strategies based on Lévy flight dynamics
