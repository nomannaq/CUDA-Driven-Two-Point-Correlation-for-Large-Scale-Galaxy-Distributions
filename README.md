# CUDA-Driven Two-Point Correlation for Large-Scale Galaxy Distributions

## Overview

This project implements a **CUDA-accelerated Two-Point Angular Correlation Function (TPACF)** for analyzing large-scale galaxy distributions. TPACF is a fundamental tool in cosmology, used to quantify the clustering of galaxies by measuring the excess probability over random distributions at different angular separations. By leveraging GPU parallelization with CUDA, we significantly speed up the computation, making it feasible to handle massive datasets efficiently.

## Features

- **GPU-accelerated** using NVIDIA CUDA for high-performance computation of the Two-Point Correlation Function.
- Implements the **Landy-Szalay estimator**, which helps in reducing edge effects and provides more accurate results.
- Capable of handling **very large datasets** consisting of millions of galaxies.

---

## Prerequisites

### Hardware Requirements

- **NVIDIA GPU** with Compute Capability **7.5** or higher (e.g., NVIDIA GeForce RTX 20xx or 30xx series).
- **Adequate GPU memory** (at least 4 GB recommended for datasets containing millions of galaxies).

### Software Requirements

- **CUDA Toolkit 10.0** or later.
- **C++11-compatible compiler** (e.g., GCC, MSVC).
- NVIDIA CUDA runtime libraries properly installed on the system.

---

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/nomannaq/CUDA-Driven-Two-Point-Correlation-for-Large-Scale-Galaxy-Distributions.git
    cd CUDA-Driven-Two-Point-Correlation-for-Large-Scale-Galaxy-Distributions
    ```

2. **Ensure CUDA is installed**:

    You can verify if CUDA is installed by running:

    ```bash
    nvcc --version
    ```

    If CUDA is not installed, download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

---

## Compilation

To compile the code, use the NVIDIA CUDA compiler `nvcc`:

```bash
nvcc -arch=sm_75 gpu_assignment.cu -o gpu_tpacf
```
## Usage
After compilation, you can run the program as follows:

```bash
./gpu_tpacf real_data.txt random_data.txt results.txt
```
## Arguments
- real_data.txt: Input file containing the real galaxy data (3D coordinates of galaxies).
- random_data.txt: Input file containing randomly generated galaxy coordinates for comparison.
- results.txt: The output file where the computed correlation function (ω values) will be saved.

## Input Data Format
Both the real_data.txt and random_data.txt should have the following format:

```bash
x1 y1 z1
x2 y2 z2
x3 y3 z3
...
```
Where each line represents the 3D coordinates of a galaxy.

## Example
To compile and run the code :
```
nvcc -arch=sm_75 gpu_assignment.cu -o gpu_tpacf
./gpu_tpacf real_data.txt random_data.txt results.txt
```

## Output

The output file (results.txt) contains the calculated omega (ω) values representing the two-point correlation function. Each line corresponds to the angular separation and its associated ω value.

## Example Output Format

``` 
0.1 0.002
0.2 0.005
0.3 0.010
...
```
Where each line contains an angular separation followed by the correlation value.

## Performance

The program has been benchmarked on an NVIDIA GeForce RTX 3050 Ti GPU, achieving a total runtime of 5.82 seconds for a dataset containing 1 million galaxies. Performance can vary depending on the GPU model and the size of the dataset used.






 


