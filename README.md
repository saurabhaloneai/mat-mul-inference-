# mat-mul-inference

> [!IMPORTANT]
>
> ... ggml_mat_mal -> trying to undertanddddd🐳

simple matrix multiplication implementation with gpu acceleration support and multiple computation methods.

## mathematical basis

### matrix multiplication
for matrices $A_{m×n}$ and $B_{n×k}$, the product $C_{m×k} = AB$ is:

$C_{ij} = \sum_{p=1}^{n} A_{ip} \times B_{pj}$

### example
```
A = [2 8]     B = [10 9 5]
    [5 1]         [5  9 4]
    [4 2]
    [8 6]
```

resulting in:

$C = \begin{bmatrix} 
60 & 90 & 42 \\
55 & 54 & 29 \\
50 & 54 & 28 \\
110 & 126 & 64
\end{bmatrix}$

## implementation methods

ggml backend
- supports gpu acceleration (cuda/metal)
- $C = A \times B^T$

single thread cpu
- basic iteration implementation
- complexity: $O(m \times n \times k)$

multi thread cpu
- parallel row computation
- threads = $min(cpu\_cores, matrix\_rows)$
- rows per thread = $\lceil \frac{matrix\_rows}{num\_threads} \rceil$

## ex

```cpp
float matrix_a[4 * 2] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};

float matrix_b[3 * 2] = {
    10, 5,
    9, 9,
    5, 4
};

MatrixProcessor processor;
auto result = processor.compute_matrix_multiplication(
    matrix_a, 
    matrix_b, 
    MatrixProcessor::MultiplyMethod::CUSTOM_MULTI_THREAD
);
```