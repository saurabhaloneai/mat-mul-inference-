#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <thread>

static void log_handler(ggml_log_level level, const char* text, void* user_data) {
    fputs(text, stderr);
    fflush(stderr);
}

class MatrixProcessor {
private:
    struct ggml_tensor* matrix_a;
    struct ggml_tensor* matrix_b;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context* ctx;

    enum class MultiplyMethod {
        GGML,
        CUSTOM_SINGLE_THREAD,
        CUSTOM_MULTI_THREAD
    };

    void initialize_backend() {
        #ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
            if (backend) return;
        #endif
        #ifdef GGML_USE_METAL
            ggml_backend_metal_log_set_callback(log_handler, nullptr);
            backend = ggml_backend_metal_init();
            if (backend) return;
        #endif
        backend = ggml_backend_cpu_init();
    }

    void setup_context() {
        ggml_init_params params{
            ggml_tensor_overhead() * 2,
            nullptr,
            true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("Failed to initialize GGML context");
        }
    }

    void create_tensors() {
        matrix_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
        matrix_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        if (!matrix_a || !matrix_b) {
            throw std::runtime_error("Failed to create tensors");
        }
    }

    std::vector<float> multiply_matrices_single_thread(const float* a, const float* b, 
                                                     int m, int n, int k) {
        std::vector<float> result(m * k, 0.0f);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += a[i * n + p] * b[j * n + p];
                }
                result[j * m + i] = sum;
            }
        }
        
        return result;
    }

    void multiply_matrices_worker(const float* a, const float* b, float* result,
                                int start_row, int end_row, int m, int n, int k) {
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += a[i * n + p] * b[j * n + p];
                }
                result[j * m + i] = sum;
            }
        }
    }

    std::vector<float> multiply_matrices_multi_thread(const float* a, const float* b,
                                                    int m, int n, int k) {
        std::vector<float> result(m * k, 0.0f);
        std::vector<std::thread> threads;
        
        unsigned int num_threads = std::thread::hardware_concurrency();
        int rows_per_thread = m / num_threads;
        int remaining_rows = m % num_threads;
        
        int start_row = 0;
        for (unsigned int i = 0; i < num_threads; i++) {
            int thread_rows = rows_per_thread + (i < (unsigned int)remaining_rows ? 1 : 0);
            int end_row = start_row + thread_rows;
            
            threads.emplace_back(&MatrixProcessor::multiply_matrices_worker, this,
                               a, b, result.data(), start_row, end_row, m, n, k);
            
            start_row = end_row;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return result;
    }

    std::vector<float> compute_using_ggml(const float* a_data, const float* b_data) {
        ggml_backend_tensor_set(matrix_a, a_data, 0, ggml_nbytes(matrix_a));
        ggml_backend_tensor_set(matrix_b, b_data, 0, ggml_nbytes(matrix_b));

        const ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        
        std::vector<uint8_t> compute_buf(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
        ggml_init_params temp_params = {
            compute_buf.size(),
            compute_buf.data(),
            true,
        };

        ggml_context* temp_ctx = ggml_init(temp_params);
        ggml_cgraph* graph = ggml_new_graph(temp_ctx);
        ggml_tensor* result = ggml_mul_mat(temp_ctx, matrix_a, matrix_b);
        ggml_build_forward_expand(graph, result);
        
        ggml_gallocr_reserve(allocr, graph);
        
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, std::thread::hardware_concurrency());
        }
        
        ggml_gallocr_alloc_graph(allocr, graph);
        ggml_backend_graph_compute(backend, graph);

        std::vector<float> output(ggml_nelements(result));
        ggml_backend_tensor_get(result, output.data(), 0, ggml_nbytes(result));

        ggml_free(temp_ctx);
        ggml_gallocr_free(allocr);

        return output;
    }

public:
    MatrixProcessor() : matrix_a(nullptr), matrix_b(nullptr), backend(nullptr), buffer(nullptr), ctx(nullptr) {
        initialize_backend();
        setup_context();
        create_tensors();
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    ~MatrixProcessor() {
        if (ctx) ggml_free(ctx);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
    }

    std::vector<float> compute_matrix_multiplication(const float* a_data, const float* b_data, 
                                                   MultiplyMethod method = MultiplyMethod::CUSTOM_MULTI_THREAD) {
        switch (method) {
            case MultiplyMethod::GGML:
                return compute_using_ggml(a_data, b_data);
            
            case MultiplyMethod::CUSTOM_SINGLE_THREAD:
                return multiply_matrices_single_thread(a_data, b_data, 4, 2, 3);
            
            case MultiplyMethod::CUSTOM_MULTI_THREAD:
                return multiply_matrices_multi_thread(a_data, b_data, 4, 2, 3);
            
            default:
                throw std::runtime_error("Unknown multiplication method");
        }
    }

    void print_result(const std::vector<float>& result, int rows, int cols) {
        printf("Matrix multiplication result (%d x %d):\n[", rows, cols);
        for (int i = 0; i < rows; ++i) {
            if (i > 0) printf("\n");
            for (int j = 0; j < cols; ++j) {
                printf(" %.2f", result[j * rows + i]);
            }
        }
        printf(" ]");
    }

    void benchmark_all_methods(const float* a_data, const float* b_data) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result_ggml = compute_matrix_multiplication(a_data, b_data, MultiplyMethod::GGML);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto ggml_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        start_time = std::chrono::high_resolution_clock::now();
        auto result_single = compute_matrix_multiplication(a_data, b_data, MultiplyMethod::CUSTOM_SINGLE_THREAD);
        end_time = std::chrono::high_resolution_clock::now();
        auto single_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        start_time = std::chrono::high_resolution_clock::now();
        auto result_multi = compute_matrix_multiplication(a_data, b_data, MultiplyMethod::CUSTOM_MULTI_THREAD);
        end_time = std::chrono::high_resolution_clock::now();
        auto multi_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        printf("\nBenchmark Results:\n");
        printf("GGML: %lld microseconds\n", ggml_duration.count());
        printf("Single Thread: %lld microseconds\n", single_duration.count());
        printf("Multi Thread: %lld microseconds\n", multi_duration.count());
    }
};

int main() {
    ggml_time_init();

    float matrix_A[4 * 2] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    float matrix_B[3 * 2] = {
        10, 5,
        9, 9,
        5, 4
    };

    try {
        MatrixProcessor processor;
        
        printf("different multiplication types");
        processor.benchmark_all_methods(matrix_A, matrix_B);
        
        printf("using custom multi-threaded implementation:");
        auto result = processor.compute_matrix_multiplication(matrix_A, matrix_B, 
                                                           MatrixProcessor::MultiplyMethod::CUSTOM_MULTI_THREAD);
        processor.print_result(result, 4, 3);
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}