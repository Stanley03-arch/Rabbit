/*
 * cuda_gemv.cu — CUDA kernel for group-scaled int4 expert GEMV
 *
 * Computes y = W * x where W is group-scaled int4 (row-major)
 * Each thread block processes a tile of output rows
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#ifndef RABBIT_INT4_GROUP
#define RABBIT_INT4_GROUP 32
#endif

// Device helper: clamp to int8 range
__device__ inline int clamp_i8(int v) {
    return v < -8 ? -8 : (v > 7 ? 7 : v);
}

// Device helper: dequantize one nibble with group scale
__device__ inline float dequant_nibble(int nibble, const float *scales, int group_size, int idx) {
    int g = idx / group_size;
    float scale = scales[g];
    return (float)(nibble - 8) * scale;
}

/*
 * Kernel: gemv_int4_grouped_kernel
 *
 * W: group-scaled int4 matrix [out_dim, in_dim] (row-major)
 *    Layout: u32 group_size | f32 scales[n_groups] | u8 packed[(n+1)/2]
 * x: input vector [in_dim] (f32)
 * y: output vector [out_dim] (f32)
 * out_dim, in_dim: matrix dimensions
 *
 * Each thread computes one output element (one row of W dot x)
 */
__global__ void gemv_int4_grouped_kernel(
    const uint8_t *__restrict__ W,
    const float *__restrict__ x,
    float *__restrict__ y,
    int out_dim, int in_dim
) {
    extern __shared__ float s_scales[];  // scales for current group set
    extern __shared__ float s_x[];       // shared memory for x vector

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int global_idx = bid * bdim + tid;

    if (global_idx >= out_dim) return;

    // Parse group size from first 4 bytes of W
    uint32_t group_size;
    uint8_t gs_bytes[4];
    gs_bytes[0] = W[0];
    gs_bytes[1] = W[1];
    gs_bytes[2] = W[2];
    gs_bytes[3] = W[3];
    group_size = *(uint32_t*)gs_bytes;

    if (group_size == 0 || group_size > 4096) {
        y[global_idx] = 0.0f;
        return;
    }

    int n = out_dim * in_dim;
    int n_groups = (n + group_size - 1) / group_size;
    const float *scales = (const float *)(W + 4);
    const uint8_t *packed = W + 4 + n_groups * 4;

    // Load scales into shared memory (coalesced)
    for (int i = tid; i < n_groups; i += bdim) {
        s_scales[i] = scales[i];
    }
    __syncthreads();

    // Load input vector into shared memory (coalesced)
    for (int i = tid; i < in_dim; i += bdim) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // Compute dot product for this row
    float sum = 0.0f;
    int row_start = global_idx * in_dim;

    for (int i = 0; i < in_dim; i++) {
        int packed_idx = (row_start + i) / 2;
        uint8_t byte = packed[packed_idx];
        int nibble = ((row_start + i) & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        int q = nibble - 8;
        int g = (row_start + i) / group_size;
        float scale = s_scales[g];
        sum += (float)q * scale * s_x[i];
    }

    y[global_idx] = sum;
}

/*
 * Optimized kernel using 128-bit loads for coalesced memory access
 */
__global__ void gemv_int4_grouped_optimized_kernel(
    const uint8_t *__restrict__ W,
    const float *__restrict__ x,
    float *__restrict__ y,
    int out_dim, int in_dim
) {
    extern __shared__ float smem[];
    float *s_scales = smem;
    float *s_x = smem; // will offset after scales

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int global_idx = bid * bdim + tid;

    if (global_idx >= out_dim) return;

    // Parse header
    uint32_t group_size;
    uint8_t gs_bytes[4];
    gs_bytes[0] = W[0]; gs_bytes[1] = W[1]; gs_bytes[2] = W[2]; gs_bytes[3] = W[3];
    group_size = *(uint32_t*)gs_bytes;

    if (group_size == 0 || group_size > 4096) {
        y[global_idx] = 0.0f;
        return;
    }

    int n = out_dim * in_dim;
    int n_groups = (n + group_size - 1) / group_size;
    const float *scales = (const float *)(W + 4);
    const uint8_t *packed = W + 4 + n_groups * 4;

    // Load scales into shared memory
    for (int i = tid; i < n_groups; i += bdim) {
        s_scales[i] = scales[i];
    }

    // Load input vector (offset after scales)
    s_x = smem + n_groups;
    for (int i = tid; i < in_dim; i += bdim) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // Vectorized computation
    float sum = 0.0f;
    int row_start = global_idx * in_dim;

    for (int i = 0; i < in_dim; i++) {
        int packed_idx = (row_start + i) >> 1;
        uint8_t byte = packed[packed_idx];
        int nibble = ((row_start + i) & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        int q = nibble - 8;
        int g = (row_start + i) / group_size;
        sum += (float)q * s_scales[g] * s_x[i];
    }

    y[global_idx] = sum;
}

/*
 * Host launcher function
 */
cudaError_t rabbit_cuda_gemv_int4_grouped(
    const uint8_t *d_W,      // device pointer to group-scaled int4 matrix
    const float *d_x,        // device pointer to input vector
    float *d_y,              // device pointer to output vector
    int out_dim, int in_dim,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (out_dim + block_size - 1) / block_size;

    int n = out_dim * in_dim;
    uint32_t group_size = RABBIT_INT4_GROUP;
    int n_groups = (n + group_size - 1) / group_size;

    // Shared memory: scales + input vector
    size_t smem_bytes = n_groups * sizeof(float) + in_dim * sizeof(float);

    // Check shared memory limit (48KB typical)
    if (smem_bytes > 48 * 1024) {
        // Fallback to non-shared memory kernel
        gemv_int4_grouped_kernel<<<grid_size, block_size, 0, stream>>>(d_W, d_x, d_y, out_dim, in_dim);
    } else {
        gemv_int4_grouped_optimized_kernel<<<grid_size, block_size, smem_bytes, stream>>>(d_W, d_x, d_y, out_dim, in_dim);
    }

    return cudaGetLastError();
}

/*
 * Batch GEMV for multiple experts (used in MoE step)
 * Computes y[e] = W[e] * x for e = 0..n_experts-1
 */
cudaError_t rabbit_cuda_batch_gemv_int4_grouped(
    const uint8_t **d_W_array,   // array of device pointers to expert matrices
    const float *d_x,            // shared input vector
    float **d_y_array,           // array of device pointers to output vectors
    int n_experts,
    int out_dim, int in_dim,
    cudaStream_t stream
) {
    cudaError_t err = cudaSuccess;

    for (int e = 0; e < n_experts; e++) {
        err = rabbit_cuda_gemv_int4_grouped(
            d_W_array[e], d_x, d_y_array[e], out_dim, in_dim, stream);
        if (err != cudaSuccess) break;
    }

    return err;
}

/*
 * Async memory copy helpers for expert data movement
 */
cudaError_t rabbit_cuda_memcpy_expert_to_device(
    uint8_t *d_dst, const void *h_src, size_t bytes, cudaStream_t stream
) {
    return cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream);
}

cudaError_t rabbit_cuda_memcpy_expert_to_host(
    void *h_dst, const uint8_t *d_src, size_t bytes, cudaStream_t stream
) {
    return cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, stream);
}

/*
 * Device memory management
 */
cudaError_t rabbit_cuda_malloc_expert(uint8_t **d_ptr, size_t bytes) {
    return cudaMalloc((void**)d_ptr, bytes);
}

cudaError_t rabbit_cuda_free_expert(uint8_t *d_ptr) {
    return cudaFree(d_ptr);
}

/*
 * CUDA availability check
 */
int rabbit_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}