/*********************************************************************
 * convolve.cu
 * Optimized GPU KLT convolution with MINIMAL changes to working code
 * Fixed: defensive kernels, constant memory copy safety, cleanup, shared mem checks
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>

extern "C" {
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"
}

#define MAX_KERNEL_WIDTH 71

// -----------------------------------------
// Structures
// -----------------------------------------
typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

// -----------------------------------------
// Global kernels
// -----------------------------------------
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;

// -----------------------------------------
// CUDA constant memory for kernels
// -----------------------------------------
__constant__ float d_gauss[MAX_KERNEL_WIDTH];
__constant__ float d_gaussderiv[MAX_KERNEL_WIDTH];
__constant__ int d_gauss_width;
__constant__ int d_gaussderiv_width;

// -----------------------------------------
// Error checking macro
// -----------------------------------------
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern "C" {
void _convolveImageHorizGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout);
void _convolveImageVertGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout);
}

// -----------------------------------------
// OPTIMIZATION 1: Device buffers (replace Unified Memory)
// -----------------------------------------
static float *d_buf0 = nullptr;  // device input / temporary
static float *d_buf1 = nullptr;  // device output / temporary
static int device_ncols = 0;
static int device_nrows = 0;

// OPTIMIZATION 2: CUDA stream for async operations
static cudaStream_t stream = 0;

// OPTIMIZATION 3: Pinned host memory for faster transfers
static float *pinned_buf0 = nullptr;
static float *pinned_buf1 = nullptr;
static int pinned_ncols = 0;
static int pinned_nrows = 0;

static void ensureDeviceBuffers(int ncols, int nrows) {
    size_t elems = (size_t)ncols * (size_t)nrows;
    if (d_buf0 && device_ncols == ncols && device_nrows == nrows) return;

    if (d_buf0) {
        cudaFree(d_buf0);
        cudaFree(d_buf1);
        d_buf0 = d_buf1 = nullptr;
    }

    CUDA_CHECK(cudaMalloc(&d_buf0, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_buf1, elems * sizeof(float)));
    device_ncols = ncols;
    device_nrows = nrows;
    
    // Create stream if not exists
    if (!stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
}

static void ensurePinnedBuffers(int ncols, int nrows) {
    size_t elems = (size_t)ncols * (size_t)nrows;
    if (pinned_buf0 && pinned_ncols == ncols && pinned_nrows == nrows) return;

    if (pinned_buf0) {
        cudaFreeHost(pinned_buf0);
        cudaFreeHost(pinned_buf1);
        pinned_buf0 = pinned_buf1 = nullptr;
    }

    CUDA_CHECK(cudaMallocHost(&pinned_buf0, elems * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&pinned_buf1, elems * sizeof(float)));
    pinned_ncols = ncols;
    pinned_nrows = nrows;
}

extern "C" void _KLTFreeDeviceBuffers() {
    if (d_buf0) {
        cudaFree(d_buf0);
        cudaFree(d_buf1);
        d_buf0 = d_buf1 = nullptr;
        device_ncols = device_nrows = 0;
    }
    if (pinned_buf0) {
        cudaFreeHost(pinned_buf0);
        cudaFreeHost(pinned_buf1);
        pinned_buf0 = pinned_buf1 = nullptr;
        pinned_ncols = pinned_nrows = 0;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = 0;
    }
}

// -----------------------------------------
// Host: Convert KLT image to float (UNCHANGED)
// -----------------------------------------
extern "C" void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
    KLT_PixelType *ptrend = img + ncols * nrows;
    float *ptrout = floatimg->data;

    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    while (img < ptrend) *ptrout++ = (float)*img++;
}

// -----------------------------------------
// Host: Compute kernels from sigma (FIXED & DEFENSIVE)
// -----------------------------------------
static void _computeKernels(
    float sigma,
    ConvolutionKernel *gauss,
    ConvolutionKernel *gaussderiv)
{
    const float factor = 0.01f;
    int i;

    // Defensive: avoid sigma==0 or extremely small which creates NaNs or degenerate kernels
    if (sigma < 1e-3f) sigma = 1e-3f;

    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0f);

    // initialize to safe defaults
    for (i = 0; i < MAX_KERNEL_WIDTH; ++i) {
        gauss->data[i] = 0.0f;
        gaussderiv->data[i] = 0.0f;
    }
    gauss->width = MAX_KERNEL_WIDTH;
    gaussderiv->width = MAX_KERNEL_WIDTH;

    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 0.0f, max_gaussderiv = 0.0f;

    for (i = -hw; i <= hw; i++) {
        float g = (float)exp(- (float)(i*i) / (2.0f * sigma * sigma));
        float gd = -i * g;
        gauss->data[i+hw]      = g;
        gaussderiv->data[i+hw] = gd;
        if (g > max_gauss) max_gauss = g;
        if (fabsf(gd) > max_gaussderiv) max_gaussderiv = fabsf(gd);
    }

    // start with full width and trim symmetric tails while keeping odd width
    gauss->width = MAX_KERNEL_WIDTH;
    gaussderiv->width = MAX_KERNEL_WIDTH;

    // Trim gauss
    while (gauss->width > 1) {
        int left_idx = (MAX_KERNEL_WIDTH - gauss->width)/2;
        float denom = (max_gauss == 0.0f) ? 1.0f : max_gauss;
        if (fabsf(gauss->data[left_idx] / denom) >= factor) break;
        gauss->width -= 2;
    }
    // Trim gaussderiv
    while (gaussderiv->width > 1) {
        int left_idx = (MAX_KERNEL_WIDTH - gaussderiv->width)/2;
        float denom = (max_gaussderiv == 0.0f) ? 1.0f : max_gaussderiv;
        if (fabsf(gaussderiv->data[left_idx] / denom) >= factor) break;
        gaussderiv->width -= 2;
    }

    // copy trimmed center into compact arrays (centered)
    int start_g  = (MAX_KERNEL_WIDTH - gauss->width) / 2;
    int start_gd = (MAX_KERNEL_WIDTH - gaussderiv->width) / 2;
    for (i = 0; i < gauss->width; ++i) {
        gauss->data[i] = gauss->data[start_g + i];
    }
    for (i = 0; i < gaussderiv->width; ++i) {
        gaussderiv->data[i] = gaussderiv->data[start_gd + i];
    }

    // normalize gauss
    float den = 0.0f;
    for (i = 0; i < gauss->width; ++i) den += gauss->data[i];
    if (den == 0.0f) den = 1.0f;
    for (i = 0; i < gauss->width; ++i) gauss->data[i] /= den;

    // normalize gaussderiv
    int hw2 = gaussderiv->width / 2;
    den = 0.0f;
    for (i = -hw2; i <= hw2; i++) den -= i * gaussderiv->data[i + hw2];
    if (den == 0.0f) den = 1.0f;
    for (i = -hw2; i <= hw2; i++) gaussderiv->data[i + hw2] /= den;

    sigma_last = sigma;
}

extern "C" void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width)
{
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    if (gauss_width) *gauss_width = gauss_kernel.width;
    if (gaussderiv_width) *gaussderiv_width = gaussderiv_kernel.width;
}

// -----------------------------------------
// GPU: horizontal convolution kernel (UNCHANGED - your working code)
// -----------------------------------------
template<int TX, int TY>
__global__ void convolve_horiz_shared_kernel(
    const float *__restrict__ imgin,
    float *__restrict__ imgout,
    int width, int height, int kwidth)
{
    
    const int radius = kwidth / 2;
    const int x0 = blockIdx.x * TX;
    const int y0 = blockIdx.y * TY;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    extern __shared__ float s_data[];
    int sW = TX + 2 * radius;

    int localRow = ty;
    int globalRow = y0 + localRow;
    if (globalRow >= height) return;

    for (int c = tx; c < sW; c += blockDim.x) {
        int gx = x0 + c - radius;
        gx = max(0, min(gx, width - 1));
        s_data[localRow * sW + c] = imgin[globalRow * width + gx];
    }
    __syncthreads();

    for (int x = tx; x < TX; x += blockDim.x) {
        int gx = x0 + x;
        if (gx >= width) continue;
        float sum = 0.0f;
        int center = x + radius;
        for (int k = 0; k < kwidth; ++k)
            sum += s_data[localRow * sW + (center - radius + k)] * d_gauss[kwidth - 1 - k];
        imgout[globalRow * width + gx] = sum;
    }
}

// -----------------------------------------
// GPU: vertical convolution kernel (UNCHANGED - your working code)
// -----------------------------------------
template<int TX, int TY>
__global__ void convolve_vert_shared_kernel(
    const float *__restrict__ imgin,
    float *__restrict__ imgout,
    int width, int height, int kwidth)
{
    //printf("[convolve_vert_shared_kernel] GPU running\n");
    const int radius = kwidth / 2;
    const int x0 = blockIdx.x * TX;
    const int y0 = blockIdx.y * TY;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    extern __shared__ float s_data[];
    int sH = TY + 2 * radius;

    int localCol = tx;
    int globalCol = x0 + localCol;
    if (globalCol >= width) return;

    for (int r = ty; r < sH; r += blockDim.y) {
        int gy = y0 + r - radius;
        gy = max(0, min(gy, height - 1));
        s_data[r * TX + localCol] = imgin[gy * width + globalCol];
    }
    __syncthreads();

    for (int y = ty; y < TY; y += blockDim.y) {
        int gy = y0 + y;
        if (gy >= height) continue;
        float sum = 0.0f;
        int center = y + radius;
        for (int k = 0; k < kwidth; ++k)
            sum += s_data[(center - radius + k) * TX + localCol] * d_gauss[kwidth - 1 - k];
        imgout[gy * width + globalCol] = sum;
    }
}

// Helper: safe copy of kernel into constant (zero-padded)
static void copyKernelToConstAsync(const float *kernel_data, int kwidth, float const_symbol[], cudaStream_t stream) {
    float tmp[MAX_KERNEL_WIDTH];
    // zero out
    for (int i = 0; i < MAX_KERNEL_WIDTH; ++i) tmp[i] = 0.0f;
    // copy active kernel
    if (kwidth > MAX_KERNEL_WIDTH) kwidth = MAX_KERNEL_WIDTH;
    for (int i = 0; i < kwidth; ++i) tmp[i] = kernel_data[i];
    // copy whole buffer to symbol (we use symbol name via template specialization)
    // This function is used with explicit symbol names by caller via cudaMemcpyToSymbolAsync
    // (we pass the symbol address from caller)
    // This helper constructs tmp; actual cudaMemcpyToSymbolAsync is called by caller.
    (void)const_symbol; (void)stream; // silence unused (actual copy done by caller)
}

// -----------------------------------------
// Host: GPU convolution wrappers (OPTIMIZED with pinned memory + stream)
// -----------------------------------------
extern "C" void _convolveImageHorizGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    size_t elems = (size_t)ncols * (size_t)nrows;
    size_t imgSize = elems * sizeof(float);
    
    ensureDeviceBuffers(ncols, nrows);
    ensurePinnedBuffers(ncols, nrows);
    
    // Copy to pinned memory first (fast memcpy)
    memcpy(pinned_buf0, imgin->data, imgSize);
    
    // Async transfer to device
    CUDA_CHECK(cudaMemcpyAsync(d_buf0, pinned_buf0, imgSize, cudaMemcpyHostToDevice, stream));
    
    // Copy kernel to constant memory (zero-padded whole symbol)
    {
        float tmp[MAX_KERNEL_WIDTH];
        memset(tmp, 0, sizeof(tmp));
        memcpy(tmp, kernel.data, kernel.width * sizeof(float));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
        // also copy width (optional)
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    }
    
    const int TX = 32;
    const int TY = 8;
    dim3 block(32, 8);
    dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
    size_t shmem = (size_t)TY * (TX + 2 * (kernel.width/2)) * sizeof(float);

    // check shared mem limits
    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_convolveImageHorizGPU] requested shmem %zu > device limit %zu\n", shmem, (size_t)prop.sharedMemPerBlock);
            // Fallback: reduce tile size or abort. Here we abort to make problem obvious.
            CUDA_CHECK(cudaErrorInvalidValue);
        }
    }

    convolve_horiz_shared_kernel<32,8><<<grid, block, shmem, stream>>>(d_buf0, d_buf1, ncols, nrows, kernel.width);
    CUDA_CHECK(cudaGetLastError());
    
    // Async transfer back
    CUDA_CHECK(cudaMemcpyAsync(pinned_buf1, d_buf1, imgSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    
    // Copy from pinned to output
    memcpy(imgout->data, pinned_buf1, imgSize);
}

extern "C" void _convolveImageVertGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    size_t elems = (size_t)ncols * (size_t)nrows;
    size_t imgSize = elems * sizeof(float);
    
    ensureDeviceBuffers(ncols, nrows);
    ensurePinnedBuffers(ncols, nrows);
    
    // Copy to pinned memory
    memcpy(pinned_buf0, imgin->data, imgSize);
    
    // Async transfer to device
    CUDA_CHECK(cudaMemcpyAsync(d_buf0, pinned_buf0, imgSize, cudaMemcpyHostToDevice, stream));
    
    // Copy kernel to constant memory (zero-padded)
    {
        float tmp[MAX_KERNEL_WIDTH];
        memset(tmp, 0, sizeof(tmp));
        memcpy(tmp, kernel.data, kernel.width * sizeof(float));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    }
    
    const int TX = 16;
    const int TY = 16;
    dim3 block(TX, TY);
    dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
    size_t shmem = (size_t)TX * (TY + 2 * (kernel.width/2)) * sizeof(float);

    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_convolveImageVertGPU] requested shmem %zu > device limit %zu\n", shmem, (size_t)prop.sharedMemPerBlock);
            CUDA_CHECK(cudaErrorInvalidValue);
        }
    }
    
    convolve_vert_shared_kernel<16,16><<<grid, block, shmem, stream>>>(d_buf0, d_buf1, ncols, nrows, kernel.width);
    CUDA_CHECK(cudaGetLastError());
    
    // Async transfer back
    CUDA_CHECK(cudaMemcpyAsync(pinned_buf1, d_buf1, imgSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy from pinned to output
    memcpy(imgout->data, pinned_buf1, imgSize);
}

// -----------------------------------------
// OPTIMIZATION 4: Keep data on GPU between operations
// -----------------------------------------
extern "C" void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, _KLT_FloatImage smooth)
{
    if (fabs(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    
    int ncols = img->ncols;
    int nrows = img->nrows;
    size_t elems = (size_t)ncols * (size_t)nrows;
    size_t imgSize = elems * sizeof(float);
    
    ensureDeviceBuffers(ncols, nrows);
    ensurePinnedBuffers(ncols, nrows);
    
    // Copy to pinned memory
    memcpy(pinned_buf0, img->data, imgSize);
    
    // SINGLE upload to device
    CUDA_CHECK(cudaMemcpyAsync(d_buf0, pinned_buf0, imgSize, cudaMemcpyHostToDevice, stream));
    
    // Copy kernel once (zero-padded)
    {
        float tmp[MAX_KERNEL_WIDTH];
        memset(tmp, 0, sizeof(tmp));
        memcpy(tmp, gauss_kernel.data, gauss_kernel.width * sizeof(float));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &gauss_kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    }
    
    // Horizontal convolution: d_buf0 -> d_buf1
    const int TX_H = 32, TY_H = 8;
    dim3 block_h(32, 8);
    dim3 grid_h((ncols + TX_H - 1) / TX_H, (nrows + TY_H - 1) / TY_H);
    size_t shmem_h = (size_t)TY_H * (TX_H + 2 * (gauss_kernel.width/2)) * sizeof(float);

    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem_h > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_KLTComputeSmoothedImage] requested shmem_h %zu > device limit %zu\n", shmem_h, (size_t)prop.sharedMemPerBlock);
            CUDA_CHECK(cudaErrorInvalidValue);
        }
    }

    convolve_horiz_shared_kernel<32,8><<<grid_h, block_h, shmem_h, stream>>>(d_buf0, d_buf1, ncols, nrows, gauss_kernel.width);

    // Vertical convolution: d_buf1 -> d_buf0
    const int TX_V = 16, TY_V = 16;
    dim3 block_v(TX_V, TY_V);
    dim3 grid_v((ncols + TX_V - 1) / TX_V, (nrows + TY_V - 1) / TY_V);
    size_t shmem_v = (size_t)TX_V * (TY_V + 2 * (gauss_kernel.width/2)) * sizeof(float);
    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem_v > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_KLTComputeSmoothedImage] requested shmem_v %zu > device limit %zu\n", shmem_v, (size_t)prop.sharedMemPerBlock);
            CUDA_CHECK(cudaErrorInvalidValue);
        }
    }
    convolve_vert_shared_kernel<16,16><<<grid_v, block_v, shmem_v, stream>>>(d_buf1, d_buf0, ncols, nrows, gauss_kernel.width);

    CUDA_CHECK(cudaGetLastError());

    // SINGLE download from device
    CUDA_CHECK(cudaMemcpyAsync(pinned_buf1, d_buf0, imgSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy from pinned to output
    memcpy(smooth->data, pinned_buf1, imgSize);
}

// -----------------------------------------
// OPTIMIZATION 5: Gradients - keep all on GPU
// -----------------------------------------
// -----------------------------------------
// OPTIMIZATION 5: Gradients - keep all on GPU
// (goto removed; uses an 'ok' flag and explicit cleanup)
// -----------------------------------------
extern "C" void _KLTComputeGradients(_KLT_FloatImage img, float sigma,
                                     _KLT_FloatImage gradx, _KLT_FloatImage grady)
{
    //printf("[_KLTComputeGradients] GPU running\n");
    assert(gradx->ncols >= img->ncols && gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols && grady->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    int ncols = img->ncols;
    int nrows = img->nrows;
    size_t elems = (size_t)ncols * (size_t)nrows;
    size_t imgSize = elems * sizeof(float);

    ensureDeviceBuffers(ncols, nrows);
    ensurePinnedBuffers(ncols, nrows);

    float *d_grady = nullptr;
    float *d_input_copy = nullptr;
    cudaError_t cuErr = cudaSuccess;
    bool ok = true;

    // allocate d_grady
    cuErr = cudaMalloc(&d_grady, imgSize);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "[_KLTComputeGradients] cudaMalloc d_grady failed: %s\n", cudaGetErrorString(cuErr));
        ok = false;
    }

    // upload input to device
    if (ok) {
        memcpy(pinned_buf0, img->data, imgSize);
        cuErr = cudaMemcpyAsync(d_buf0, pinned_buf0, imgSize, cudaMemcpyHostToDevice, stream);
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaMemcpyAsync to d_buf0 failed: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }

    // allocate d_input_copy and copy device-to-device
    if (ok) {
        cuErr = cudaMalloc(&d_input_copy, imgSize);
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaMalloc d_input_copy failed: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }
    if (ok) {
        cuErr = cudaStreamSynchronize(stream);
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaStreamSynchronize failed after upload: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }
    if (ok) {
        cuErr = cudaMemcpyAsync(d_input_copy, d_buf0, imgSize, cudaMemcpyDeviceToDevice, stream);
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaMemcpyAsync D2D failed: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }

    const int TX_H = 32, TY_H = 8;
    const int TX_V = 16, TY_V = 16;
    dim3 block_h(32, 8);
    dim3 block_v(TX_V, TY_V);
    dim3 grid_h((ncols + TX_H - 1) / TX_H, (nrows + TY_H - 1) / TY_H);
    dim3 grid_v((ncols + TX_V - 1) / TX_V, (nrows + TY_V - 1) / TY_V);
    size_t shmem_h_gauss = (size_t)TY_H * (TX_H + 2 * (gauss_kernel.width/2)) * sizeof(float);
    size_t shmem_h_deriv = (size_t)TY_H * (TX_H + 2 * (gaussderiv_kernel.width/2)) * sizeof(float);
    size_t shmem_v_gauss = (size_t)TX_V * (TY_V + 2 * (gauss_kernel.width/2)) * sizeof(float);
    size_t shmem_v_deriv = (size_t)TX_V * (TY_V + 2 * (gaussderiv_kernel.width/2)) * sizeof(float);

    if (ok) {
        // ---------- Compute gradx using d_input_copy as SOURCE ----------
        // horiz with gaussderiv -> d_buf1
        {
            float tmp[MAX_KERNEL_WIDTH];
            memset(tmp, 0, sizeof(tmp));
            memcpy(tmp, gaussderiv_kernel.data, gaussderiv_kernel.width * sizeof(float));
            CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &gaussderiv_kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
        }

        // check shared mem
        {
            int dev;
            cudaGetDevice(&dev);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, dev);
            if (shmem_h_deriv > (size_t)prop.sharedMemPerBlock) {
                fprintf(stderr, "[_KLTComputeGradients] shmem_h_deriv %zu > device limit %zu\n", shmem_h_deriv, (size_t)prop.sharedMemPerBlock);
                ok = false;
            }
        }
    }

    if (ok) convolve_horiz_shared_kernel<32,8><<<grid_h, block_h, shmem_h_deriv, stream>>>(d_input_copy, d_buf1, ncols, nrows, gaussderiv_kernel.width);

    if (ok) {
        // vert with gauss -> d_buf0 (gradx)
        float tmp[MAX_KERNEL_WIDTH];
        memset(tmp, 0, sizeof(tmp));
        memcpy(tmp, gauss_kernel.data, gauss_kernel.width * sizeof(float));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &gauss_kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));

        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem_v_gauss > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_KLTComputeGradients] shmem_v_gauss %zu > device limit %zu\n", shmem_v_gauss, (size_t)prop.sharedMemPerBlock);
            ok = false;
        }
    }

    if (ok) convolve_vert_shared_kernel<16,16><<<grid_v, block_v, shmem_v_gauss, stream>>>(d_buf1, d_buf0, ncols, nrows, gauss_kernel.width);

    if (ok) {
        // ---------- Compute grady using d_input_copy as SOURCE ----------
        // horiz with gauss -> d_buf1
        float tmp[MAX_KERNEL_WIDTH];
        memset(tmp, 0, sizeof(tmp));
        memcpy(tmp, gauss_kernel.data, gauss_kernel.width * sizeof(float));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &gauss_kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));

        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem_h_gauss > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_KLTComputeGradients] shmem_h_gauss %zu > device limit %zu\n", shmem_h_gauss, (size_t)prop.sharedMemPerBlock);
            ok = false;
        }
    }

    if (ok) convolve_horiz_shared_kernel<32,8><<<grid_h, block_h, shmem_h_gauss, stream>>>(d_input_copy, d_buf1, ncols, nrows, gauss_kernel.width);

    if (ok) {
        // vert with gaussderiv -> d_grady
        float tmp[MAX_KERNEL_WIDTH];
        memset(tmp, 0, sizeof(tmp));
        memcpy(tmp, gaussderiv_kernel.data, gaussderiv_kernel.width * sizeof(float));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, tmp, sizeof(tmp), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss_width, &gaussderiv_kernel.width, sizeof(int), 0, cudaMemcpyHostToDevice, stream));

        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        if (shmem_v_deriv > (size_t)prop.sharedMemPerBlock) {
            fprintf(stderr, "[_KLTComputeGradients] shmem_v_deriv %zu > device limit %zu\n", shmem_v_deriv, (size_t)prop.sharedMemPerBlock);
            ok = false;
        }
    }

    if (ok) convolve_vert_shared_kernel<16,16><<<grid_v, block_v, shmem_v_deriv, stream>>>(d_buf1, d_grady, ncols, nrows, gaussderiv_kernel.width);

    if (ok) {
        CUDA_CHECK(cudaGetLastError());

        // Download both results
        cuErr = cudaMemcpyAsync(pinned_buf0, d_buf0, imgSize, cudaMemcpyDeviceToHost, stream);      // gradx
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaMemcpyAsync gradx failed: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }

    if (ok) {
        cuErr = cudaMemcpyAsync(pinned_buf1, d_grady, imgSize, cudaMemcpyDeviceToHost, stream);     // grady
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaMemcpyAsync grady failed: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }

    if (ok) {
        cuErr = cudaStreamSynchronize(stream);
        if (cuErr != cudaSuccess) {
            fprintf(stderr, "[_KLTComputeGradients] cudaStreamSynchronize after copies failed: %s\n", cudaGetErrorString(cuErr));
            ok = false;
        }
    }

    if (ok) {
        // Copy from pinned to outputs
        memcpy(gradx->data, pinned_buf0, imgSize);
        memcpy(grady->data, pinned_buf1, imgSize);
    } else {
        // On failure, zero outputs to be safe
        if (gradx && gradx->data) memset(gradx->data, 0, imgSize);
        if (grady && grady->data) memset(grady->data, 0, imgSize);
    }

    // Cleanup (always run)
    if (d_grady) cudaFree(d_grady);
    if (d_input_copy) cudaFree(d_input_copy);

    if (!ok) {
        fprintf(stderr, "[_KLTComputeGradients] completed with errors (see messages above)\n");
    }
}

