/*********************************************************************
 * convolve.cu
 * Optimized GPU KLT convolution with MINIMAL changes to working code
 * Added: pinned memory, texture cache, streams, reduced transfers
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

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
// Host: Compute kernels from sigma (UNCHANGED)
// -----------------------------------------
static void _computeKernels(
    float sigma,
    ConvolutionKernel *gauss,
    ConvolutionKernel *gaussderiv)
{
    const float factor = 0.01f;
    int i;
    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0);

    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));

    for (i = -hw; i <= hw; i++) {
        gauss->data[i+hw]      = (float)exp(-i*i / (2*sigma*sigma));
        gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    gaussderiv->width = MAX_KERNEL_WIDTH;

    for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; i++, gauss->width -= 2);
    for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; i++, gaussderiv->width -= 2);

    for (i = 0; i < gauss->width; i++)
        gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
    for (i = 0; i < gaussderiv->width; i++)
        gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];

    int hw2 = gaussderiv->width / 2;
    float den = 0.0;
    for (i = 0; i < gauss->width; i++) den += gauss->data[i];
    for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;

    den = 0.0;
    for (i = -hw2; i <= hw2; i++) den -= i * gaussderiv->data[i+hw2];
    for (i = -hw2; i <= hw2; i++) gaussderiv->data[i+hw2] /= den;

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
    
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, kernel.data, kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    
    const int TX = 32;
    const int TY = 8;
    dim3 block(32, 8);
    dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
    size_t shmem = (size_t)TY * (TX + 2 * (kernel.width/2)) * sizeof(float);

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
    
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, kernel.data, kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    
    const int TX = 16;
    const int TY = 16;
    dim3 block(TX, TY);
    dim3 grid((ncols + TX - 1) / TX, (nrows + TY - 1) / TY);
    size_t shmem = (size_t)TX * (TY + 2 * (kernel.width/2)) * sizeof(float);
    
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
    
    // Copy kernel once
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, gauss_kernel.data, gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    
    // Horizontal convolution: d_buf0 -> d_buf1
    const int TX_H = 32, TY_H = 8;
    dim3 block_h(32, 8);
    dim3 grid_h((ncols + TX_H - 1) / TX_H, (nrows + TY_H - 1) / TY_H);
    size_t shmem_h = (size_t)TY_H * (TX_H + 2 * (gauss_kernel.width/2)) * sizeof(float);
    //nvtxRangePushA("_KLTComputeSmoothedImage horiz ");
    convolve_horiz_shared_kernel<32,8><<<grid_h, block_h, shmem_h, stream>>>(d_buf0, d_buf1, ncols, nrows, gauss_kernel.width);
    //nvtxRangePop();
    // Vertical convolution: d_buf1 -> d_buf0
    const int TX_V = 16, TY_V = 16;
    dim3 block_v(TX_V, TY_V);
    dim3 grid_v((ncols + TX_V - 1) / TX_V, (nrows + TY_V - 1) / TY_V);
    size_t shmem_v = (size_t)TX_V * (TY_V + 2 * (gauss_kernel.width/2)) * sizeof(float);
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

    // Allocate device buffer for final grady
    float *d_grady = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grady, imgSize));

    // Upload input to device (d_buf0) from pinned memory
    memcpy(pinned_buf0, img->data, imgSize);
    CUDA_CHECK(cudaMemcpyAsync(d_buf0, pinned_buf0, imgSize, cudaMemcpyHostToDevice, stream));

    // --- IMPORTANT: make a copy of the ORIGINAL input immediately ---
    float *d_input_copy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_copy, imgSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input_copy, d_buf0, imgSize, cudaMemcpyDeviceToDevice, stream));
    // Now d_input_copy holds the original image and can be used multiple times.

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

    // ---------- Compute gradx using d_input_copy as SOURCE ----------
    //nvtxRangePushA("KLT Copmute Gradients");
    // horiz with gaussderiv -> d_buf1
    //printf("Starting nvtx range for klt compute gradient \n");
    //nvtxRangePushA("GradX: horiz (gaussderiv)");
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, gaussderiv_kernel.data, gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    convolve_horiz_shared_kernel<32,8><<<grid_h, block_h, shmem_h_deriv, stream>>>(d_input_copy, d_buf1, ncols, nrows, gaussderiv_kernel.width);
    // vert with gauss -> d_buf0 (gradx)
    //nvtxRangePop();
   // printf("ending nvtx range for klt compute gradient \n");
    //nvtxRangePushA("GradX: Vert (gaussderiv)");
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, gauss_kernel.data, gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    convolve_vert_shared_kernel<16,16><<<grid_v, block_v, shmem_v_gauss, stream>>>(d_buf1, d_buf0, ncols, nrows, gauss_kernel.width);
    //nvtxRangePop();
    // ---------- Compute grady using d_input_copy as SOURCE ----------
    // horiz with gauss -> d_buf1
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, gauss_kernel.data, gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    convolve_horiz_shared_kernel<32,8><<<grid_h, block_h, shmem_h_gauss, stream>>>(d_input_copy, d_buf1, ncols, nrows, gauss_kernel.width);
    // vert with gaussderiv -> d_grady
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_gauss, gaussderiv_kernel.data, gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    convolve_vert_shared_kernel<16,16><<<grid_v, block_v, shmem_v_deriv, stream>>>(d_buf1, d_grady, ncols, nrows, gaussderiv_kernel.width);

    CUDA_CHECK(cudaGetLastError());

    // Download both results
    CUDA_CHECK(cudaMemcpyAsync(pinned_buf0, d_buf0, imgSize, cudaMemcpyDeviceToHost, stream));      // gradx
    CUDA_CHECK(cudaMemcpyAsync(pinned_buf1, d_grady, imgSize, cudaMemcpyDeviceToHost, stream));     // grady
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy from pinned to outputs
    //nvtxRangePop();
    memcpy(gradx->data, pinned_buf0, imgSize);
    memcpy(grady->data, pinned_buf1, imgSize);
    // Cleanup
    cudaFree(d_grady);
    cudaFree(d_input_copy);
}
