/*********************************************************************
 * convolve.cu
 * CUDA error-checked version
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C" {
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"
}

#define MAX_KERNEL_WIDTH 71

typedef struct {
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;

// CUDA error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*********************************************************************
 * GPU Kernels
 *********************************************************************/
__global__ void convolveImageHorizKernel(
    const float* imgin, float* imgout,
    const float* kernel, int kwidth,
    int ncols, int nrows)
{
    //printf("[GPU]convolve horz runnning      ");
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ncols || y >= nrows) return;

    int radius = kwidth / 2;
    float sum = 0.0f;

    // Left/right boundaries
    if (x < radius || x >= ncols - radius) {
        imgout[y * ncols + x] = 0.0f;
        return;
    }

    // Horizontal convolution with reversed kernel (matches CPU)
    for (int k = -radius; k <= radius; ++k) {
        int pixelIdx = y * ncols + (x + k);
        int kernelIdx = kwidth - 1 - (radius + k);  // reverse kernel
        sum += imgin[pixelIdx] * kernel[kernelIdx];
    }

    imgout[y * ncols + x] = sum;
}


__global__ void convolveImageVertKernel(
    const float* imgin, float* imgout,
    const float* kernel, int kwidth,
    int ncols, int nrows)
{
    //printf("[GPU]convolve vert runnning      ");
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ncols || y >= nrows) return;

    int radius = kwidth / 2;
    float sum = 0.0f;

    // Top/bottom boundaries
    if (y < radius || y >= nrows - radius) {
        imgout[y * ncols + x] = 0.0f;
        return;
    }

    // Vertical convolution with reversed kernel (matches CPU)
    for (int k = -radius; k <= radius; ++k) {
        int pixelIdx = (y + k) * ncols + x;
        int kernelIdx = kwidth - 1 - (radius + k);  // reverse kernel
        sum += imgin[pixelIdx] * kernel[kernelIdx];
    }

    imgout[y * ncols + x] = sum;
}


/*********************************************************************
 * Host Functions
 *********************************************************************/
extern "C" void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
    KLT_PixelType *ptrend = img + ncols*nrows;
    float *ptrout = floatimg->data;

    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    while (img < ptrend) *ptrout++ = (float)*img++;
}

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
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * GPU Convolution Wrappers with error checking
 *********************************************************************/
extern "C" void _convolveImageHorizGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int imgSize = ncols * nrows * sizeof(float);
    int kSize = kernel.width * sizeof(float);

    float *d_imgin = nullptr, *d_imgout = nullptr, *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_imgin, imgSize));
    CUDA_CHECK(cudaMalloc(&d_imgout, imgSize));
    CUDA_CHECK(cudaMalloc(&d_kernel, kSize));

    CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, imgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

    convolveImageHorizKernel<<<grid, block>>>(d_imgin, d_imgout, d_kernel, kernel.width, ncols, nrows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, imgSize, cudaMemcpyDeviceToHost));

    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
}

extern "C" void _convolveImageVertGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int imgSize = ncols * nrows * sizeof(float);
    int kSize = kernel.width * sizeof(float);

    float *d_imgin = nullptr, *d_imgout = nullptr, *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_imgin, imgSize));
    CUDA_CHECK(cudaMalloc(&d_imgout, imgSize));
    CUDA_CHECK(cudaMalloc(&d_kernel, kSize));

    CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, imgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

    convolveImageVertKernel<<<grid, block>>>(d_imgin, d_imgout, d_kernel, kernel.width, ncols, nrows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, imgSize, cudaMemcpyDeviceToHost));

    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
}

/*********************************************************************
 * Gradient and smoothing functions
 *********************************************************************/
// Fixed GPU Compute Gradients
extern "C" void _KLTComputeGradients(_KLT_FloatImage img, float sigma,
                                     _KLT_FloatImage gradx, _KLT_FloatImage grady)
{
    assert(gradx->ncols >= img->ncols && gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols && grady->nrows >= img->nrows);

    // Recompute kernels if sigma changed
    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    // Temporary buffer for horizontal pass
    _KLT_FloatImage tmp = _KLTCreateFloatImage(img->ncols, img->nrows);

    // gradx = dG/dx * img
    _convolveImageHorizGPU(img, gaussderiv_kernel, tmp);
    _convolveImageVertGPU(tmp, gauss_kernel, gradx);

    // grady = dG/dy * img
    _convolveImageHorizGPU(img, gauss_kernel, tmp);
    _convolveImageVertGPU(tmp, gaussderiv_kernel, grady);

    // Free temporary buffer
    _KLTFreeFloatImage(tmp);
}

// Fixed GPU Compute Smoothed Image
extern "C" void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma,
                                         _KLT_FloatImage smooth)
{
    assert(smooth->ncols >= img->ncols && smooth->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _KLT_FloatImage tmp = _KLTCreateFloatImage(img->ncols, img->nrows);

    // smooth = G * img
    _convolveImageHorizGPU(img, gauss_kernel, tmp);
    _convolveImageVertGPU(tmp, gauss_kernel, smooth);

    _KLTFreeFloatImage(tmp);
}
