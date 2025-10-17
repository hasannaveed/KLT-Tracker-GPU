/*********************************************************************
 * convolve.cu
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Our includes */
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

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * GPU Kernels (no extern "C")
 *********************************************************************/
__global__ void convolveImageHorizKernel(
    const float* imgin, float* imgout,
    const float* kernel, int kwidth,
    int ncols, int nrows)
{
  printf("Working on gpu succesfully\n");
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ncols || y >= nrows) return;

    int radius = kwidth / 2;
    float sum = 0.0f;

    if (x < radius || x >= ncols - radius) {
        imgout[y * ncols + x] = 0.0f;
        return;
    }

    for (int k = -radius; k <= radius; ++k)
        sum += imgin[y * ncols + (x + k)] * kernel[radius - k];

    imgout[y * ncols + x] = sum;
}

__global__ void convolveImageVertKernel(
    const float* imgin, float* imgout,
    const float* kernel, int kwidth,
    int ncols, int nrows)
{
  printf("Working on gpu succesfully\n");
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ncols || y >= nrows) return;

    int radius = kwidth / 2;
    float sum = 0.0f;

    if (y < radius || y >= nrows - radius) {
        imgout[y * ncols + x] = 0.0f;
        return;
    }

    for (int k = -radius; k <= radius; ++k)
        sum += imgin[(y + k) * ncols + x] * kernel[radius - k];

    imgout[y * ncols + x] = sum;
}

/*********************************************************************
 * Host Functions (must be extern "C")
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

extern "C" void _convolveImageHorizGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
  printf("Working on gpu succesfully\n");
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int imgSize = ncols * nrows * sizeof(float);
  int kSize = kernel.width * sizeof(float);

  float *d_imgin, *d_imgout, *d_kernel;
  cudaMalloc(&d_imgin, imgSize);
  cudaMalloc(&d_imgout, imgSize);
  cudaMalloc(&d_kernel, kSize);

  cudaMemcpy(d_imgin, imgin->data, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

  convolveImageHorizKernel<<<grid, block>>>(d_imgin, d_imgout, d_kernel, kernel.width, ncols, nrows);
  cudaMemcpy(imgout->data, d_imgout, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_imgin);
  cudaFree(d_imgout);
  cudaFree(d_kernel);
  cudaDeviceSynchronize();
}

extern "C" void _convolveImageVertGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int imgSize = ncols * nrows * sizeof(float);
  int kSize = kernel.width * sizeof(float);

  float *d_imgin, *d_imgout, *d_kernel;
  cudaMalloc(&d_imgin, imgSize);
  cudaMalloc(&d_imgout, imgSize);
  cudaMalloc(&d_kernel, kSize);

  cudaMemcpy(d_imgin, imgin->data, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

  convolveImageVertKernel<<<grid, block>>>(d_imgin, d_imgout, d_kernel, kernel.width, ncols, nrows);
  cudaMemcpy(imgout->data, d_imgout, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_imgin);
  cudaFree(d_imgout);
  cudaFree(d_kernel);
  cudaDeviceSynchronize();
}

extern "C" void _KLTComputeGradients(_KLT_FloatImage img, float sigma, _KLT_FloatImage gradx, _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveImageHorizGPU(img, gaussderiv_kernel, gradx);
  _convolveImageVertGPU(img, gauss_kernel, grady);
}

extern "C" void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma, _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveImageHorizGPU(img, gauss_kernel, smooth);
  _convolveImageVertGPU(smooth, gauss_kernel, smooth);
}
