/*********************************************************************
 * convolve_openacc.c
 * Optimized OpenACC KLT convolution
 * Fixed: Proper kernel data management on device
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

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
// Device buffers (managed by OpenACC)
// -----------------------------------------
static float *d_buf0 = NULL;
static float *d_buf1 = NULL;
static float *d_grady = NULL;
static int device_ncols = 0;
static int device_nrows = 0;

// Kernel data on device - persistent allocation
static float *d_kernel_data = NULL;
static int d_kernel_allocated = 0;

// -----------------------------------------
// Buffer management
// -----------------------------------------
static void ensureDeviceBuffers(int ncols, int nrows) {
    size_t elems = (size_t)ncols * (size_t)nrows;
    
    if (d_buf0 && device_ncols == ncols && device_nrows == nrows) return;

    // Free old buffers
    if (d_buf0) {
        #pragma acc exit data delete(d_buf0[0:device_ncols*device_nrows])
        #pragma acc exit data delete(d_buf1[0:device_ncols*device_nrows])
        free(d_buf0);
        free(d_buf1);
        d_buf0 = d_buf1 = NULL;
    }

    d_buf0 = (float*)malloc(elems * sizeof(float));
    d_buf1 = (float*)malloc(elems * sizeof(float));
    
    if (!d_buf0 || !d_buf1) {
        fprintf(stderr, "Failed to allocate device buffers\n");
        exit(1);
    }

    #pragma acc enter data create(d_buf0[0:elems])
    #pragma acc enter data create(d_buf1[0:elems])
    
    device_ncols = ncols;
    device_nrows = nrows;
}

static void ensureKernelBuffer() {
    if (d_kernel_allocated) return;
    
    d_kernel_data = (float*)malloc(MAX_KERNEL_WIDTH * sizeof(float));
    if (!d_kernel_data) {
        fprintf(stderr, "Failed to allocate kernel buffer\n");
        exit(1);
    }
    
    #pragma acc enter data create(d_kernel_data[0:MAX_KERNEL_WIDTH])
    d_kernel_allocated = 1;
}

void _KLTFreeDeviceBuffers() {
    if (d_buf0) {
        #pragma acc exit data delete(d_buf0[0:device_ncols*device_nrows])
        #pragma acc exit data delete(d_buf1[0:device_ncols*device_nrows])
        free(d_buf0);
        free(d_buf1);
        d_buf0 = d_buf1 = NULL;
    }
    
    if (d_grady) {
        #pragma acc exit data delete(d_grady[0:device_ncols*device_nrows])
        free(d_grady);
        d_grady = NULL;
    }
    
    if (d_kernel_allocated) {
        #pragma acc exit data delete(d_kernel_data[0:MAX_KERNEL_WIDTH])
        free(d_kernel_data);
        d_kernel_data = NULL;
        d_kernel_allocated = 0;
    }
    
    device_ncols = device_nrows = 0;
}

// -----------------------------------------
// Host: Convert KLT image to float
// -----------------------------------------
void _KLTToFloatImage(
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
// Host: Compute kernels from sigma
// -----------------------------------------
static void _computeKernels(
    float sigma,
    ConvolutionKernel *gauss,
    ConvolutionKernel *gaussderiv)
{
    const float factor = 0.01f;
    int i;

    if (sigma < 1e-3f) sigma = 1e-3f;

    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0f);

    for (i = 0; i < MAX_KERNEL_WIDTH; ++i) {
        gauss->data[i] = 0.0f;
        gaussderiv->data[i] = 0.0f;
    }
    gauss->width = MAX_KERNEL_WIDTH;
    gaussderiv->width = MAX_KERNEL_WIDTH;

    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 0.0f, max_gaussderiv = 0.0f;

    for (i = -hw; i <= hw; i++) {
        float g = (float)exp(-(float)(i*i) / (2.0f * sigma * sigma));
        float gd = -i * g;
        gauss->data[i+hw] = g;
        gaussderiv->data[i+hw] = gd;
        if (g > max_gauss) max_gauss = g;
        if (fabsf(gd) > max_gaussderiv) max_gaussderiv = fabsf(gd);
    }

    // Trim gauss
    gauss->width = MAX_KERNEL_WIDTH;
    while (gauss->width > 1) {
        int left_idx = (MAX_KERNEL_WIDTH - gauss->width)/2;
        float denom = (max_gauss == 0.0f) ? 1.0f : max_gauss;
        if (fabsf(gauss->data[left_idx] / denom) >= factor) break;
        gauss->width -= 2;
    }
    
    // Trim gaussderiv
    gaussderiv->width = MAX_KERNEL_WIDTH;
    while (gaussderiv->width > 1) {
        int left_idx = (MAX_KERNEL_WIDTH - gaussderiv->width)/2;
        float denom = (max_gaussderiv == 0.0f) ? 1.0f : max_gaussderiv;
        if (fabsf(gaussderiv->data[left_idx] / denom) >= factor) break;
        gaussderiv->width -= 2;
    }

    // Compact arrays
    int start_g = (MAX_KERNEL_WIDTH - gauss->width) / 2;
    int start_gd = (MAX_KERNEL_WIDTH - gaussderiv->width) / 2;
    for (i = 0; i < gauss->width; ++i) {
        gauss->data[i] = gauss->data[start_g + i];
    }
    for (i = 0; i < gaussderiv->width; ++i) {
        gaussderiv->data[i] = gaussderiv->data[start_gd + i];
    }

    // Normalize gauss
    float den = 0.0f;
    for (i = 0; i < gauss->width; ++i) den += gauss->data[i];
    if (den == 0.0f) den = 1.0f;
    for (i = 0; i < gauss->width; ++i) gauss->data[i] /= den;

    // Normalize gaussderiv
    int hw2 = gaussderiv->width / 2;
    den = 0.0f;
    for (i = -hw2; i <= hw2; i++) den -= i * gaussderiv->data[i + hw2];
    if (den == 0.0f) den = 1.0f;
    for (i = -hw2; i <= hw2; i++) gaussderiv->data[i + hw2] /= den;

    sigma_last = sigma;
}

void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width)
{
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    if (gauss_width) *gauss_width = gauss_kernel.width;
    if (gaussderiv_width) *gaussderiv_width = gaussderiv_kernel.width;
}

// -----------------------------------------
// OpenACC: Horizontal convolution
// -----------------------------------------
static void _convolveImageHorizGPU(
    float *in, float *out,
    int ncols, int nrows,
    float *kern, int kwidth)
{
    int radius = kwidth / 2;

    // Use copyin for kernel since it's small and changes between calls
    #pragma acc parallel loop gang vector_length(32) \
        copyin(kern[0:kwidth]) \
        present(in[0:ncols*nrows], out[0:ncols*nrows])
    for (int j = 0; j < nrows; j++) {
        #pragma acc loop vector
        for (int i = 0; i < ncols; i++) {
            if (i < radius || i >= ncols - radius) {
                out[j * ncols + i] = 0.0f;
            } else {
                float sum = 0.0f;
                #pragma acc loop seq
                for (int k = 0; k < kwidth; k++) {
                    int idx = j * ncols + (i - radius + k);
                    sum += in[idx] * kern[kwidth - 1 - k];
                }
                out[j * ncols + i] = sum;
            }
        }
    }
}

// -----------------------------------------
// OpenACC: Vertical convolution
// -----------------------------------------
static void _convolveImageVertGPU(
    float *in, float *out,
    int ncols, int nrows,
    float *kern, int kwidth)
{
    int radius = kwidth / 2;
    
    #pragma acc parallel loop gang vector_length(32) \
        copyin(kern[0:kwidth]) \
        present(in[0:ncols*nrows], out[0:ncols*nrows])
    for (int i = 0; i < ncols; i++) {
        #pragma acc loop vector
        for (int j = 0; j < nrows; j++) {
            if (j < radius || j >= nrows - radius) {
                out[j * ncols + i] = 0.0f;
            } else {
                float sum = 0.0f;
                #pragma acc loop seq
                for (int k = 0; k < kwidth; k++) {
                    int idx = (j - radius + k) * ncols + i;
                    sum += in[idx] * kern[kwidth - 1 - k];
                }
                out[j * ncols + i] = sum;
            }
        }
    }
}

// -----------------------------------------
// OpenACC: Compute smoothed image
// -----------------------------------------
void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth)
{
    if (fabsf(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    
    int ncols = img->ncols;
    int nrows = img->nrows;
    size_t elems = (size_t)ncols * (size_t)nrows;
    
    ensureDeviceBuffers(ncols, nrows);
    ensureKernelBuffer();
    
    // Copy input to d_buf0
    memcpy(d_buf0, img->data, elems * sizeof(float));
    #pragma acc update device(d_buf0[0:elems])
    
    memcpy(d_kernel_data, gauss_kernel.data, gauss_kernel.width * sizeof(float));
    #pragma acc update device(d_kernel_data[0:gauss_kernel.width])
    
    _convolveImageHorizGPU(d_buf0, d_buf1, ncols, nrows, 
                          d_kernel_data, gauss_kernel.width);
    
    _convolveImageVertGPU(d_buf1, d_buf0, ncols, nrows,
                         d_kernel_data, gauss_kernel.width);
    
    #pragma acc update host(d_buf0[0:elems])
    memcpy(smooth->data, d_buf0, elems * sizeof(float));
}

// -----------------------------------------
// OpenACC: Compute gradients
// -----------------------------------------
void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{
    assert(gradx->ncols >= img->ncols && gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols && grady->nrows >= img->nrows);

    if (fabsf(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    int ncols = img->ncols;
    int nrows = img->nrows;
    size_t elems = (size_t)ncols * (size_t)nrows;
    
    ensureDeviceBuffers(ncols, nrows);
    ensureKernelBuffer();
    
    // Allocate d_grady if needed
    if (!d_grady) {
        d_grady = (float*)malloc(elems * sizeof(float));
        #pragma acc enter data create(d_grady[0:elems])
    }
    
    // Upload input image to d_buf0
    memcpy(d_buf0, img->data, elems * sizeof(float));
    #pragma acc update device(d_buf0[0:elems])
    
    // Horizontal with gaussderiv: d_buf0 -> d_buf1
    memcpy(d_kernel_data, gaussderiv_kernel.data, gaussderiv_kernel.width * sizeof(float));
    #pragma acc update device(d_kernel_data[0:gaussderiv_kernel.width])
    
    _convolveImageHorizGPU(d_buf0, d_buf1, ncols, nrows,
                          d_kernel_data, gaussderiv_kernel.width);
    
    // Vertical with gauss: d_buf1 -> d_buf0 (gradx result)
    memcpy(d_kernel_data, gauss_kernel.data, gauss_kernel.width * sizeof(float));
    #pragma acc update device(d_kernel_data[0:gauss_kernel.width])
    
    _convolveImageVertGPU(d_buf1, d_buf0, ncols, nrows,
                         d_kernel_data, gauss_kernel.width);
    
    // Re-upload original input to d_buf1
    memcpy(d_buf1, img->data, elems * sizeof(float));
    #pragma acc update device(d_buf1[0:elems])
    
    float *temp_buf = d_buf1; // input
    
    float *temp_horiz = (float*)malloc(elems * sizeof(float));
    #pragma acc enter data create(temp_horiz[0:elems])
    
    _convolveImageHorizGPU(temp_buf, temp_horiz, ncols, nrows,
                          d_kernel_data, gauss_kernel.width);
    
    // Vertical with gaussderiv: temp_horiz -> d_grady (grady result)
    memcpy(d_kernel_data, gaussderiv_kernel.data, gaussderiv_kernel.width * sizeof(float));
    #pragma acc update device(d_kernel_data[0:gaussderiv_kernel.width])
    
    _convolveImageVertGPU(temp_horiz, d_grady, ncols, nrows,
                         d_kernel_data, gaussderiv_kernel.width);
    
    // Download results
    #pragma acc update host(d_buf0[0:elems])
    #pragma acc update host(d_grady[0:elems])
    
    memcpy(gradx->data, d_buf0, elems * sizeof(float));
    memcpy(grady->data, d_grady, elems * sizeof(float));
    
    // Clean up temp buffer
    #pragma acc exit data delete(temp_horiz[0:elems])
    free(temp_horiz);
}