/**********************************************************************
 * computeGradientSum_CUDA.cu
 **********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "klt.h"
#include "base.h"

// Minimal struct definition for compatibility with KLT
typedef struct  {
    float *data;
    int ncols;
    int nrows;
} _FloatWindow;


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Bilinear Interpolation on GPU */
__device__ float _interpolateGPU(const float *img, int width, int height, float x, float y) {
    int ix = floorf(x);
    int iy = floorf(y);
    if (ix < 0 || iy < 0 || ix >= width - 1 || iy >= height - 1)
        return 0.0f;
    float a = x - ix;
    float b = y - iy;
    float v00 = img[iy * width + ix];
    float v01 = img[iy * width + ix + 1];
    float v10 = img[(iy + 1) * width + ix];
    float v11 = img[(iy + 1) * width + ix + 1];
    return (1 - a) * (1 - b) * v00 +
           a * (1 - b) * v01 +
           (1 - a) * b * v10 +
           a * b * v11;
}

/* CUDA Kernel */
__global__ void _computeGradientSumKernel(
    const float* gradx1, const float* grady1,
    const float* gradx2, const float* grady2,
    float x1, float y1, float x2, float y2,
    int width, int height, int win_w, int win_h,
    float* gradx_out, float* grady_out)
{
    int i = threadIdx.x - win_w / 2;
    int j = threadIdx.y - win_h / 2;

    float g1x = _interpolateGPU(gradx1, width, height, x1 + i, y1 + j);
    float g2x = _interpolateGPU(gradx2, width, height, x2 + i, y2 + j);
    float g1y = _interpolateGPU(grady1, width, height, x1 + i, y1 + j);
    float g2y = _interpolateGPU(grady2, width, height, x2 + i, y2 + j);

    int idx = (j + win_h / 2) * win_w + (i + win_w / 2);
    gradx_out[idx] = g1x + g2x;
    grady_out[idx] = g1y + g2y;
}

/* GPU Wrapper */
extern "C" void _computeGradientSum_CUDA(
  _KLT_FloatImage gradx1,
  _KLT_FloatImage grady1,
  _KLT_FloatImage gradx2,
  _KLT_FloatImage grady2,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  _FloatWindow gradx,
  _FloatWindow grady)
{
    const float *img_gradx1 = gradx1->data;
    const float *img_grady1 = grady1->data;
    const float *img_gradx2 = gradx2->data;
    const float *img_grady2 = grady2->data;
    float *gradx_out = gradx.data;
    float *grady_out = grady.data;

    
    int ncols = gradx1->ncols;
    int nrows = gradx1->nrows;
    int win_w = gradx.ncols;
    int win_h = gradx.nrows;

    size_t img_bytes = ncols * nrows * sizeof(float);
    size_t out_bytes = win_w * win_h * sizeof(float);

    float *d_gradx1, *d_grady1, *d_gradx2, *d_grady2;
    float *d_gradx_out, *d_grady_out;

    CUDA_CHECK(cudaMalloc(&d_gradx1, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_grady1, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_gradx2, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_grady2, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_gradx_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_grady_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_gradx1, img_gradx1, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grady1, img_grady1, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gradx2, img_gradx2, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grady2, img_grady2, img_bytes, cudaMemcpyHostToDevice));

    dim3 block(win_w, win_h);
    dim3 grid(1);

    _computeGradientSumKernel<<<grid, block>>>(d_gradx1, d_grady1, d_gradx2, d_grady2,
                                               x1, y1, x2, y2, ncols, nrows, win_w, win_h,
                                               d_gradx_out, d_grady_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gradx_out, d_gradx_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grady_out, d_grady_out, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_gradx1); cudaFree(d_grady1);
    cudaFree(d_gradx2); cudaFree(d_grady2);
    cudaFree(d_gradx_out); cudaFree(d_grady_out);
}
