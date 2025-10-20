/**********************************************************************
 * computeGradientSum_CUDA.cu
 **********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "klt.h"
#include "base.h"

// _FloatWindow is just a float pointer
typedef float *_FloatWindow;

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

/* CUDA Kernel using 1D threads */
__global__ void _computeGradientSumKernel(
    const float* gradx1, const float* grady1,
    const float* gradx2, const float* grady2,
    float x1, float y1, float x2, float y2,
    int img_width, int img_height,
    int win_w, int win_h,
    float* gradx_out, float* grady_out)
{
    //printf("GPU riunning compute gradient sum kernel\n");
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= win_w * win_h) return;

    int i = idx % win_w - win_w / 2;
    int j = idx / win_w - win_h / 2;

    float gx1 = _interpolateGPU(gradx1, img_width, img_height, x1 + i, y1 + j);
    float gx2 = _interpolateGPU(gradx2, img_width, img_height, x2 + i, y2 + j);
    float gy1 = _interpolateGPU(grady1, img_width, img_height, x1 + i, y1 + j);
    float gy2 = _interpolateGPU(grady2, img_width, img_height, x2 + i, y2 + j);

    gradx_out[idx] = gx1 + gx2;
    grady_out[idx] = gy1 + gy2;
}

/* GPU Wrapper */
extern "C" void _computeGradientSum_CUDA(
    _KLT_FloatImage gradx1,
    _KLT_FloatImage grady1,
    _KLT_FloatImage gradx2,
    _KLT_FloatImage grady2,
    float x1, float y1,
    float x2, float y2,
    int win_w, int win_h,     // size of window
    _FloatWindow gradx,       // output pointers
    _FloatWindow grady)
{
    const float *img_gradx1 = gradx1->data;
    const float *img_grady1 = grady1->data;
    const float *img_gradx2 = gradx2->data;
    const float *img_grady2 = grady2->data;

    int img_w = gradx1->ncols;
    int img_h = gradx1->nrows;

    size_t img_bytes = img_w * img_h * sizeof(float);
    size_t win_bytes = win_w * win_h * sizeof(float);

    float *d_gradx1, *d_grady1, *d_gradx2, *d_grady2;
    float *d_gradx_out, *d_grady_out;

    CUDA_CHECK(cudaMalloc(&d_gradx1, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_grady1, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_gradx2, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_grady2, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_gradx_out, win_bytes));
    CUDA_CHECK(cudaMalloc(&d_grady_out, win_bytes));

    CUDA_CHECK(cudaMemcpy(d_gradx1, img_gradx1, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grady1, img_grady1, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gradx2, img_gradx2, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grady2, img_grady2, img_bytes, cudaMemcpyHostToDevice));

    int threads = 1024;
    int blocks = (win_w * win_h + threads - 1) / threads;

    //printf("[CUDA] Window size: (%d x %d)\n", win_w, win_h);
    //printf("[CUDA] Launch config: blocks=%d, threads=%d\n", blocks, threads);
    fflush(stdout);

    _computeGradientSumKernel<<<blocks, threads>>>(
        d_gradx1, d_grady1, d_gradx2, d_grady2,
        x1, y1, x2, y2,
        img_w, img_h,
        win_w, win_h,
        d_gradx_out, d_grady_out);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gradx, d_gradx_out, win_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grady, d_grady_out, win_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_gradx1); cudaFree(d_grady1);
    cudaFree(d_gradx2); cudaFree(d_grady2);
    cudaFree(d_gradx_out); cudaFree(d_grady_out);
}
