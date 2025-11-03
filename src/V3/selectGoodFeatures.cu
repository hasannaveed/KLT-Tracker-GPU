/*********************************************************************
 * selectGoodFeatures_optimized.cu - Further optimized GPU version of KLTSelectGoodFeatures
 
 *********************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define fsqrt(X) sqrt(X)

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

int KLT_verbose = 1;
typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------- */
/* Quicksort for pointlist (3 integers per point) */
#define SWAP3(list, i, j)               \
{register int *pi, *pj, tmp;            \
     pi=list+3*(i); pj=list+3*(j);      \
     tmp=*pi;    *pi++=*pj;  *pj++=tmp; \
     tmp=*pi;    *pi++=*pj;  *pj++=tmp; \
     tmp=*pi;    *pi=*pj;    *pj=tmp;   \
}

void _quicksort(int *pointlist, int n)
{
  unsigned int i, j, ln, rn;

  while (n > 1)
  {
    SWAP3(pointlist, 0, n/2);
    for (i = 0, j = n; ; )
    {
      do --j; while (pointlist[3*j+2] < pointlist[2]);
      do ++i; while (i < j && pointlist[3*i+2] > pointlist[2]);
      if (i >= j) break;
      SWAP3(pointlist, i, j);
    }
    SWAP3(pointlist, j, 0);
    ln = j;
    rn = n - ++j;
    if (ln < rn) { _quicksort(pointlist, ln); pointlist += 3*j; n = rn; }
    else { _quicksort(pointlist + 3*j, rn); n = ln; }
  }
}
#undef SWAP3

/* ------------------------------------------------------------------- */
static void _fillFeaturemap(int x, int y, uchar *featuremap, int mindist, int ncols, int nrows)
{
  int ix, iy;
  for (iy = y - mindist ; iy <= y + mindist ; iy++)
    for (ix = x - mindist ; ix <= x + mindist ; ix++)
      if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        featuremap[iy*ncols+ix] = 1;
}

/* ------------------------------------------------------------------- */
static void _enforceMinimumDistance(
  int *pointlist, int npoints, KLT_FeatureList featurelist,
  int ncols, int nrows, int mindist, int min_eigenvalue, KLT_BOOL overwriteAllFeatures)
{
  int indx, x, y, val;
  uchar *featuremap;
  int *ptr;
	
  if (min_eigenvalue < 1)  min_eigenvalue = 1;
  featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
  memset(featuremap, 0, ncols*nrows);
  mindist--;

  if (!overwriteAllFeatures)
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
      if (featurelist->feature[indx]->val >= 0)
        _fillFeaturemap((int)featurelist->feature[indx]->x,
                        (int)featurelist->feature[indx]->y,
                        featuremap, mindist, ncols, nrows);

  ptr = pointlist;
  indx = 0;
  while (1)
  {
    if (ptr >= pointlist + 3*npoints) {
      while (indx < featurelist->nFeatures) {
        if (overwriteAllFeatures || featurelist->feature[indx]->val < 0) {
          featurelist->feature[indx]->x   = -1;
          featurelist->feature[indx]->y   = -1;
          featurelist->feature[indx]->val = KLT_NOT_FOUND;
          featurelist->feature[indx]->aff_img = NULL;
          featurelist->feature[indx]->aff_img_gradx = NULL;
          featurelist->feature[indx]->aff_img_grady = NULL;
          featurelist->feature[indx]->aff_x = -1.0;
          featurelist->feature[indx]->aff_y = -1.0;
          featurelist->feature[indx]->aff_Axx = 1.0;
          featurelist->feature[indx]->aff_Ayx = 0.0;
          featurelist->feature[indx]->aff_Axy = 0.0;
          featurelist->feature[indx]->aff_Ayy = 1.0;
        }
        indx++;
      }
      break;
    }

    x   = *ptr++;
    y   = *ptr++;
    val = *ptr++;

    assert(x >= 0 && x < ncols && y >= 0 && y < nrows);

    while (!overwriteAllFeatures && indx < featurelist->nFeatures && featurelist->feature[indx]->val >= 0)
      indx++;
    if (indx >= featurelist->nFeatures) break;

    if (!featuremap[y*ncols+x] && val >= min_eigenvalue) {
      featurelist->feature[indx]->x   = (KLT_locType) x;
      featurelist->feature[indx]->y   = (KLT_locType) y;
      featurelist->feature[indx]->val = val;
      featurelist->feature[indx]->aff_img = NULL;
      featurelist->feature[indx]->aff_img_gradx = NULL;
      featurelist->feature[indx]->aff_img_grady = NULL;
      featurelist->feature[indx]->aff_x = -1.0;
      featurelist->feature[indx]->aff_y = -1.0;
      featurelist->feature[indx]->aff_Axx = 1.0;
      featurelist->feature[indx]->aff_Ayx = 0.0;
      featurelist->feature[indx]->aff_Axy = 0.0;
      featurelist->feature[indx]->aff_Ayy = 1.0;
      indx++;
      _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
    }
  }

  free(featuremap);
}

/* ------------------------------------------------------------------- */
#ifdef KLT_USE_QSORT
static int _comparePoints(const void *a, const void *b)
{
  int v1 = *(((int *) a) + 2);
  int v2 = *(((int *) b) + 2);
  return (v1 > v2) ? -1 : (v1 < v2) ? 1 : 0;
}
#endif

static void _sortPointList(int *pointlist, int npoints)
{
#ifdef KLT_USE_QSORT
  qsort(pointlist, npoints, 3*sizeof(int), _comparePoints);
#else
  _quicksort(pointlist, npoints);
#endif
}

/* ------------------------------------------------------------------- */
static float _minEigenvalue(float gxx, float gxy, float gyy)
{
  return 0.5f * (gxx + gyy - sqrtf((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy));
}

/* ------------------------------------------------------------------- */
/* GPU includes */
#include <cuda_runtime.h>

/* GPU min eigenvalue (device) */
__device__ float minEigenvalueGPU(float gxx, float gxy, float gyy)
{
  return 0.5f * (gxx + gyy - sqrtf((gxx - gyy)*(gxx - gyy) + 4.0f*gxy*gxy));
}

/* -------------------------------------------------------------------
 * Kernel choices and occupancy tuning:
 * We'll select a block size at runtime using occupancy API. For the
 * tiled kernel we still require shared memory allocation passed at launch.
 *
 * We retain a 2D block mapping (block.x, block.y) for coalescing and
 * warp-friendliness, but compute block.y from chosen block.x when needed.
 */

/* Default block shape used as fallback */
#define DEFAULT_BLOCK_X 32
#define DEFAULT_BLOCK_Y 8

/* Shared-memory tiled kernel (stages tile + halo into shared mem and computes using shared mem) */
__global__ void computeTrackabilityKernel_tex_shared(
    cudaTextureObject_t gradxTex,
    cudaTextureObject_t gradyTex,
    int ncols, int nrows,
    int window_hw, int window_hh,
    int skipPixels,
    int borderx, int bordery,
    float* eigvals,
    int tile_w, int tile_h
) {
    // Global pixel indices for this thread (pixel to compute)
    int gx = blockIdx.x * blockDim.x + threadIdx.x + borderx;
    int gy = blockIdx.y * blockDim.y + threadIdx.y + bordery;

    // dynamic shared memory passed by caller: s_gradx (tile_w*tile_h) then s_grady (tile_w*tile_h)
    extern __shared__ float s_mem[];
    float* s_gradx = s_mem;
    float* s_grady = s_mem + (size_t)tile_w * (size_t)tile_h;

    // Tile origin (global coords) including halo
    int tile_origin_x = blockIdx.x * blockDim.x + borderx - window_hw;
    int tile_origin_y = blockIdx.y * blockDim.y + bordery - window_hh;

    // cooperative load: each thread strides across tile; precompute clamps to avoid branching inside inner loops
    for (int dy = threadIdx.y; dy < tile_h; dy += blockDim.y) {
        int global_y = tile_origin_y + dy;
        int gy_clamped = global_y;
        if (gy_clamped < 0) gy_clamped = 0;
        if (gy_clamped > nrows - 1) gy_clamped = nrows - 1;
        int rowBase = dy * tile_w;
        for (int dx = threadIdx.x; dx < tile_w; dx += blockDim.x) {
            int global_x = tile_origin_x + dx;
            int gx_clamped = global_x;
            if (gx_clamped < 0) gx_clamped = 0;
            if (gx_clamped > ncols - 1) gx_clamped = ncols - 1;
            // fetch once per position
            float gxv = tex2D<float>(gradxTex, (float)gx_clamped + 0.5f, (float)gy_clamped + 0.5f);
            float gyv = tex2D<float>(gradyTex, (float)gx_clamped + 0.5f, (float)gy_clamped + 0.5f);
            int idx = rowBase + dx;
            s_gradx[idx] = gxv;
            s_grady[idx] = gyv;
        }
    }

    __syncthreads();

    // Bounds & skip (done after tile is ready, reduces redundant checks in cooperative loads)
    if (gx >= ncols - borderx || gy >= nrows - bordery) return;
    if ((gx - borderx) % (skipPixels + 1) != 0 || (gy - bordery) % (skipPixels + 1) != 0) return;

    // local coordinates inside shared tile for the pixel
    int local_x = threadIdx.x + window_hw;
    int local_y = threadIdx.y + window_hh;

    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;

    // Unroll small windows; window size often small so unroll gives benefit
    #pragma unroll 4
    for (int yy = -window_hh; yy <= window_hh; yy++) {
        int row = (local_y + yy) * tile_w;
        #pragma unroll 4
        for (int xx = -window_hw; xx <= window_hw; xx++) {
            int idx = row + (local_x + xx);
            float gx_val = s_gradx[idx];
            float gy_val = s_grady[idx];
            gxx += gx_val * gx_val;
            gxy += gx_val * gy_val;
            gyy += gy_val * gy_val;
        }
    }

    int outIdx = gy * ncols + gx;
    eigvals[outIdx] = minEigenvalueGPU(gxx, gxy, gyy);
}

/* ------------------------------------------------------------------- */
/* No-shared kernel: samples texture directly for each neighbor (uses texture cache heavily) */
__global__ void computeTrackabilityKernel_tex_noshared(
    cudaTextureObject_t gradxTex,
    cudaTextureObject_t gradyTex,
    int ncols, int nrows,
    int window_hw, int window_hh,
    int skipPixels,
    int borderx, int bordery,
    float* eigvals
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x + borderx;
    int gy = blockIdx.y * blockDim.y + threadIdx.y + bordery;

    if (gx >= ncols - borderx || gy >= nrows - bordery) return;
    if ((gx - borderx) % (skipPixels + 1) != 0 || (gy - bordery) % (skipPixels + 1) != 0) return;

    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    #pragma unroll 4
    for (int yy = -window_hh; yy <= window_hh; yy++) {
        int ry = gy + yy;
        int ry_clamped = ry;
        if (ry_clamped < 0) ry_clamped = 0;
        if (ry_clamped > nrows - 1) ry_clamped = nrows - 1;
        #pragma unroll 4
        for (int xx = -window_hw; xx <= window_hw; xx++) {
            int rx = gx + xx;
            int rx_clamped = rx;
            if (rx_clamped < 0) rx_clamped = 0;
            if (rx_clamped > ncols - 1) rx_clamped = ncols - 1;
            float gxv = tex2D<float>(gradxTex, (float)rx_clamped + 0.5f, (float)ry_clamped + 0.5f);
            float gyv = tex2D<float>(gradyTex, (float)rx_clamped + 0.5f, (float)ry_clamped + 0.5f);
            gxx += gxv * gxv;
            gxy += gxv * gyv;
            gyy += gyv * gyv;
        }
    }
    int outIdx = gy * ncols + gx;
    eigvals[outIdx] = minEigenvalueGPU(gxx, gxy, gyy);
}

/* ------------------------------------------------------------------- */
/* Optimized GPU wrapper with textures, pinned mem, streams, and occupancy-aware kernel selection */
void _KLTSelectGoodFeaturesGPU(
    KLT_TrackingContext tc,
    KLT_PixelType *img,
    int ncols,
    int nrows,
    KLT_FeatureList featurelist,
    selectionMode mode
) {
    int window_hw = tc->window_width / 2;
    int window_hh = tc->window_height / 2;

    /* Allocate CPU pinned memory for float image & gradients so memcpy to device is fast */
    _KLT_FloatImage floatimg = _KLTCreateFloatImage(ncols, nrows);

    // Allocate pinned host memory for gradients (page-locked)
    float *gradx_host = NULL;
    float *grady_host = NULL;
    size_t img_size_bytes = (size_t)ncols * (size_t)nrows * sizeof(float);

    if (cudaHostAlloc((void**)&gradx_host, img_size_bytes, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc gradx_host failed - falling back to malloc\n");
        gradx_host = (float*) malloc(img_size_bytes);
    }
    if (cudaHostAlloc((void**)&grady_host, img_size_bytes, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc grady_host failed - falling back to malloc\n");
        grady_host = (float*) malloc(img_size_bytes);
    }

    _KLT_FloatImage gradx = (_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    _KLT_FloatImage grady = (_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    gradx->data = gradx_host;
    grady->data = grady_host;
    gradx->ncols = grady->ncols = ncols;
    gradx->nrows = grady->nrows = nrows;

    // Preprocessing (unchanged)
    if (tc->smoothBeforeSelecting) {
        _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(img, ncols, nrows, tmpimg);
        _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
        _KLTFreeFloatImage(tmpimg);
    } else {
        _KLTToFloatImage(img, ncols, nrows, floatimg);
    }

    // Compute gradients into pinned host buffers
    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);

    // Create a CUDA stream for async ops
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Create cudaArrays for texture objects
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t gradxArray = NULL;
    cudaArray_t gradyArray = NULL;
    if (cudaMallocArray(&gradxArray, &channelDesc, ncols, nrows, cudaArrayDefault) != cudaSuccess) {
        fprintf(stderr, "cudaMallocArray gradxArray failed\n");
    }
    if (cudaMallocArray(&gradyArray, &channelDesc, ncols, nrows, cudaArrayDefault) != cudaSuccess) {
        fprintf(stderr, "cudaMallocArray gradyArray failed\n");
    }

    // Copy pinned host gradients -> cudaArray asynchronously (2D copy)
    size_t srcPitch = (size_t)ncols * sizeof(float);
    cudaMemcpy2DToArrayAsync(gradxArray, 0, 0, gradx_host, srcPitch, srcPitch, nrows, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DToArrayAsync(gradyArray, 0, 0, grady_host, srcPitch, srcPitch, nrows, cudaMemcpyHostToDevice, stream);

    // Create resource descriptors for texture objects
    struct cudaResourceDesc resDescX, resDescY;
    memset(&resDescX, 0, sizeof(resDescX));
    memset(&resDescY, 0, sizeof(resDescY));
    resDescX.resType = cudaResourceTypeArray;
    resDescX.res.array.array = gradxArray;
    resDescY.resType = cudaResourceTypeArray;
    resDescY.res.array.array = gradyArray;

    // Texture descriptors
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; // exact texel fetch
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // use unnormalized coordinates

    cudaTextureObject_t gradxTex = 0;
    cudaTextureObject_t gradyTex = 0;
    cudaCreateTextureObject(&gradxTex, &resDescX, &texDesc, NULL);
    cudaCreateTextureObject(&gradyTex, &resDescY, &texDesc, NULL);

    // Allocate device memory for eigenvalues
    float *eigvals_dev = NULL;
    if (cudaMalloc((void**)&eigvals_dev, img_size_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc eigvals_dev failed\n");
        eigvals_dev = NULL;
    }

    // Determine grid and block configuration using occupancy helper
    int dev = 0;
    cudaGetDevice(&dev);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // target active blocks per SM: use occupancy calculator for no-shared kernel (lower shared mem)
    int minGridSize = 0;
    int blockSizeFromOccupancy = 0;
    size_t dynamicSMemSize = 0;
    // Query a good block size for the no-shared kernel (1D)
    cudaError_t occErr = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSizeFromOccupancy,
        (void*)computeTrackabilityKernel_tex_noshared,  // function pointer
        0, // dynamicSMemSize
        prop.maxThreadsPerBlock
    );
    if (occErr != cudaSuccess) {
        // fallback
        blockSizeFromOccupancy = DEFAULT_BLOCK_X * DEFAULT_BLOCK_Y;
    }

    // Map the 1D occupancy block size to a 2D block shape (choose a warp-aligned X)
    int blkThreads = blockSizeFromOccupancy;
    if (blkThreads <= 0) blkThreads = DEFAULT_BLOCK_X * DEFAULT_BLOCK_Y;
    // choose block.x to be a multiple of 32 (warp) and <= 64 or 128, and block.y accordingly
    int block_x = DEFAULT_BLOCK_X;
    int block_y = DEFAULT_BLOCK_Y;
    // try to pick block_x as 32 and block_y = blkThreads / 32 if feasible
    if (blkThreads >= 32) {
        int candidate_y = blkThreads / 32;
        if (candidate_y < 1) candidate_y = 1;
        if (candidate_y > prop.maxThreadsPerBlock / 32) candidate_y = prop.maxThreadsPerBlock / 32;
        // clamp candidate_y to a sensible range (2..16)
        if (candidate_y > 16) candidate_y = 16;
        if (candidate_y < 1) candidate_y = 1;
        block_x = 32;
        block_y = candidate_y;
    } else {
        // small occupancy result - fallback to DEFAULT
        block_x = DEFAULT_BLOCK_X;
        block_y = DEFAULT_BLOCK_Y;
    }
    if (block_x * block_y > prop.maxThreadsPerBlock) {
        block_y = prop.maxThreadsPerBlock / block_x;
        if (block_y < 1) block_y = 1;
    }

    dim3 block(block_x, block_y);

    // compute grid dims covering valid inner region (consider border)
    int borderx = tc->borderx;
    int bordery = tc->bordery;
    if (borderx < tc->window_width/2) borderx = tc->window_width/2;
    if (bordery < tc->window_height/2) bordery = tc->window_height/2;

    int inner_cols = ncols - 2*borderx;
    int inner_rows = nrows - 2*bordery;
    if (inner_cols < 1) inner_cols = 1;
    if (inner_rows < 1) inner_rows = 1;

    dim3 grid(
        (inner_cols + block.x - 1) / block.x,
        (inner_rows + block.y - 1) / block.y
    );

    // Compute required shared memory for the tiled kernel
    int tile_w = block.x + 2 * window_hw;
    int tile_h = block.y + 2 * window_hh;
    // safety: avoid integer overflow
    size_t shared_bytes_required = 0;
    if ((size_t)tile_w * (size_t)tile_h <= ((size_t)1<<31)) {
        shared_bytes_required = (size_t)tile_w * (size_t)tile_h * sizeof(float) * 2; // gradx + grady
    } else {
        shared_bytes_required = prop.sharedMemPerBlock + 1; // force no-shared
    }

    // Keep a conservative margin: require shared_bytes_required to be <= 75% of sharedMemPerBlock
    size_t max_shared_allowed = (size_t)(prop.sharedMemPerBlock * 3 / 4);

    bool use_shared_kernel = true;
    if (shared_bytes_required == 0 || shared_bytes_required > max_shared_allowed) {
        use_shared_kernel = false;
    }
    if ((int)(block.x * block.y) > prop.maxThreadsPerBlock) use_shared_kernel = false;

    // Launch appropriate kernel in the stream (after async copies are queued)
    if (use_shared_kernel) {
        // Launch tiled kernel with dynamic shared mem and occupancy-chosen block shape
        computeTrackabilityKernel_tex_shared<<<grid, block, shared_bytes_required, stream>>>(
            gradxTex, gradyTex,
            ncols, nrows,
            window_hw, window_hh,
            tc->nSkippedPixels, tc->borderx, tc->bordery,
            eigvals_dev,
            tile_w, tile_h
        );
    } else {
        // Launch the no-shared kernel to maximize occupancy
        computeTrackabilityKernel_tex_noshared<<<grid, block, 0, stream>>>(
            gradxTex, gradyTex,
            ncols, nrows,
            window_hw, window_hh,
            tc->nSkippedPixels, tc->borderx, tc->bordery,
            eigvals_dev
        );
    }

    // Check kernel launch errors
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kerr));
    }

    // Allocate pinned host memory for eigenvalues copy-back
    float *eigvals_host = NULL;
    if (cudaHostAlloc((void**)&eigvals_host, img_size_bytes, cudaHostAllocDefault) != cudaSuccess) {
        eigvals_host = (float*) malloc(img_size_bytes);
    }

    // Async copy eigenvalues back to host
    if (eigvals_dev != NULL) {
        cudaMemcpyAsync(eigvals_host, eigvals_dev, img_size_bytes, cudaMemcpyDeviceToHost, stream);
    } else {
        // If allocation failed, zero host buffer and skip kernel results (safe fallback)
        memset(eigvals_host, 0, img_size_bytes);
    }

    // Wait for stream to finish all operations
    cudaStreamSynchronize(stream);

    // Now create pointlist from eigenvalues (same logic as before)
    int *pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));
    int npoints = 0;
    unsigned int limit = 1;
    for (int i = 0; i < sizeof(int); i++) limit *= 256;
    limit = limit/2 - 1;

    int *ptr = pointlist;
    for (int y = bordery; y < nrows - bordery; y += tc->nSkippedPixels+1) {
        for (int x = borderx; x < ncols - borderx; x += tc->nSkippedPixels+1) {
            *ptr++ = x;
            *ptr++ = y;
            float val = eigvals_host[y*ncols + x];
            if (val > (float)limit) val = (float) limit;
            *ptr++ = (int) val;
            npoints++;
        }
    }

    // Sort the pointlist
    _sortPointList(pointlist, npoints);

    // Enforce minimum distance and fill featurelist
    _enforceMinimumDistance(
        pointlist, npoints, featurelist,
        ncols, nrows,
        tc->mindist,
        tc->min_eigenvalue,
        (mode == SELECTING_ALL)
    );

    // Free memory and destroy texture objects, stream
    free(pointlist);
    if (eigvals_host) {
        cudaFreeHost(eigvals_host);
    }
    if (eigvals_dev) {
        cudaFree(eigvals_dev);
    }
    cudaDestroyTextureObject(gradxTex);
    cudaDestroyTextureObject(gradyTex);
    if (gradxArray) cudaFreeArray(gradxArray);
    if (gradyArray) cudaFreeArray(gradyArray);
    cudaStreamDestroy(stream);

    // Free pinned gradient host memory (cudaFreeHost if pinned, else free)
    if (gradx_host) {
        cudaFreeHost(gradx_host);
    }
    if (grady_host) {
        cudaFreeHost(grady_host);
    }
    free(gradx);
    free(grady);
    _KLTFreeFloatImage(floatimg);
}

/* ------------------------------------------------------------------- */

void KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList fl)
{
  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Selecting the %d best features "
            "from a %d by %d image...  ", fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeaturesGPU(tc, img, ncols, nrows, fl, SELECTING_ALL);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features found.\n", 
            KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

void _KLTSelectGoodFeatures(
    KLT_TrackingContext tc,
    KLT_PixelType *img,
    int ncols,
    int nrows,
    KLT_FeatureList featurelist,
    selectionMode mode
) {
    _KLTSelectGoodFeaturesGPU(tc, img, ncols, nrows, featurelist, mode);
}

void KLTReplaceLostFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList fl)
{
  int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Attempting to replace %d features "
            "in a %d by %d image...  ", nLostFeatures, ncols, nrows);
    fflush(stderr);
  }

  if (nLostFeatures > 0)
    _KLTSelectGoodFeaturesGPU(tc, img, ncols, nrows, fl, REPLACING_SOME);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features replaced.\n",
            nLostFeatures - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

#ifdef __cplusplus
} // extern "C"
#endif
