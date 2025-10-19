/*********************************************************************
 * selectGoodFeatures.c
 *
 *********************************************************************/

/* Standard includes */
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

/* GPU min eigenvalue */
__device__ float minEigenvalueGPU(float gxx, float gxy, float gyy)
{
  //printf("GPU riunning eigenvlaue\n");
  return 0.5f * (gxx + gyy - sqrtf((gxx - gyy)*(gxx - gyy) + 4.0f*gxy*gxy));
}

/* GPU kernel */
__global__ void computeTrackabilityKernel(
    const float* gradx, const float* grady,
    int ncols, int nrows,
    int window_hw, int window_hh,
    int skipPixels,
    int borderx, int bordery,
    float* eigvals
) {
    printf("GPU running compute kernel: selectgoodfeatire\n");
    int x = borderx + blockIdx.x * blockDim.x + threadIdx.x;
    int y = bordery + blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= ncols - borderx || y >= nrows - bordery) return;
    if ((x - borderx) % (skipPixels + 1) != 0 || (y - bordery) % (skipPixels + 1) != 0) return;

    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;

    for (int yy = -window_hh; yy <= window_hh; yy++) {
        for (int xx = -window_hw; xx <= window_hw; xx++) {
            float gx = gradx[(y+yy)*ncols + (x+xx)];
            float gy = grady[(y+yy)*ncols + (x+xx)];
            gxx += gx*gx;
            gxy += gx*gy;
            gyy += gy*gy;
        }
    }

    eigvals[y*ncols + x] = minEigenvalueGPU(gxx, gxy, gyy);
}

/* ------------------------------------------------------------------- */
/* GPU wrapper */
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

    _KLT_FloatImage floatimg = _KLTCreateFloatImage(ncols, nrows);
    _KLT_FloatImage gradx    = _KLTCreateFloatImage(ncols, nrows);
    _KLT_FloatImage grady    = _KLTCreateFloatImage(ncols, nrows);

    if (tc->smoothBeforeSelecting) {
        _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(img, ncols, nrows, tmpimg);
        _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
        _KLTFreeFloatImage(tmpimg);
    } else _KLTToFloatImage(img, ncols, nrows, floatimg);

    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);

    float *d_gradx, *d_grady, *d_eigvals;
    cudaMalloc(&d_gradx, ncols*nrows*sizeof(float));
    cudaMalloc(&d_grady, ncols*nrows*sizeof(float));
    cudaMalloc(&d_eigvals, ncols*nrows*sizeof(float));

    cudaMemcpy(d_gradx, gradx->data, ncols*nrows*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady, grady->data, ncols*nrows*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((ncols+15)/16, (nrows+15)/16);

    computeTrackabilityKernel<<<grid, block>>>(
        d_gradx, d_grady, ncols, nrows, window_hw, window_hh,
        tc->nSkippedPixels, tc->borderx, tc->bordery, d_eigvals
    );

    float* eigvals = (float*)malloc(ncols*nrows*sizeof(float));
    cudaMemcpy(eigvals, d_eigvals, ncols*nrows*sizeof(float), cudaMemcpyDeviceToHost);

    /* TODO: convert eigvals -> pointlist, sort, enforce minimum distance on CPU */
    // Create pointlist (x, y, value) from GPU eigenvalues
    int *pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));
    int npoints = 0;
    unsigned int limit = 1;
    for (int i = 0; i < sizeof(int); i++) limit *= 256;
    limit = limit/2 - 1;

    int borderx = tc->borderx;
    int bordery = tc->bordery;
    if (borderx < tc->window_width/2) borderx = tc->window_width/2;
    if (bordery < tc->window_height/2) bordery = tc->window_height/2;

    int *ptr = pointlist;
    for (int y = bordery; y < nrows - bordery; y += tc->nSkippedPixels+1)
        for (int x = borderx; x < ncols - borderx; x += tc->nSkippedPixels+1) {
            *ptr++ = x;
            *ptr++ = y;
            float val = eigvals[y*ncols + x];
            if (val > limit) val = (float) limit;
            *ptr++ = (int) val;
            npoints++;
        }

    // Sort the pointlist (CPU)
    _sortPointList(pointlist, npoints);

    // Enforce minimum distance and fill featurelist (CPU)
    _enforceMinimumDistance(
        pointlist, npoints, featurelist,
        ncols, nrows,
        tc->mindist,
        tc->min_eigenvalue,
        (mode == SELECTING_ALL)
    );

    // Free temporary memory
    free(pointlist);
    free(eigvals);

    cudaFree(d_gradx); cudaFree(d_grady); cudaFree(d_eigvals);
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
}

/* ------------------------------------------------------------------- */

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
// Rename this function
void _KLTSelectGoodFeatures(  // <-- use the CPU function name
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
