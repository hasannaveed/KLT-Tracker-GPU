/*********************************************************************
 * trackFeatures_optimized_globalcache.cu
 *
 * OpenACC version - replaces previous CUDA-kernel implementations
 *
 *********************************************************************/

#include <assert.h>
#include <math.h>       /* floorf, fabs */
#include <stdlib.h>     /* malloc, free */
#include <stdio.h>      /* fprintf, fflush */
#include <string.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"   /* _KLT_FloatImage */
#include "pyramid.h"    /* _KLT_Pyramid */

#include <openacc.h>

extern int KLT_verbose;
typedef float *_FloatWindow;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* -------------------------
   Persistent cache struct
   ------------------------- */
typedef struct {
    float *d_ptr;     // device pointer (unused in OpenACC path but kept for compatibility)
    int width;        // image width
    int height;       // image height
    size_t capacity;  // allocated elements (width*height)
    bool valid;
} ImgCache;

/* Static persistent caches for the four gradient images (kept for compatibility) */
static ImgCache cache_gx1 = {0,0,0,0,false};
static ImgCache cache_gy1 = {0,0,0,0,false};
static ImgCache cache_gx2 = {0,0,0,0,false};
static ImgCache cache_gy2 = {0,0,0,0,false};

/* Static stream flags (not used by OpenACC but kept to avoid changing other code) */
static cudaStream_t cached_stream = NULL;
static bool stream_initialized = false;

/* Helper to ensure a device buffer is allocated with at least width*height capacity
   (kept for compatibility; not used by the OpenACC implementations below) */
static void ensure_device_image(ImgCache *c, int width, int height) {
    size_t elems = (size_t)width * (size_t)height;
    if (c->valid && c->width == width && c->height == height && c->capacity >= elems) {
        // already allocated and fits
        return;
    }
    // if existing allocation is insufficient, free and reallocate
    if (c->valid && c->d_ptr != NULL) {
        CUDA_CHECK(cudaFree(c->d_ptr));
        c->d_ptr = NULL;
        c->valid = false;
    }
    // allocate new
    float *d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d, elems * sizeof(float)));
    c->d_ptr = d;
    c->width = width;
    c->height = height;
    c->capacity = elems;
    c->valid = true;
}

/* -------------------------
   Replace CUDA kernels with OpenACC parallel loops
   ------------------------- */

/* host-side bilinear sampler matching device semantics used previously
   returns 0 for out-of-bounds samples exactly like original CUDA device sampler */
static inline float host_sampleBilinear(const float *img, int img_w, int img_h, float fx, float fy) {
    if (fx < 0.f || fy < 0.f || fx > (float)(img_w-1) || fy > (float)(img_h-1)) return 0.0f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = x0 + 1; if (x1 >= img_w) x1 = img_w - 1;
    int y1 = y0 + 1; if (y1 >= img_h) y1 = img_h - 1;
    float dx = fx - (float)x0;
    float dy = fy - (float)y0;
    float v00 = img[y0 * img_w + x0];
    float v10 = img[y0 * img_w + x1];
    float v01 = img[y1 * img_w + x0];
    float v11 = img[y1 * img_w + x1];
    float v0 = v00 * (1.0f - dx) + v10 * dx;
    float v1 = v01 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

/* Batched version using OpenACC directives.
   Mirrors original logic: for each feature/window compute bilinear samples of 4 gradient images,
   store sum-of-gradients per pixel in output arrays out_gradx_all / out_grady_all.
*/
void _computeGradientSum_CUDA_batched(
    _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
    _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
    const float *features_x1, const float *features_y1,
    const float *features_x2, const float *features_y2,
    int count, int win_w, int win_h,
    _FloatWindow out_gradx_all, _FloatWindow out_grady_all)
{
    if (count <= 0) return;
    if (!gradx1 || !grady1 || !gradx2 || !grady2) return;
    if (!features_x1 || !features_y1 || !features_x2 || !features_y2) return;
    if (!out_gradx_all || !out_grady_all) return;
    if (win_w <= 0 || win_h <= 0) return;

    const int img_w = gradx1->ncols;
    const int img_h = gradx1->nrows;
    const int win_elems = win_w * win_h;
    const size_t total_out_elems = (size_t)win_elems * (size_t)count;

    /* Use OpenACC data region to transfer images and coordinate arrays to the device.
       We copyout the full batched result array at the end. */
    #pragma acc data copyin(gradx1->data[0:(size_t)img_w*img_h], \
                             grady1->data[0:(size_t)img_w*img_h], \
                             gradx2->data[0:(size_t)img_w*img_h], \
                             grady2->data[0:(size_t)img_w*img_h], \
                             features_x1[0:count], features_y1[0:count], \
                             features_x2[0:count], features_y2[0:count]) \
                     copyout(out_gradx_all[0:total_out_elems], out_grady_all[0:total_out_elems])
    {
        /* Parallelize outer loop (features/windows) using gang. Inner loops over window pixels are vectorized.
           This maps each window to a parallel gang (coarse grain) and pixels inside to vector lanes. */
        #pragma acc parallel loop gang present(gradx1->data,grady1->data,gradx2->data,grady2->data, \
                                             features_x1,features_y1,features_x2,features_y2, \
                                             out_gradx_all,out_grady_all)
        for (int fid = 0; fid < count; ++fid) {
            const float cx1 = features_x1[fid];
            const float cy1 = features_y1[fid];
            const float cx2 = features_x2[fid];
            const float cy2 = features_y2[fid];

            float *out_gx = out_gradx_all + (size_t)fid * win_elems;
            float *out_gy = out_grady_all + (size_t)fid * win_elems;

            /* collapse inner loops to expose 2D parallelism and vectorize them */
            #pragma acc loop vector collapse(2)
            for (int v = 0; v < win_h; ++v) {
                for (int u = 0; u < win_w; ++u) {
                    int hw = win_w / 2;
                    int hh = win_h / 2;
                    float sample_x1 = cx1 + (float)(u - hw);
                    float sample_y1 = cy1 + (float)(v - hh);
                    float sample_x2 = cx2 + (float)(u - hw);
                    float sample_y2 = cy2 + (float)(v - hh);

                    float gx1 = host_sampleBilinear(gradx1->data, img_w, img_h, sample_x1, sample_y1);
                    float gy1 = host_sampleBilinear(grady1->data, img_w, img_h, sample_x1, sample_y1);
                    float gx2 = host_sampleBilinear(gradx2->data, img_w, img_h, sample_x2, sample_y2);
                    float gy2 = host_sampleBilinear(grady2->data, img_w, img_h, sample_x2, sample_y2);

                    int idx = v * win_w + u;
                    out_gx[idx] = gx1 + gx2;
                    out_gy[idx] = gy1 + gy2;
                }
            }
        }
    } /* end acc data region */
}

/* Per-window (single window) OpenACC implementation.
   Produces gradx[], grady[] for a single window centered at (x1,y1) and (x2,y2).
*/
void _computeGradientSum_CUDA(
    _KLT_FloatImage gradx1,
    _KLT_FloatImage grady1,
    _KLT_FloatImage gradx2,
    _KLT_FloatImage grady2,
    float x1, float y1,
    float x2, float y2,
    int win_w, int win_h,     // size of window
    _FloatWindow gradx,       // output pointers (host)
    _FloatWindow grady)
{
    /* Quick checks */
    if (!gradx1 || !grady1 || !gradx2 || !grady2) return;
    if (!gradx || !grady) return;
    if (win_w <= 0 || win_h <= 0) return;

    const int img_w = gradx1->ncols;
    const int img_h = gradx1->nrows;
    size_t win_elems = (size_t)win_w * (size_t)win_h;

    /* Move the gradient images to device and compute the window there, copying out only the window result */
    #pragma acc data copyin(gradx1->data[0:(size_t)img_w*img_h], \
                             grady1->data[0:(size_t)img_w*img_h], \
                             gradx2->data[0:(size_t)img_w*img_h], \
                             grady2->data[0:(size_t)img_w*img_h]) \
                     copyout(gradx[0:win_elems], grady[0:win_elems])
    {
        /* Single parallel region; collapse 2D window loops and map to gang/worker/vector.
           This preserves the per-pixel sampling and sum-of-gradients semantics. */
        #pragma acc parallel present(gradx1->data,grady1->data,gradx2->data,grady2->data, gradx, grady)
        {
            #pragma acc loop gang collapse(2)
            for (int local_y = 0; local_y < win_h; ++local_y) {
                for (int local_x = 0; local_x < win_w; ++local_x) {
                    int i = local_x - (win_w / 2);
                    int j = local_y - (win_h / 2);

                    float sx1 = x1 + (float)i;
                    float sy1 = y1 + (float)j;
                    float sx2 = x2 + (float)i;
                    float sy2 = y2 + (float)j;

                    float gx1 = host_sampleBilinear(gradx1->data, img_w, img_h, sx1, sy1);
                    float gy1 = host_sampleBilinear(grady1->data, img_w, img_h, sx1, sy1);
                    float gx2 = host_sampleBilinear(gradx2->data, img_w, img_h, sx2, sy2);
                    float gy2 = host_sampleBilinear(grady2->data, img_w, img_h, sx2, sy2);

                    int out_idx = local_y * win_w + local_x;
                    gradx[out_idx] = gx1 + gx2;
                    grady[out_idx] = gy1 + gy2;
                }
            }
        } /* end parallel */
    } /* end data */
}

/* Extern "C" wrapper - signature preserved */
#ifdef __cplusplus
extern "C" {
#endif
void _compute2by2GradientMatrixRaw(float *gradx, float *grady, int w, int h,
                                   float *gxx, float *gxy, float *gyy) {
    *gxx = *gxy = *gyy = 0.0f;
    for (int i = 0; i < w * h; i++) {
        *gxx += gradx[i] * gradx[i];
        *gxy += gradx[i] * grady[i];
        *gyy += grady[i] * grady[i];
    }
}

void _compute2by1ErrorVectorRaw(float *gradx, float *grady, int w, int h,
                                float step_factor, float *ex, float *ey) {
    *ex = *ey = 0.0f;
    for (int i = 0; i < w * h; i++) {
        *ex += gradx[i] * step_factor;
        *ey += grady[i] * step_factor;
    }
}


static float _interpolate(
  float x,
  float y,
  _KLT_FloatImage img)
{
  int xt = (int) x;  /* coordinates of top-left corner */
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  float *ptr = img->data + (img->ncols*yt) + xt;

#ifndef _DNDEBUG
  if (xt<0 || yt<0 || xt>=img->ncols-1 || yt>=img->nrows-1) {
    fprintf(stderr, "(xt,yt)=(%d,%d)  imgsize=(%d,%d)\n"
            "(x,y)=(%f,%f)  (ax,ay)=(%f,%f)\n",
            xt, yt, img->ncols, img->nrows, x, y, ax, ay);
    fflush(stderr);
  }
#endif

  assert (xt >= 0 && yt >= 0 && xt <= img->ncols - 2 && yt <= img->nrows - 2);

  return ( (1-ax) * (1-ay) * *ptr +
           ax   * (1-ay) * *(ptr+1) +
           (1-ax) *   ay   * *(ptr+(img->ncols)) +
           ax   *   ay   * *(ptr+(img->ncols)+1) );
}
static void _computeIntensityDifference(
  _KLT_FloatImage img1,   /* images */
  _KLT_FloatImage img2,
  float x1, float y1,     /* center of window in 1st img */
  float x2, float y2,     /* center of window in 2nd img */
  int width, int height,  /* size of window */
  _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      *imgdiff++ = g1 - g2;
    }
}
/* Simple wrapper to free a _FloatWindow allocated with _allocateFloatWindow() */
static inline void _freeFloatWindow(_FloatWindow fw)
{
    if (fw) free(fw);
}

static void _computeIntensityDifferenceLightingInsensitive(
  _KLT_FloatImage img1,   /* images */
  _KLT_FloatImage img2,
  float x1, float y1,     /* center of window in 1st img */
  float x2, float y2,     /* center of window in 2nd img */
  int width, int height,  /* size of window */
  _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  float g1, g2, sum1_squared = 0, sum2_squared = 0;
  register int i, j;
  
  float sum1 = 0, sum2 = 0;
  float mean1, mean2,alpha,belta;
  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      sum1 += g1;    sum2 += g2;
      sum1_squared += g1*g1;
      sum2_squared += g2*g2;
   }
  mean1=sum1_squared/(width*height);
  mean2=sum2_squared/(width*height);
  alpha = (float) sqrt(mean1/mean2);
  mean1=sum1/(width*height);
  mean2=sum2/(width*height);
  belta = mean1-alpha*mean2;

  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      *imgdiff++ = g1- g2*alpha-belta;
    } 
}


/*********************************************************************
 * _computeGradientSumLightingInsensitive
 *
 * Given two gradients and the window center in both images,
 * aligns the gradients wrt the window and computes the sum of the two 
 * overlaid gradients; normalizes for overall gain and bias.
 */

static void _computeGradientSumLightingInsensitive(
  _KLT_FloatImage gradx1,  /* gradient images */
  _KLT_FloatImage grady1,
  _KLT_FloatImage gradx2,
  _KLT_FloatImage grady2,
  _KLT_FloatImage img1,   /* images */
  _KLT_FloatImage img2,
 
  float x1, float y1,      /* center of window in 1st img */
  float x2, float y2,      /* center of window in 2nd img */
  int width, int height,   /* size of window */
  _FloatWindow gradx,      /* output */
  _FloatWindow grady)      /*   " */
{
  register int hw = width/2, hh = height/2;
  float g1, g2, sum1_squared = 0, sum2_squared = 0;
  register int i, j;
  
  float sum1 = 0, sum2 = 0;
  float mean1, mean2, alpha;
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      sum1_squared += g1;    sum2_squared += g2;
    }
  mean1 = sum1_squared/(width*height);
  mean2 = sum2_squared/(width*height);
  alpha = (float) sqrt(mean1/mean2);
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, gradx1);
      g2 = _interpolate(x2+i, y2+j, gradx2);
      *gradx++ = g1 + g2*alpha;
      g1 = _interpolate(x1+i, y1+j, grady1);
      g2 = _interpolate(x2+i, y2+j, grady2);
      *grady++ = g1+ g2*alpha;
    }  
}

/*********************************************************************
 * _compute2by2GradientMatrix
 *
 */

static void _compute2by2GradientMatrix(
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float *gxx,  /* return values */
  float *gxy, 
  float *gyy) 

{
  register float gx, gy;
  register int i;

  /* Compute values */
  *gxx = 0.0;  *gxy = 0.0;  *gyy = 0.0;
  for (i = 0 ; i < width * height ; i++)  {
    gx = *gradx++;
    gy = *grady++;
    *gxx += gx*gx;
    *gxy += gx*gy;
    *gyy += gy*gy;
  }
}
	
	
/*********************************************************************
 * _compute2by1ErrorVector
 *
 */

static void _compute2by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
  float *ex,   /* return values */
  float *ey)
{
  register float diff;
  register int i;

  /* Compute values */
  *ex = 0;  *ey = 0;  
  for (i = 0 ; i < width * height ; i++)  {
    diff = *imgdiff++;
    *ex += diff * (*gradx++);
    *ey += diff * (*grady++);
  }
  *ex *= step_factor;
  *ey *= step_factor;
}


/*********************************************************************
 * _solveEquation
 *
 * Solves the 2x2 matrix equation
 *         [gxx gxy] [dx] = [ex]
 *         [gxy gyy] [dy] = [ey]
 * for dx and dy.
 *
 * Returns KLT_TRACKED on success and KLT_SMALL_DET on failure
 */

static int _solveEquation(
  float gxx, float gxy, float gyy,
  float ex, float ey,
  float small,
  float *dx, float *dy)
{
  float det = gxx*gyy - gxy*gxy;

	
  if (det < small)  return KLT_SMALL_DET;

  *dx = (gyy*ex - gxy*ey)/det;
  *dy = (gxx*ey - gxy*ex)/det;
  return KLT_TRACKED;
}


/*********************************************************************
 * _allocateFloatWindow
 */
	
static _FloatWindow _allocateFloatWindow(
  int width,
  int height)
{
  _FloatWindow fw;

  fw = (_FloatWindow) malloc(width*height*sizeof(float));
  if (fw == NULL)  KLTError("(_allocateFloatWindow) Out of memory.");
  return fw;
}


/*********************************************************************
 * _printFloatWindow
 * (for debugging purposes)
 */

/*
static void _printFloatWindow(
  _FloatWindow fw,
  int width,
  int height)
{
  int i, j;

  fprintf(stderr, "\n");
  for (i = 0 ; i < width ; i++)  {
    for (j = 0 ; j < height ; j++)  {
      fprintf(stderr, "%6.1f ", *fw++);
    }
    fprintf(stderr, "\n");
  }
}
*/
	

/*********************************************************************
 * _sumAbsFloatWindow
 */

static float _sumAbsFloatWindow(
  _FloatWindow fw,
  int width,
  int height)
{
  float sum = 0.0;
  int w;

  for ( ; height > 0 ; height--)
    for (w=0 ; w < width ; w++)
      sum += (float) fabs(*fw++);

  return sum;
}


/*********************************************************************
 * _trackFeature
 *
 * Tracks a feature point from one image to the next.
 *
 * RETURNS
 * KLT_SMALL_DET if feature is lost,
 * KLT_MAX_ITERATIONS if tracking stopped because iterations timed out,
 * KLT_TRACKED otherwise.
 */

static int _trackFeature(
  float x1,  /* location of window in first image */
  float y1,
  float *x2, /* starting location of search in second image */
  float *y2,
  _KLT_FloatImage img1, 
  _KLT_FloatImage gradx1,
  _KLT_FloatImage grady1,
  _KLT_FloatImage img2, 
  _KLT_FloatImage gradx2,
  _KLT_FloatImage grady2,
  int width,           /* size of window */
  int height,
  float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
  int max_iterations,
  float small,         /* determinant threshold for declaring KLT_SMALL_DET */
  float th,            /* displacement threshold for stopping               */
  float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
  int lighting_insensitive)  /* whether to normalize for gain and bias */
{
  _FloatWindow imgdiff, gradx, grady;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status;
  int hw = width/2;
  int hh = height/2;
  int nc = img1->ncols;
  int nr = img1->nrows;
  float one_plus_eps = 1.001f;   /* To prevent rounding errors */

	
  /* Allocate memory for windows */
  imgdiff = _allocateFloatWindow(width, height);
  gradx   = _allocateFloatWindow(width, height);
  grady   = _allocateFloatWindow(width, height);

  /* Iteratively update the window position */
  do  {

    /* If out of bounds, exit loop */
    if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
         *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
          y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
         *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
      status = KLT_OOB;
      break;
    }

    /* Compute gradient and difference windows */
    if (lighting_insensitive) {
      _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                  width, height, imgdiff);
      _computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2, 
			  img1, img2, x1, y1, *x2, *y2, width, height, gradx, grady);
    } else {
      _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                  width, height, imgdiff);
        
///printf("[DEBUG] gradx pointer in CUDA = %p, grady pointer = %p\n", gradx, grady);

	  _computeGradientSum_CUDA(gradx1, grady1, gradx2, grady2,
			  x1, y1, *x2, *y2, width, height, gradx, grady);
      //printf("[compute gradient cuda sum called]: %d",n);

    }
		

    /* Use these windows to construct matrices */
    _compute2by2GradientMatrix(gradx, grady, width, height, 
                               &gxx, &gxy, &gyy);
    _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                            &ex, &ey);
				
    /* Using matrices, solve equation for new displacement */
    status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
    if (status == KLT_SMALL_DET)  break;

    *x2 += dx;
    *y2 += dy;
    iteration++;

  }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);

  /* Check whether window is out of bounds */
  if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps || 
      *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
    status = KLT_OOB;

  /* Check whether residue is too large */
  if (status == KLT_TRACKED)  {
    if (lighting_insensitive)
      _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                  width, height, imgdiff);
    else
      _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                  width, height, imgdiff);
    if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
      status = KLT_LARGE_RESIDUE;
  }

  /* Free memory */
  free(imgdiff);  free(gradx);  free(grady);

  /* Return appropriate value */
  if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
  else if (status == KLT_OOB)  return KLT_OOB;
  else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
  else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
  else  return KLT_TRACKED;

}


/*********************************************************************/

static KLT_BOOL _outOfBounds(
  float x,
  float y,
  int ncols,
  int nrows,
  int borderx,
  int bordery)
{
  return (x < borderx || x > ncols-1-borderx ||
          y < bordery || y > nrows-1-bordery );
}




/********************************************************************** 
* CONSISTENCY CHECK OF FEATURES BY AFFINE MAPPING (BEGIN)
* 
* Created by: Thorsten Thormaehlen (University of Hannover) June 2004    
* thormae@tnt.uni-hannover.de
* 
* Permission is granted to any individual or institution to use, copy, modify,
* and distribute this part of the software, provided that this complete authorship 
* and permission notice is maintained, intact, in all copies. 
*
* This software is provided  "as is" without express or implied warranty.
*
*
* The following static functions are helpers for the affine mapping.
* They all start with "_am". 
* There are also small changes in other files for the
* affine mapping these are all marked by "for affine mapping"
* 
* Thanks to Kevin Koeser (koeser@mip.informatik.uni-kiel.de) for fixing a bug 
*/

#define SWAP_ME(X,Y) {temp=(X);(X)=(Y);(Y)=temp;}

static float **_am_matrix(long nr, long nc)
{
  float **m;
  int a;
  m = (float **) malloc((size_t)(nr*sizeof(float*)));
  m[0] = (float *) malloc((size_t)((nr*nc)*sizeof(float)));
  for(a = 1; a < nr; a++) m[a] = m[a-1]+nc;
  return m;
}

static void _am_free_matrix(float **m)
{
  free(m[0]);
  free(m);
}


static int _am_gauss_jordan_elimination(float **a, int n, float **b, int m)
{
  /* re-implemented from Numerical Recipes in C */
  int *indxc,*indxr,*ipiv;
  int i,j,k,l,ll;
  float big,dum,pivinv,temp;
  int col = 0;
  int row = 0;

  indxc=(int *)malloc((size_t) (n*sizeof(int)));
  indxr=(int *)malloc((size_t) (n*sizeof(int)));
  ipiv=(int *)malloc((size_t) (n*sizeof(int)));
  for (j=0;j<n;j++) ipiv[j]=0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if (ipiv[j] != 1)
	for (k=0;k<n;k++) {
	  if (ipiv[k] == 0) {
	    if (fabs(a[j][k]) >= big) {
	      big= (float) fabs(a[j][k]);
	      row=j;
	      col=k;
	    }
	  } else if (ipiv[k] > 1) return KLT_SMALL_DET;
	}
    ++(ipiv[col]);
    if (row != col) {
      for (l=0;l<n;l++) SWAP_ME(a[row][l],a[col][l])
			  for (l=0;l<m;l++) SWAP_ME(b[row][l],b[col][l])
					      }
    indxr[i]=row;
    indxc[i]=col;
    if (a[col][col] == 0.0) return KLT_SMALL_DET;
    pivinv=1.0f/a[col][col];
    a[col][col]=1.0;
    for (l=0;l<n;l++) a[col][l] *= pivinv;
    for (l=0;l<m;l++) b[col][l] *= pivinv;
    for (ll=0;ll<n;ll++)
      if (ll != col) {
	dum=a[ll][col];
	a[ll][col]=0.0;
	for (l=0;l<n;l++) a[ll][l] -= a[col][l]*dum;
	for (l=0;l<m;l++) b[ll][l] -= b[col][l]*dum;
      }
  }
  for (l=n-1;l>=0;l--) {
    if (indxr[l] != indxc[l])
      for (k=0;k<n;k++)
	SWAP_ME(a[k][indxr[l]],a[k][indxc[l]]);
  }
  free(ipiv);
  free(indxr);
  free(indxc);

  return KLT_TRACKED;
}

/*********************************************************************
 * _am_getGradientWinAffine
 *
 * aligns the gradients with the affine transformed window 
 */

static void _am_getGradientWinAffine(
				     _KLT_FloatImage in_gradx,
				     _KLT_FloatImage in_grady,
				     float x, float y,      /* center of window*/
				     float Axx, float Ayx , float Axy, float Ayy,    /* affine mapping */
				     int width, int height,   /* size of window */
				     _FloatWindow out_gradx,      /* output */
				     _FloatWindow out_grady)      /* output */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float mi, mj;
 
  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      mi = Axx * i + Axy * j;
      mj = Ayx * i + Ayy * j;
      *out_gradx++ = _interpolate(x+mi, y+mj, in_gradx);
      *out_grady++ = _interpolate(x+mi, y+mj, in_grady);
    }
  
}

/*********************************************************************
 * _computeAffineMappedImage
 * used only for DEBUG output
 *     
*/

static void _am_computeAffineMappedImage(
					 _KLT_FloatImage img,   /* images */
					 float x, float y,      /* center of window  */
					 float Axx, float Ayx , float Axy, float Ayy,    /* affine mapping */   
					 int width, int height,  /* size of window */
					 _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float mi, mj;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      mi = Axx * i + Axy * j;
      mj = Ayx * i + Ayy * j;
      *imgdiff++ = _interpolate(x+mi, y+mj, img);
    }
}


/*********************************************************************
 * _getSubFloatImage
 */

static void _am_getSubFloatImage(
				 _KLT_FloatImage img,   /* image */
				 float x, float y,     /* center of window */
				 _KLT_FloatImage window)   /* output */
{
  register int hw = window->ncols/2, hh = window->nrows/2;
  int x0 = (int) x;
  int y0 = (int) y;
  float * windata = window->data; 
  int offset;
  register int i, j;

  assert(x0 - hw >= 0);
  assert(y0 - hh >= 0);
  assert(x0 + hw <= img->ncols);
  assert(y0 + hh <= img->nrows); 

  /* copy values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      offset = (j+y0)*img->ncols + (i+x0);
      *windata++ = *(img->data+offset);
    }
}

/*********************************************************************
 * _am_computeIntensityDifferenceAffine
 *
 * Given two images and the window center in both images,
 * aligns the images with the window and computes the difference 
 * between the two overlaid images using the affine mapping.
 *       A =  [ Axx Axy]
 *            [ Ayx Ayy]        
*/

static void _am_computeIntensityDifferenceAffine(
						 _KLT_FloatImage img1,   /* images */
						 _KLT_FloatImage img2,
						 float x1, float y1,     /* center of window in 1st img */
						 float x2, float y2,      /* center of window in 2nd img */
						 float Axx, float Ayx , float Axy, float Ayy,    /* affine mapping */   
						 int width, int height,  /* size of window */
						 _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;
  float mi, mj;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      mi = Axx * i + Axy * j;
      mj = Ayx * i + Ayy * j;
      g2 = _interpolate(x2+mi, y2+mj, img2);
      *imgdiff++ = g1 - g2;
    }
}

/*********************************************************************
 * _am_compute6by6GradientMatrix
 *
 */

static void _am_compute6by6GradientMatrix(
					  _FloatWindow gradx,
					  _FloatWindow grady,
					  int width,   /* size of window */
					  int height,
					  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, gxx, gxy, gyy,  x, y, xx, xy, yy;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 6 ; j++)  {
    for (i = j ; i < 6 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      gx = *gradx++;
      gy = *grady++;
      gxx = gx * gx;
      gxy = gx * gy;
      gyy = gy * gy;
      x = (float) i; 
      y = (float) j; 
      xx = x * x;
      xy = x * y;
      yy = y * y;
      
      T[0][0] += xx * gxx; 
      T[0][1] += xx * gxy;
      T[0][2] += xy * gxx;
      T[0][3] += xy * gxy;
      T[0][4] += x  * gxx;
      T[0][5] += x  * gxy;
	
      T[1][1] += xx * gyy;
      T[1][2] += xy * gxy;
      T[1][3] += xy * gyy;
      T[1][4] += x  * gxy;
      T[1][5] += x  * gyy;
			 
      T[2][2] += yy * gxx;
      T[2][3] += yy * gxy;
      T[2][4] += y  * gxx;
      T[2][5] += y  * gxy;
	 
      T[3][3] += yy * gyy;
      T[3][4] += y  * gxy;
      T[3][5] += y  * gyy; 

      T[4][4] += gxx; 
      T[4][5] += gxy;
      
      T[5][5] += gyy; 
    }
  }
  
  for (j = 0 ; j < 5 ; j++)  {
    for (i = j+1 ; i < 6 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}



/*********************************************************************
 * _am_compute6by1ErrorVector
 *
 */

static void _am_compute6by1ErrorVector(
				       _FloatWindow imgdiff,
				       _FloatWindow gradx,
				       _FloatWindow grady,
				       int width,   /* size of window */
				       int height,
				       float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff,  diffgradx,  diffgrady;

  /* Set values to zero */  
  for(i = 0; i < 6; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      diff = *imgdiff++;
      diffgradx = diff * (*gradx++);
      diffgrady = diff * (*grady++);
      e[0][0] += diffgradx * i;
      e[1][0] += diffgrady * i;
      e[2][0] += diffgradx * j; 
      e[3][0] += diffgrady * j; 
      e[4][0] += diffgradx;
      e[5][0] += diffgrady; 
    }
  }
  
  for(i = 0; i < 6; i++) e[i][0] *= 0.5;
  
}


/*********************************************************************
 * _am_compute4by4GradientMatrix
 *
 */

static void _am_compute4by4GradientMatrix(
					  _FloatWindow gradx,
					  _FloatWindow grady,
					  int width,   /* size of window */
					  int height,
					  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, x, y;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 4 ; j++)  {
    for (i = 0 ; i < 4 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      gx = *gradx++;
      gy = *grady++;
      x = (float) i; 
      y = (float) j; 
      T[0][0] += (x*gx+y*gy) * (x*gx+y*gy);
      T[0][1] += (x*gx+y*gy)*(x*gy-y*gx);
      T[0][2] += (x*gx+y*gy)*gx;
      T[0][3] += (x*gx+y*gy)*gy;
   
      T[1][1] += (x*gy-y*gx) * (x*gy-y*gx);
      T[1][2] += (x*gy-y*gx)*gx;
      T[1][3] += (x*gy-y*gx)*gy;
     
      T[2][2] += gx*gx;
      T[2][3] += gx*gy;
      
      T[3][3] += gy*gy;
    }
  }
  
  for (j = 0 ; j < 3 ; j++)  {
    for (i = j+1 ; i < 4 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}

/*********************************************************************
 * _am_compute4by1ErrorVector
 *
 */

static void _am_compute4by1ErrorVector(
				       _FloatWindow imgdiff,
				       _FloatWindow gradx,
				       _FloatWindow grady,
				       int width,   /* size of window */
				       int height,
				       float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff,  diffgradx,  diffgrady;

  /* Set values to zero */  
  for(i = 0; i < 4; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      diff = *imgdiff++;
      diffgradx = diff * (*gradx++);
      diffgrady = diff * (*grady++);
      e[0][0] += diffgradx * i + diffgrady * j;
      e[1][0] += diffgrady * i - diffgradx * j;
      e[2][0] += diffgradx;
      e[3][0] += diffgrady;
    }
  }
  
  for(i = 0; i < 4; i++) e[i][0] *= 0.5;
  
}



/*********************************************************************
 * _am_trackFeatureAffine
 *
 * Tracks a feature point from the image of first occurrence to the actual image.
 *
 * RETURNS
 * KLT_SMALL_DET or KLT_LARGE_RESIDUE or KLT_OOB if feature is lost,
 * KLT_TRACKED otherwise.
 */

/* if you enalbe the DEBUG_AFFINE_MAPPING make sure you have created a directory "./debug" */
/* #define DEBUG_AFFINE_MAPPING */

#ifdef DEBUG_AFFINE_MAPPING
static int counter = 0;
static int glob_index = 0;
#endif

static int _am_trackFeatureAffine(
				  float x1,  /* location of window in first image */
				  float y1,
				  float *x2, /* starting location of search in second image */
				  float *y2,
				  _KLT_FloatImage img1, 
				  _KLT_FloatImage gradx1,
				  _KLT_FloatImage grady1,
				  _KLT_FloatImage img2, 
				  _KLT_FloatImage gradx2,
				  _KLT_FloatImage grady2,
				  int width,           /* size of window */
				  int height,
				  float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
				  int max_iterations,
				  float small,         /* determinant threshold for declaring KLT_SMALL_DET */
				  float th,            /* displacement threshold for stopping  */
				  float th_aff,
				  float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
				  int lighting_insensitive,  /* whether to normalize for gain and bias */
				  int affine_map,      /* whether to evaluates the consistency of features with affine mapping */
				  float mdd,           /* difference between the displacements */
				  float *Axx, float *Ayx, 
				  float *Axy, float *Ayy)        /* used affine mapping */
{


  _FloatWindow imgdiff, gradx, grady;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status = 0;
  int hw = width/2;
  int hh = height/2;
  int nc1 = img1->ncols;
  int nr1 = img1->nrows;
  int nc2 = img2->ncols;
  int nr2 = img2->nrows;
  float **a;
  float **T; 
  float one_plus_eps = 1.001f;   /* To prevent rounding errors */
  float old_x2 = *x2;
  float old_y2 = *y2;
  KLT_BOOL convergence = FALSE;

#ifdef DEBUG_AFFINE_MAPPING
  char fname[80];
  _KLT_FloatImage aff_diff_win = _KLTCreateFloatImage(width,height);
  printf("starting location x2=%f y2=%f\n", *x2, *y2);
#endif
  
  /* Allocate memory for windows */
  imgdiff = _allocateFloatWindow(width, height);
  gradx   = _allocateFloatWindow(width, height);
  grady   = _allocateFloatWindow(width, height);
  T = _am_matrix(6,6);
  a = _am_matrix(6,1);

  /* Iteratively update the window position */
  do  {
    if(!affine_map) {
      /* pure translation tracker */
      
      /* If out of bounds, exit loop */
      if ( x1-hw < 0.0f || nc1-( x1+hw) < one_plus_eps ||
          *x2-hw < 0.0f || nc2-(*x2+hw) < one_plus_eps ||
           y1-hh < 0.0f || nr1-( y1+hh) < one_plus_eps ||
          *y2-hh < 0.0f || nr2-(*y2+hh) < one_plus_eps) {
        status = KLT_OOB;
        break;
      }
      
      /* Compute gradient and difference windows */
      if (lighting_insensitive) {
        _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                    width, height, imgdiff);
        _computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2, 
			    img1, img2, x1, y1, *x2, *y2, width, height, gradx, grady);
      } else {
        _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                    width, height, imgdiff);
		_computeGradientSum_CUDA(gradx1, grady1, gradx2, grady2,
					x1, y1, *x2, *y2, width, height, gradx, grady);

      }
      
#ifdef DEBUG_AFFINE_MAPPING	
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_trans_diff_win%03d.%03d.pgm", glob_index, counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
      printf("iter = %d translation tracker res: %f\n", iteration, _sumAbsFloatWindow(imgdiff, width, height)/(width*height));
#endif
  
      /* Use these windows to construct matrices */
      _compute2by2GradientMatrix(gradx, grady, width, height, 
				 &gxx, &gxy, &gyy);
      _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
			      &ex, &ey);
				
      /* Using matrices, solve equation for new displacement */
      status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);

      convergence = (fabs(dx) < th && fabs(dy) < th);
      
      *x2 += dx;
      *y2 += dy;
      
    }else{
      /* affine tracker */
      
      float ul_x =  *Axx * (-hw) + *Axy *   hh  + *x2;  /* upper left corner */
      float ul_y =  *Ayx * (-hw) + *Ayy *   hh  + *y2; 
      float ll_x =  *Axx * (-hw) + *Axy * (-hh) + *x2;  /* lower left corner */
      float ll_y =  *Ayx * (-hw) + *Ayy * (-hh) + *y2;
      float ur_x =  *Axx *   hw  + *Axy *   hh  + *x2;  /* upper right corner */
      float ur_y =  *Ayx *   hw  + *Ayy *   hh  + *y2;
      float lr_x =  *Axx *   hw  + *Axy * (-hh) + *x2;  /* lower right corner */
      float lr_y =  *Ayx *   hw  + *Ayy * (-hh) + *y2;

      /* If out of bounds, exit loop */
      if ( x1-hw < 0.0f ||  nc1-(x1+hw) < one_plus_eps ||
           y1-hh < 0.0f ||  nr1-(y1+hh) < one_plus_eps ||
           ul_x  < 0.0f ||  nc2-(ul_x ) < one_plus_eps ||
           ll_x  < 0.0f ||  nc2-(ll_x ) < one_plus_eps ||
           ur_x  < 0.0f ||  nc2-(ur_x ) < one_plus_eps ||
           lr_x  < 0.0f ||  nc2-(lr_x ) < one_plus_eps ||
           ul_y  < 0.0f ||  nr2-(ul_y ) < one_plus_eps ||
           ll_y  < 0.0f ||  nr2-(ll_y ) < one_plus_eps ||
           ur_y  < 0.0f ||  nr2-(ur_y ) < one_plus_eps ||
           lr_y  < 0.0f ||  nr2-(lr_y ) < one_plus_eps) {
        status = KLT_OOB;
        break;
      }

#ifdef DEBUG_AFFINE_MAPPING
      counter++;
      _am_computeAffineMappedImage(img1, x1, y1,  1.0, 0.0 , 0.0, 1.0, width, height, imgdiff);
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_aff_diff_win%03d.%03d_1.pgm", glob_index, counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
      
      _am_computeAffineMappedImage(img2, *x2, *y2,  *Axx, *Ayx , *Axy, *Ayy, width, height, imgdiff);
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_aff_diff_win%03d.%03d_2.pgm", glob_index, counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
#endif
      
      _am_computeIntensityDifferenceAffine(img1, img2, x1, y1, *x2, *y2,  *Axx, *Ayx , *Axy, *Ayy,
					   width, height, imgdiff);
#ifdef DEBUG_AFFINE_MAPPING    
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_aff_diff_win%03d.%03d_3.pgm", glob_index,counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
      
      printf("iter = %d affine tracker res: %f\n", iteration, _sumAbsFloatWindow(imgdiff, width, height)/(width*height));
#endif      
      
      _am_getGradientWinAffine(gradx2, grady2, *x2, *y2, *Axx, *Ayx , *Axy, *Ayy,
			       width, height, gradx, grady);

      switch(affine_map){
      case 1:
	_am_compute4by1ErrorVector(imgdiff, gradx, grady, width, height, a);
	_am_compute4by4GradientMatrix(gradx, grady, width, height, T);
	
	status = _am_gauss_jordan_elimination(T,4,a,1);
	
	*Axx += a[0][0];
	*Ayx += a[1][0];
	*Ayy = *Axx;
	*Axy = -(*Ayx);
	
	dx = a[2][0];
	dy = a[3][0];
	
	break;
      case 2:
	_am_compute6by1ErrorVector(imgdiff, gradx, grady, width, height, a);
	_am_compute6by6GradientMatrix(gradx, grady, width, height, T);
      
	status = _am_gauss_jordan_elimination(T,6,a,1);
	
	*Axx += a[0][0];
	*Ayx += a[1][0];
	*Axy += a[2][0];
	*Ayy += a[3][0];

	dx = a[4][0];
	dy = a[5][0];
      
	break;
      }
      
      *x2 += dx;
      *y2 += dy;
      
      /* old upper left corner - new upper left corner */
      ul_x -=  *Axx * (-hw) + *Axy *   hh  + *x2;  
      ul_y -=  *Ayx * (-hw) + *Ayy *   hh  + *y2; 
      /* old lower left corner - new lower left corner */
      ll_x -=  *Axx * (-hw) + *Axy * (-hh) + *x2;  
      ll_y -=  *Ayx * (-hw) + *Ayy * (-hh) + *y2;
      /* old upper right corner - new upper right corner */
      ur_x -=  *Axx *   hw  + *Axy *   hh  + *x2;  
      ur_y -=  *Ayx *   hw  + *Ayy *   hh  + *y2;
      /* old lower right corner - new lower right corner */
      lr_x -=  *Axx *   hw  + *Axy * (-hh) + *x2;  
      lr_y -=  *Ayx *   hw  + *Ayy * (-hh) + *y2;

#ifdef DEBUG_AFFINE_MAPPING 
      printf ("iter = %d, ul_x=%f ul_y=%f ll_x=%f ll_y=%f ur_x=%f ur_y=%f lr_x=%f lr_y=%f \n",
	      iteration, ul_x, ul_y, ll_x, ll_y, ur_x, ur_y, lr_x, lr_y);
#endif  

      convergence = (fabs(dx) < th && fabs(dy) < th  &&
		     fabs(ul_x) < th_aff && fabs(ul_y) < th_aff &&
		     fabs(ll_x) < th_aff && fabs(ll_y) < th_aff &&
		     fabs(ur_x) < th_aff && fabs(ur_y) < th_aff &&
		     fabs(lr_x) < th_aff && fabs(lr_y) < th_aff);
    }
    
    if (status == KLT_SMALL_DET)  break;
    iteration++;
#ifdef DEBUG_AFFINE_MAPPING 
    printf ("iter = %d, x1=%f, y1=%f, x2=%f, y2=%f,  Axx=%f, Ayx=%f , Axy=%f, Ayy=%f \n",iteration, x1, y1, *x2, *y2,  *Axx, *Ayx , *Axy, *Ayy);
#endif   
    }  while ( !convergence  && iteration < max_iterations); 
    /*}  while ( (fabs(dx)>=th || fabs(dy)>=th || (affine_map && iteration < 8) ) && iteration < max_iterations); */
  _am_free_matrix(T);
  _am_free_matrix(a);

  /* Check whether window is out of bounds */
  if (*x2-hw < 0.0f || nc2-(*x2+hw) < one_plus_eps || 
      *y2-hh < 0.0f || nr2-(*y2+hh) < one_plus_eps)
    status = KLT_OOB;

  /* Check whether feature point has moved to much during iteration*/
  if ( (*x2-old_x2) > mdd || (*y2-old_y2) > mdd )
    status = KLT_OOB;

  /* Check whether residue is too large */
  if (status == KLT_TRACKED)  {
    if(!affine_map){
      _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
				  width, height, imgdiff);
    }else{
      _am_computeIntensityDifferenceAffine(img1, img2, x1, y1, *x2, *y2,  *Axx, *Ayx , *Axy, *Ayy,
					   width, height, imgdiff);
    }
#ifdef DEBUG_AFFINE_MAPPING
    printf("iter = %d final_res = %f\n", iteration, _sumAbsFloatWindow(imgdiff, width, height)/(width*height));
#endif 
    if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
      status = KLT_LARGE_RESIDUE;
  }

  /* Free memory */
  free(imgdiff);  free(gradx);  free(grady);

#ifdef DEBUG_AFFINE_MAPPING
  printf("iter = %d status=%d\n", iteration, status);
  _KLTFreeFloatImage( aff_diff_win );
#endif 
  
  /* Return appropriate value */
  return status;
}

/*
 * CONSISTENCY CHECK OF FEATURES BY AFFINE MAPPING (END)
 **********************************************************************/



/*********************************************************************
 * KLTTrackFeatures
 *
 * Tracks feature points from one image to the next.
 */

void KLTTrackFeatures(
    KLT_TrackingContext tc,
    KLT_PixelType *img1,
    KLT_PixelType *img2,
    int ncols,
    int nrows,
    KLT_FeatureList featurelist)
{
    _KLT_FloatImage tmpimg, floatimg1, floatimg2;
    _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
                 pyramid2, pyramid2_gradx, pyramid2_grady;
    float subsampling = (float) tc->subsampling;
    int val;
    int indx, r;
    KLT_BOOL floatimg1_created = FALSE;
    int i;

    if (KLT_verbose >= 1) {
        fprintf(stderr, "(KLT) Tracking %d features in a %d by %d image...  ",
                KLTCountRemainingFeatures(featurelist), ncols, nrows);
        fflush(stderr);
    }

    // Ensure valid window sizes
    if (tc->window_width % 2 != 1) tc->window_width++;
    if (tc->window_height % 2 != 1) tc->window_height++;
    if (tc->window_width < 3) tc->window_width = 3;
    if (tc->window_height < 3) tc->window_height = 3;

    tmpimg = _KLTCreateFloatImage(ncols, nrows);

    // Build first image pyramids
    if (tc->sequentialMode && tc->pyramid_last != NULL) {
        pyramid1 = (_KLT_Pyramid)tc->pyramid_last;
        pyramid1_gradx = (_KLT_Pyramid)tc->pyramid_last_gradx;
        pyramid1_grady = (_KLT_Pyramid)tc->pyramid_last_grady;
    } else {
        floatimg1_created = TRUE;
        floatimg1 = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(img1, ncols, nrows, tmpimg);
        _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);
        pyramid1 = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        _KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
        pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        for (i = 0; i < tc->nPyramidLevels; i++)
            _KLTComputeGradients(pyramid1->img[i], tc->grad_sigma,
                                 pyramid1_gradx->img[i], pyramid1_grady->img[i]);
    }

    // Build second image pyramids
    floatimg2 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img2, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
    pyramid2 = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    _KLTComputePyramid(floatimg2, pyramid2, tc->pyramid_sigma_fact);
    pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    nvtxRangePushA("Batched call");
    for (i = 0; i < tc->nPyramidLevels; i++)
        _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma,
                             pyramid2_gradx->img[i], pyramid2_grady->img[i]);

    /* === Corrected batched tracking with proper scaling & mapping === */
{
    int total = featurelist->nFeatures;
    int winW = tc->window_width;
    int winH = tc->window_height;
    int winElems = winW * winH;
    float subsampling = (float)tc->subsampling;

    /* Build list of valid features and mapping to original indices */
    int validCount = 0;
    int *validIdx = (int*)malloc(total * sizeof(int));
    for (int i = 0; i < total; ++i) {
        if (featurelist->feature[i]->val >= 0) {
            validIdx[validCount++] = i;
        }
    }

    if (validCount == 0) {
        free(validIdx);
    } else {
        /* Per-feature coordinate arrays, one entry per valid feature */
        float *cur_x = (float*)malloc(validCount * sizeof(float));
        float *cur_y = (float*)malloc(validCount * sizeof(float));
        float *out_x = (float*)malloc(validCount * sizeof(float));
        float *out_y = (float*)malloc(validCount * sizeof(float));

        /* Initialize cur_x/cur_y from featurelist (full-res), then scale down to coarsest */
        for (int vi = 0; vi < validCount; ++vi) {
            int orig = validIdx[vi];
            cur_x[vi] = featurelist->feature[orig]->x;
            cur_y[vi] = featurelist->feature[orig]->y;
        }
        /* divide to coarsest */
        for (int s = tc->nPyramidLevels - 1; s >= 0; --s) {
            for (int vi = 0; vi < validCount; ++vi) {
                cur_x[vi] /= subsampling;
                cur_y[vi] /= subsampling;
            }
        }
        /* out_x/out_y start equal to cur_x/cur_y at coarsest */
        for (int vi = 0; vi < validCount; ++vi) {
            out_x[vi] = cur_x[vi];
            out_y[vi] = cur_y[vi];
        }

        /* Pinned staging buffers reused per iteration */
        float *batch_x1 = NULL, *batch_y1 = NULL, *batch_x2 = NULL, *batch_y2 = NULL;
        float *gradx_batch = NULL, *grady_batch = NULL;
        cudaMallocHost((void**)&batch_x1, validCount * sizeof(float));
        cudaMallocHost((void**)&batch_y1, validCount * sizeof(float));
        cudaMallocHost((void**)&batch_x2, validCount * sizeof(float));
        cudaMallocHost((void**)&batch_y2, validCount * sizeof(float));
        cudaMallocHost((void**)&gradx_batch, validCount * winElems * sizeof(float));
        cudaMallocHost((void**)&grady_batch, validCount * winElems * sizeof(float));

        /* For per-iteration temporary imgdiffs */
        _FloatWindow *imgdiff_buf = NULL;

        /* Per-level coarse -> fine */
        for (int lvl = tc->nPyramidLevels - 1; lvl >= 0; --lvl) {
            /* Move coordinates to this level (multiply once) */
            for (int vi = 0; vi < validCount; ++vi) {
                cur_x[vi] *= subsampling;
                cur_y[vi] *= subsampling;
                out_x[vi] *= subsampling;
                out_y[vi] *= subsampling;
            }

            /* Iteratively refine at this level */
            int *active = (int*)malloc(validCount * sizeof(int)); /* 1 if still active for iteration */
            int *done = (int*)calloc(validCount, sizeof(int));   /* 1 if converged/lost for this level */
            int anyActive = 0;
            for (int vi = 0; vi < validCount; ++vi) { active[vi] = 1; anyActive |= 1; }

            /* We'll run up to max_iterations; on each iteration we batch all currently-active features */
            for (int iter = 0; iter < tc->max_iterations; ++iter) {
                /* Build batch lists for active & not-done features */
                int batchN = 0;
                for (int vi = 0; vi < validCount; ++vi) {
                    if (!active[vi] || done[vi]) continue;

                    /* OOB check at this level (same logic as single-feature path) */
                    int nc = pyramid1->ncols[lvl];
                    int nr = pyramid1->nrows[lvl];
                    int hw = winW/2, hh = winH/2;
                    float one_plus_eps = 1.001f;
                    if (cur_x[vi]-hw < 0.0f || nc - (cur_x[vi]+hw) < one_plus_eps ||
                        out_x[vi]-hw < 0.0f || nc - (out_x[vi]+hw) < one_plus_eps ||
                        cur_y[vi]-hh < 0.0f || nr - (cur_y[vi]+hh) < one_plus_eps ||
                        out_y[vi]-hh < 0.0f || nr - (out_y[vi]+hh) < one_plus_eps) {
                        /* Mark lost for this feature */
                        int orig = validIdx[vi];
                        featurelist->feature[orig]->x = -1.0f;
                        featurelist->feature[orig]->y = -1.0f;
                        featurelist->feature[orig]->val = KLT_OOB;
                        done[vi] = 1;
                        active[vi] = 0;
                        continue;
                    }

                    /* Fill batch arrays */
                    batch_x1[batchN] = cur_x[vi];
                    batch_y1[batchN] = cur_y[vi];
                    batch_x2[batchN] = out_x[vi];
                    batch_y2[batchN] = out_y[vi];
                    batchN++;
                }

                if (batchN == 0) break; /* nothing left this iteration */

                /* Compute imgdiff per batched feature on CPU (same as original) */
                imgdiff_buf = (_FloatWindow*)malloc(batchN * sizeof(_FloatWindow));
                for (int b = 0; b < batchN; ++b) {
                    imgdiff_buf[b] = _allocateFloatWindow(winW, winH);
                    _computeIntensityDifference(pyramid1->img[lvl], pyramid2->img[lvl],
                                               batch_x1[b], batch_y1[b], batch_x2[b], batch_y2[b],
                                               winW, winH, imgdiff_buf[b]);
                }

                /* Compute gradient windows for the batch on GPU (single H->D + kernel + D->H) */
                _computeGradientSum_CUDA_batched(
                    pyramid1_gradx->img[lvl], pyramid1_grady->img[lvl],
                    pyramid2_gradx->img[lvl], pyramid2_grady->img[lvl],
                    batch_x1, batch_y1, batch_x2, batch_y2,
                    batchN, winW, winH,
                    gradx_batch, grady_batch
                );

                /* Now consume the batch: we must iterate over validCount in same order as batch was built */
                int bIndex = 0;
                for (int vi = 0; vi < validCount; ++vi) {
                    if (!active[vi] || done[vi]) continue;

                    /* grad window for this batch item */
                    float *gx = gradx_batch + (size_t)bIndex * winElems;
                    float *gy = grady_batch + (size_t)bIndex * winElems;

                    /* compute 2x2 matrix and error vector (using CPU imgdiff) */
                    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f, ex = 0.0f, ey = 0.0f;
                    for (int p = 0; p < winElems; ++p) {
                        float gxv = gx[p];
                        float gyv = gy[p];
                        gxx += gxv * gxv;
                        gxy += gxv * gyv;
                        gyy += gyv * gyv;
                        ex += imgdiff_buf[bIndex][p] * gxv * tc->step_factor;
                        ey += imgdiff_buf[bIndex][p] * gyv * tc->step_factor;
                    }

                    /* Solve */
                    float dx = 0.0f, dy = 0.0f;
                    int stat = _solveEquation(gxx, gxy, gyy, ex, ey, tc->min_determinant, &dx, &dy);
                    if (stat == KLT_SMALL_DET) {
                        int orig = validIdx[vi];
                        featurelist->feature[orig]->x = -1.0f;
                        featurelist->feature[orig]->y = -1.0f;
                        featurelist->feature[orig]->val = KLT_SMALL_DET;
                        done[vi] = 1;
                        active[vi] = 0;
                    } else {
                        /* update output position */
                        out_x[vi] += dx;
                        out_y[vi] += dy;
                        /* check convergence threshold (min_displacement) */
                        if (fabs(dx) < tc->min_displacement && fabs(dy) < tc->min_displacement) {
                            /* mark done for this level (but keep tracked) */
                            done[vi] = 1;
                            active[vi] = 0;
                        }
                    }

                    bIndex++;
                } /* end consume batch */

                /* free imgdiff windows for this batch */
                for (int b = 0; b < batchN; ++b) _freeFloatWindow(imgdiff_buf[b]);
                free(imgdiff_buf);
                imgdiff_buf = NULL;

                /* if no active left, break */
                int anyLeft = 0;
                for (int vi = 0; vi < validCount; ++vi) if (!done[vi]) { anyLeft = 1; break; }
                if (!anyLeft) break;
            } /* end iter loop */

            free(active);
            free(done);
            /* After finishing level, rescale coords for next (finer) level will be done at top of next loop iteration */
        } /* end per-level */

        /* After all levels: finalize results and do residue checks similar to original code */
        for (int vi = 0; vi < validCount; ++vi) {
            int orig = validIdx[vi];
            /* if featurelist was already marked lost, skip */
            if (featurelist->feature[orig]->val < 0) continue;

            /* final out_x/out_y are in full resolution? We kept scaling consistent (we multiplied at each level)
               So out_x/out_y should now be at full resolution. Check OOB and residue similar to original. */
            if (_outOfBounds(out_x[vi], out_y[vi], ncols, nrows, tc->borderx, tc->bordery)) {
                featurelist->feature[orig]->x = -1.0f;
                featurelist->feature[orig]->y = -1.0f;
                featurelist->feature[orig]->val = KLT_OOB;
            } else {
                /* compute final imgdiff and check residue (as original) */
                _FloatWindow imgdiffFinal = _allocateFloatWindow(winW, winH);
                _computeIntensityDifference(pyramid1->img[0], pyramid2->img[0],
                                           cur_x[vi] /* original cur scaled back? */, cur_y[vi],
                                           out_x[vi], out_y[vi], winW, winH, imgdiffFinal);
                float meanAbs = _sumAbsFloatWindow(imgdiffFinal, winW, winH) / (winW * winH);
                _freeFloatWindow(imgdiffFinal);
                if (meanAbs > tc->max_residue) {
                    featurelist->feature[orig]->x = -1.0f;
                    featurelist->feature[orig]->y = -1.0f;
                    featurelist->feature[orig]->val = KLT_LARGE_RESIDUE;
                } else {
                    featurelist->feature[orig]->x = out_x[vi];
                    featurelist->feature[orig]->y = out_y[vi];
                    featurelist->feature[orig]->val = KLT_TRACKED;
                }
            }
        }

        /* free resources */
        free(cur_x); free(cur_y); free(out_x); free(out_y);
        cudaFreeHost(batch_x1); cudaFreeHost(batch_y1);
        cudaFreeHost(batch_x2); cudaFreeHost(batch_y2);
        cudaFreeHost(gradx_batch); cudaFreeHost(grady_batch);
        free(validIdx);
    }
}
/* === end corrected batched tracking === */

nvtxRangePop();
    if (tc->sequentialMode) {
        tc->pyramid_last = pyramid2;
        tc->pyramid_last_gradx = pyramid2_gradx;
        tc->pyramid_last_grady = pyramid2_grady;
    } else {
        _KLTFreePyramid(pyramid2);
        _KLTFreePyramid(pyramid2_gradx);
        _KLTFreePyramid(pyramid2_grady);
    }

    // Free temp data
    _KLTFreeFloatImage(tmpimg);
    if (floatimg1_created) _KLTFreeFloatImage(floatimg1);
    _KLTFreeFloatImage(floatimg2);
    _KLTFreePyramid(pyramid1);
    _KLTFreePyramid(pyramid1_gradx);
    _KLTFreePyramid(pyramid1_grady);

    if (KLT_verbose >= 1) {
        fprintf(stderr, "\n\t%d features successfully tracked.\n",
                KLTCountRemainingFeatures(featurelist));
        fflush(stderr);
    }
}


#ifdef __cplusplus
}
#endif
