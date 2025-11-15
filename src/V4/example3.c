/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "pnmio.h"
#include "klt.h"
#include <cuda_runtime.h>
#include <time.h>      // for timing

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  // ----------------- Timing Variables -----------------
  clock_t start, end;
  double cpu_time_used;
  cudaEvent_t gpu_start, gpu_end;
  cudaEvent_t copy_start, copy_end;
  float gpu_time = 0.0f;
  float copy_time = 0.0f;

  // Start CPU wall timer
  start = clock();

  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150;
  //int nFrames = 10; // for old dataset
  int nFrames = 200;
  int ncols, nrows;
  int i;

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  img1 = pgmReadFile("../../data/750/frame_099.pgm", NULL, &ncols, &nrows);
  //img1 = pgmReadFile("../../data/img0.pgm", NULL, &ncols, &nrows); // for old dataset
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));

  // Create CUDA events
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_end);
  cudaEventCreate(&copy_start);
  cudaEventCreate(&copy_end);

  // Start GPU timing
  cudaEventRecord(gpu_start, 0);

  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  //KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "output/feat99.ppm");
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat0.ppm");

  for (i = 100 ; i < nFrames ; i++)  { // set the loop to start from 1 if want to run old dataset
    //printf("Processing frame %d\n", i);
    sprintf(fnamein, "../../data/750/frame_%d.pgm", i);
	//sprintf(fnamein, "../../data/img%d.pgm", i); // for old dataset

    // Record copy time (I/O)
    cudaEventRecord(copy_start, 0);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    cudaEventRecord(copy_end, 0);
    cudaEventSynchronize(copy_end);

    float tempCopy = 0.0f;
    cudaEventElapsedTime(&tempCopy, copy_start, copy_end);
    copy_time += tempCopy;

    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    //sprintf(fnameout, "output/feat%d.ppm", i);
    //sprintf(fnameout, "feat%d.ppm", i);
    //KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }

  // Stop GPU timing
  cudaEventRecord(gpu_end, 0);
  cudaEventSynchronize(gpu_end);
  cudaEventElapsedTime(&gpu_time, gpu_start, gpu_end);

  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  // Stop total CPU timer
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("\n---------------------------------------------\n");
  printf("GPU Kernel Execution Time : %.4f ms (%.4f s)\n", gpu_time, gpu_time / 1000.0);
  printf("Data Transfer (I/O) Time  : %.4f ms (%.4f s)\n", copy_time, copy_time / 1000.0);
  printf("Total CPU Wall Time       : %.4f s\n", cpu_time_used);
  printf("---------------------------------------------\n");

  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_end);
  cudaEventDestroy(copy_start);
  cudaEventDestroy(copy_end);

  return 0;
}
