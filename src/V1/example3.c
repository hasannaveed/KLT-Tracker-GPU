/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "pnmio.h"
#include <time.h>      // for timing
#include "klt.h"

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  clock_t start, end;
  double cpu_time_used;

    // Start timing
  start = clock();
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150; 
  //int nFrames = 10;
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

  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
//KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "output/feat99.ppm");
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat0.ppm");

  for (i = 100 ; i < nFrames ; i++)  {
    sprintf(fnamein, "../../data/750/frame_%d.pgm", i);
	  //sprintf(fnamein, "../../data/img%d.pgm", i); // for old dataset
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "feat%d.ppm", i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }
  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("\n---------------------------------------------\n");
  printf("Execution Time: %.4f seconds\n", cpu_time_used);
  printf("---------------------------------------------\n");
  return 0;
}