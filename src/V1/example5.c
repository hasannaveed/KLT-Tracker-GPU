/**********************************************************************
Demonstrates manually tweaking the tracking context parameters.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include <time.h>   // for timing

#ifdef WIN32
int RunExample5()
#else
int main()
#endif
{
  clock_t start, end;
  double cpu_time_used;
  // Start timing
  start = clock();

  unsigned char *img1, *img2;
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  int nFeatures = 100;
  int ncols, nrows;

  tc = KLTCreateTrackingContext();
  tc->mindist = 20;
  tc->window_width  = 9;
  tc->window_height = 9;
  KLTChangeTCPyramid(tc, 15);
  KLTUpdateTCBorder(tc);
  fl = KLTCreateFeatureList(nFeatures);

  img1 = pgmReadFile("../../data/img0.pgm", NULL, &ncols, &nrows);
  img2 = pgmReadFile("../../data/img2.pgm", NULL, &ncols, &nrows);

  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1b.ppm");

  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2b.ppm");
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("\n---------------------------------------------\n");
  printf("Execution Time: %.4f seconds\n", cpu_time_used);
  printf("---------------------------------------------\n");

  return 0;
}

