#ifndef __ARDRONE_COMMON_H_19APR12__
#define __ARDRONE_COMMON_H_19APR12__

#ifndef NULL
#define NULL 0
#endif

#include<cmath>
#include<opencv/cv.h>

typedef cv::Mat_<float> matf;
typedef cv::Mat_<cv::Vec3b> mat3b;

const double PI = 3.1415926535897932384626433832795028841971693993751058209;

double getTimeInSec();
double randn(double mu=0.0, double sigma=1.0); 

template<typename T> int inline round2(T a) {
  return floor(a+(T)0.5);
}

template<typename T> T inline saturate(T a, T min, T max) {
  if (a < min)
    a = min;
  if (a > max)
    a = max;
  return a;
}

#endif
