#ifndef __ARDRONE_COMMON_H_19APR12__
#define __ARDRONE_COMMON_H_19APR12__

#ifndef NULL
#define NULL 0
#endif

#include<cmath>

double getTimeInSec();

template<typename T> int inline round2(T a) {
  return floor(a+(T)0.5);
}

#endif
