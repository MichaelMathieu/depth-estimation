#include "common.h"
#include<sys/time.h>

double getTimeInSec() {
  timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}
