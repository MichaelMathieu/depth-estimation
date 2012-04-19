#ifndef __ARDRONE_API_H_19APR12__
#define __ARDRONE_API_H_19APR12__

#include<string>

#include<opencv/cv.h>
typedef cv::Mat_<float> matf;

class ArdroneAPI {
public:
  virtual ~ArdroneAPI() {};
public:
  virtual void next() =0;
  virtual matf getDepthMap() const =0;
  virtual matf getIMUAccel() const =0;
  virtual matf getIMUGyro() const =0;
  
  virtual void setControl(float pitch, float gaz, float roll, float yaw) =0;

  virtual std::string toString() const =0;
};

#endif
