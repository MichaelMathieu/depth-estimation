#ifndef __ARDRONE_API_H_19APR12__
#define __ARDRONE_API_H_19APR12__

#include<string>

#include "drone_api.h"

class ARdroneAPI : public DroneAPI {
public:
  ARdroneAPI(const std::string & fifo_path);
  virtual ~ARdroneAPI();
public:
  virtual void next();
  virtual matf getDepthMap() const;
  virtual matf getIMUAccel() const;
  virtual matf getIMUGyro() const;
  
  virtual void setControl(float pitch, float gaz, float roll, float yaw);

  virtual std::string toString() const;
private:
  int fifo;
};

#endif
