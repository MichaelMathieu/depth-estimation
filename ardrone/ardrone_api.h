#ifndef __ARDRONE_API_H_19APR12__
#define __ARDRONE_API_H_19APR12__

#include<string>

#include "drone_api.h"

const size_t fifoBufferLen = 33;

class ARdroneAPI : public DroneAPI {
public:
  ARdroneAPI(const std::string & fifo_path);
  virtual ~ARdroneAPI();
public:
  virtual void next();
  virtual matf getDepthMap() const;
  virtual matf getIMUAccel() const;
  virtual matf getIMUGyro() const;

  virtual void takeoff();
  virtual void land();
  virtual void setControl(float pitch, float gaz, float roll, float yaw);

  virtual std::string toString() const;
private:
  int fifo;
  char fifoBuffer[fifoBufferLen+1];
  enum Order {
    TAKEOFF, LAND, CONTROL,
  };
  void sendOnFifo(Order order, float pitch = 0.0f, float gaz = 0.0f,
		  float roll = 0.0f, float yaw = 0.0f);
};

#endif
