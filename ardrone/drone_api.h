#ifndef __DRONE_API_H_19APR12__
#define __DRONE_API_H_19APR12__

#include<string>
#include "common.h"

class DroneAPI {
public:
  virtual ~DroneAPI() {};
public:
  virtual void next() =0;
  virtual float getDeltaT() const =0;
  virtual matf getDepthMap() const =0;
  virtual matf getIMUTranslation() const =0;
  //virtual matf getVisualOdometryTranslation() const =0;
  //virtual matf getFilteredTranslation() const =0;
  virtual matf getIMUGyro() const =0;
  virtual float getIMUAltitude() const =0;
  virtual float getBatteryState() const =0;
  virtual int getDroneState() const =0; //not ready yet (what are the states?)
  
  virtual void takeoff() =0;
  virtual void land() =0;
  virtual void setControl(float pitch, float gaz, float roll, float yaw) =0;

  virtual std::string toString() const =0;
};

#endif
