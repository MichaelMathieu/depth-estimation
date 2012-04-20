#ifndef __ARDRONE_SIMULATOR_H_19APR12__
#define __ARDRONE_SIMULATOR_H_19APR12__

#include<string>
#include<vector>

#include "drone_api.h"

class SimulatedAPI : public DroneAPI {
public:
  SimulatedAPI(int depthMapWidth = 320, int depthMapHeight = 240);
  virtual ~SimulatedAPI();
public:
  virtual void next();
  virtual matf getDepthMap() const;
  virtual matf getIMUAccel() const;
  virtual matf getIMUGyro() const;

  virtual void takeoff();
  virtual void land();
  virtual void setControl(float pitch, float gaz, float roll, float dyaw);

  virtual std::string toString() const;


public:
  struct Obstacle {
    matf center;
    float radius;
    Obstacle(const matf & center, float radius)
      :center(center), radius(radius) {};
    Obstacle(float x, float y, float z, float radius)
      :center(3, 1), radius(radius) {center(0,0)=x;center(1,0)=y;center(2,0)=z;};
  };
private:
  double last_time;
  bool flying;
  float theta, dtheta;
  matf x, dx, ddx;
  float pitch, gaz, roll, dyaw;
  int dmH, dmW;
  float alpha_friction;
  void updatePosition(float delta_t);
  matf getPRay() const;
  matf getNPRay() const;
  matf getUp() const;
  
  float focal_length;
  std::vector<Obstacle> obstacles;
};

#endif
