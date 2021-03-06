#include "simulator.h"
#include "common.h"

#include<iostream>
#include<cstdio>
#include<cmath>
using namespace std;

SimulatedAPI::SimulatedAPI(int depthMapWidth, int depthMapHeight)
  :DroneAPI(), last_time(getTimeInSec()), delta_t(0.0f),
   flying(false), theta(0.0f), dtheta(0.0f),
   x(3, 1, 0.0f), dx(3, 1, 0.0f), ddx(3, 1, 0.0f),
   pitch(0.0f), gaz(0.0f), roll(0.0f), dyaw(0.0f),
   dmH(depthMapHeight), dmW(depthMapWidth),
   alpha_friction(0.5f), focal_length(depthMapWidth),
   obstacles() {
  for (int i=0; i< 100; ++i) {
    obstacles.push_back(Obstacle(5*(i+1), -10+sin(i*0.5)*2*i, 0, 1.0f));
    obstacles.push_back(Obstacle(5*(i+1), +10+sin(i*0.5)*2*i, 0, 1.0f));
  }
  // for (int i=0; i< 100; ++i) {
  //   obstacles.push_back(Obstacle(10*2*i, -30+i, 0, 1.0f));
  //   obstacles.push_back(Obstacle(10*(2*i+1), 30-i, 0, 1.0f));
  // }
}

SimulatedAPI::~SimulatedAPI() {
  flying = false;
}

void SimulatedAPI::next() {
  double time = getTimeInSec();
  delta_t = (float)(time-last_time);
  updatePosition(delta_t);
  last_time = time;
}

float SimulatedAPI::getDeltaT() const {
  return delta_t;
}

matf SimulatedAPI::getDepthMap() const {
  //matf map(dmH, dmW, 1e30f);
  matf map(dmH, dmW, 100);
  matf pray = getPRay();
  matf npray = getNPRay();
  matf up = getUp();
  float hh = floor(dmH/2);
  float hw = floor(dmW/2);
  const float epsilon = 1e-5;
  for (size_t i = 0; i < obstacles.size(); ++i) {
    matf center = obstacles[i].center;
    float radius = obstacles[i].radius;
    
    matf v = center-x;
    float D = pray.dot(v);
    if (D <= epsilon) // the point is behind
      continue;
    //cout <<  pray << endl;
    float k = focal_length / D;
    float a = k * npray.dot(v);
    float b = k * up.dot(v);
    float D2 = norm(v);
    float k2 = focal_length / D2;
    float r = k2 * radius;
    // printf("Map\n");
    // printf("%f %f %f %f\n", r, radius, k2, D2);
    for (int ii = max(0,round2(a+hw-r)); ii < min(dmW, round2(a+hw+r)); ++ii)
      for (int jj = max(0, round2(b+hh-r)); jj < min(dmH, round2(b+hh+r)); ++jj)
        if (D2 < map(jj, ii))
          map(jj, ii) = D2;
  }
  return map;
}

matf SimulatedAPI::getConfidenceMap() const {
  //matf map(dmH, dmW, 1e30f);
  return matf(dmH, dmW, 1.0f);
}

matf SimulatedAPI::getIMUTranslation() const {
  matf pray = getPRay();
  matf npray = getNPRay();
  matf up = getUp();
  matf v = dx * delta_t;
  matf ret(3,1);
  ret(0,0) = v.dot(pray) + randn(0, 0.2);
  ret(1,0) = v.dot(npray) + randn(0, 0.2);
  ret(2,0) = v.dot(up) + randn(0, 0.2);
  return ret;
}

matf SimulatedAPI::getVisualOdometryTranslation() const {
  matf pray = getPRay();
  matf npray = getNPRay();
  matf up = getUp();
  matf v = dx * delta_t;
  matf ret(3,1);
  ret(0,0) = v.dot(pray) + randn(0, 0.1);
  ret(1,0) = v.dot(npray) + randn(0, 0.1);
  ret(2,0) = v.dot(up) + randn(0, 0.1);
  return ret;
}

matf SimulatedAPI::getFilteredTranslation() const {
  matf imuTranslation = getIMUTranslation();
  float imuVar = getIMUVariance();
  matf voTranslation = getVisualOdometryTranslation();
  float voVar = getVisualOdometryVariance();

  float K = imuVar/(imuVar+voVar);

  matf ret(3,1);
  ret(0,0) = imuTranslation(0,0) + K*(voTranslation(0,0) - imuTranslation(0,0));
  ret(1,0) = imuTranslation(1,0) + K*(voTranslation(1,0) - imuTranslation(1,0));
  ret(2,0) = imuTranslation(2,0) + K*(voTranslation(2,0) - imuTranslation(2,0));
  return ret;
}

matf SimulatedAPI::getIMUGyro() const {
  return getPRay();
}

float SimulatedAPI::getIMUAltitude() const {
  return x(2,0);
}

float SimulatedAPI::getBatteryState() const {
  return 100.0f;
}

int SimulatedAPI::getDroneState() const {
  return 1; //not implemented (I don't know what are the states anyway)
}

float SimulatedAPI::getIMUVariance() const {
  return 1.0f;
}

float SimulatedAPI::getVisualOdometryVariance() const {
  return 1.0f;
}

void SimulatedAPI::takeoff() {
  flying = true;
}

void SimulatedAPI::land() {
  flying = false;
}

void SimulatedAPI::setControl(float pitch_, float gaz_, float roll_, float dyaw_) {
  pitch = pitch_;
  gaz = gaz_;
  roll = roll_;
  dyaw = dyaw_;
}

string SimulatedAPI::toString() const {
  ostringstream oss;
  char buffer[128];
  oss << "SimulatedAPI:\n";
  sprintf(buffer, "  x     = (%.5f %.5f %.5f)", x(0,0), x(1,0), x(2,0));
  oss << buffer << "\n";
  sprintf(buffer, "  dx    = (%.5f %.5f %.5f)", dx(0,0), dx(1,0), dx(2,0));
  oss << buffer << "\n";
  sprintf(buffer, "  theta = %.5f", theta);
  oss << buffer << "\n";
  return oss.str();
}

void SimulatedAPI::updatePosition(float delta_t) {
  if (!flying)
    return;
  dtheta= dyaw * delta_t;
  theta = theta + dtheta;
  matf up = getUp();
  matf pray = getPRay();
  matf npray = getNPRay();
  if (alpha_friction*delta_t > 1.0f)
    ddx = -dx;
  else
    ddx = -alpha_friction * dx;
  ddx += pitch*10.0f * pray;
  ddx += roll*10.0f * npray;
  ddx += gaz*10.0f * up;
  dx += ddx * delta_t;
  x += dx * delta_t;
}

matf SimulatedAPI::getUp() const {
  matf up(3, 1, 0.0f);
  up(2, 0) = 1.0f;
  return up;
}

matf SimulatedAPI::getPRay() const {
  matf pray(3, 1, 0.0f);
  pray(0, 0) = cos(theta);
  pray(1, 0) = sin(theta);
  return pray;
}

matf SimulatedAPI::getNPRay() const {
  matf npray(3, 1, 0.0f);
  npray(0, 0) = -sin(theta);
  npray(1, 0) = cos(theta);
  return npray;
}
