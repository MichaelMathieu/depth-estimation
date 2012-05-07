#include "radial_depth_map.h"
#include "common.h"
#include "simulator.h"
#include<cmath>
#include<ostream>
#include<iostream>
using namespace std;

#define MAX_VARIANCE 5

RadialDepthMap::RadialDepthMap(size_t nBinsTheta, float maxDepth, float unseenDecay,
		   float focalLength)
  :maxDepth(maxDepth), focalLength(focalLength), unseenDecay(unseenDecay),
   map(nBinsTheta, maxDepth), theta_sight(0.0f) {
  assert((nBinsTheta > 0));
};

RadialDepthMap::Ray RadialDepthMap::getRayFromPixel(float x, float y, float wImg, float hImg) {
  float theta = atan2(x, focalLength);
  return Ray(this, theta);
}

void RadialDepthMap::newPixel(float x, float y, float depth, float variance,
      float wImg, float hImg) {
  Ray ray = getRayFromPixel(x-wImg/2, y, wImg, hImg);
  size_t iTheta = getIThetaFromTheta(ray.theta);

  float mapVariance = map.getVariance(iTheta);
  float K = variance/(variance+mapVariance);

  depth;// += randn(0, 0.1);
  // printf("depth = %f\n",depth);
  // printf("K = %f\n",K);

  map(iTheta) = map(iTheta) + K*(depth-map(iTheta));
  map.setVariance(iTheta, variance*mapVariance/(variance+mapVariance));
}

void RadialDepthMap::newDisplacement(const matf & pos, const matf & sight) {
  // position
  Map new_map(map);
  double tx = pos(0,0), ty = pos(1,0);
  for (int iTheta = 0; iTheta < nBinsTheta(); ++iTheta) {
    float theta = getThetaFromITheta(iTheta);
    float rho = map(iTheta);

    // float rho = sqrt(tx*tx + ty*ty);
    // float 

    float x = rho*cos(theta) - tx;
    float y = rho*sin(theta) - ty;

    float new_rho = sqrt(x*x+y*y);
    float new_theta = atan2(y,x);
    if (new_theta < 0.0f)
      new_theta = new_theta + 2.0f*PI;
    
    // printf("theta %f\n", theta);
    // printf("new theta %f\n", new_theta);

    // printf("i %d\n", iTheta);
    // printf("new i %d\n", getIThetaFromTheta(new_theta));

    new_map(getIThetaFromTheta(new_theta)) = new_rho;

    float decay = (map.getVariance(iTheta)<MAX_VARIANCE/1.0)?1.0:1;
    new_map.setVariance(iTheta, map.getVariance(iTheta)*decay);
  }
  
  map = new_map;
  // angle
  float theta = atan2(sight(1,0), sight(0,0));
  if (theta < 0.0f)
    theta = theta + 2.0f*PI;
  theta_sight = theta;
}

void RadialDepthMap::newFrame(matf pixels) {
  int j = pixels.size().height/2;
  for (int i = 0; i < pixels.size().width; ++i)
    newPixel(i, j, pixels(j, i), 0.1, pixels.size().width, pixels.size().height);
}

mat3b RadialDepthMap::to2DMap() {
  int size = 300;
  float k = 2.0f*maxDepth;
  mat3b ret(size, size, cv::Vec3b(0.0f, 0.0f, 0.0f));
  for (int i = 1; i < 5; ++i)
    ret(150+i, 150)[1] = 255;
  ret(150, 150)[2] = 255;

  for (int iTheta = 0; iTheta < nBinsTheta(); ++iTheta) {
    float theta = getThetaFromITheta(iTheta);
    float rho = map(iTheta);

    float x = rho*cos(theta)+150;
    float y = rho*sin(theta)+150;

    if (x<size && y<size && x>0 && y>0) {
      ret(x, y)[0] = 255*(1-map.getVariance(iTheta)/MAX_VARIANCE);
    }

  }

  return ret;
}
