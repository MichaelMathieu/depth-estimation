#ifndef __RADIAL_DEPTH_MAP_H__
#define __RADIAL_DEPTH_MAP_H__

#include<vector>
#include<cstdlib>
#include <stdio.h>
#include "drone_api.h"

class RadialDepthMap {
public:
  struct SphericCoordinates;
  struct CartesianCoordinates;
  struct Ray;
  struct Map;
  
  struct SphericCoordinates {
    RadialDepthMap* map;
    float rho, theta;
    inline SphericCoordinates(RadialDepthMap* map = NULL, float rho = 0.0f, float theta = 0.0f);
    inline CartesianCoordinates toCartesianCoordinates() const;
  };
  struct CartesianCoordinates {
    RadialDepthMap* map;
    float x, y;
    inline CartesianCoordinates(RadialDepthMap* map = NULL, float x = 0.0f, float y = 0.0f);
    inline void add(float x, float y);
    inline SphericCoordinates toSphericCoordinates() const;
  };
  struct Ray {
    RadialDepthMap* map;
    float theta;
    float value;
    inline Ray(RadialDepthMap* map = NULL, float theta = 0.0f);
  };
  class Map { //vector with smart pointers
  public:
    std::vector<float>* map;
    std::vector<float>* var;
    size_t dim;
  public:
    inline Map();
    inline Map(size_t dim, float init = 0.0f);
    inline Map(const Map & src);
    inline ~Map();
    inline Map & operator= (const Map & src);
    inline size_t size () const;
    inline const float & operator() (size_t i) const;
    inline float & operator() (size_t i);
    inline float getVariance(size_t i);
    inline void setVariance(size_t i, float variance);
    inline Map clone() const;
  };
public:
  RadialDepthMap(size_t nBinsTheta, float maxDepth, float unseenDecay,
	   float focalLength);
public:
  // if possible, do not use these functions outside BinIndex, SphericCoordinates and BinRay
  // (so that it will be easier to add features)
  inline size_t getIRhoFromRho(float rho) const;
  inline size_t getIThetaFromTheta(float theta) const;
  inline float getRhoFromIRho(size_t iRho) const;
  inline float getThetaFromITheta(size_t iTheta) const; //can be > PI (by small amount)
  inline float getTheta1FromITheta(size_t iTheta) const; //never > PI
  inline float getTheta2FromITheta(size_t iTheta) const; //can be > PI (by small amount)
  
  inline size_t nBinsTheta () const;
  Ray getRayFromPixel(float x, float y, float wImg, float hImg);
public:
  void newPixel(float x, float y, float depth, float confidence, float WImg, float hImg);
  void newDisplacement(const matf & pos, const matf & sight); 
  void newFrame(matf pixels);
  std::string toString();
  mat3b to2DMap();
private:
  float maxDepth, focalLength, unseenDecay;
  Map map;
  float theta_sight;
};

#include "radial_depth_map.hpp"

#endif
