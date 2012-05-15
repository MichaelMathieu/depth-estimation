#ifndef __DEPTH_MAP_H_20APR2012__
#define __DEPTH_MAP_H_20APR2012__

#include<vector>
#include<cstdlib>
#include "drone_api.h"

class DepthMap {
public:
  struct BinIndex;
  struct SphericCoordinates;
  struct CartesianCoordinates;
  struct BinRay;
  struct Map;
  
  struct BinIndex {
    DepthMap* map;
    size_t iRho, iTheta;
    inline BinIndex(DepthMap* map = NULL, size_t iRho = 0, size_t iTheta = 0);
    inline SphericCoordinates toSphericCoordinates() const;
    inline CartesianCoordinates toCartesianCoordinates() const;
    inline float getRho1() const;
    inline float getRho2() const;
    inline float getTheta1() const;
    inline float getTheta2() const;
    void getPointsInside(size_t nRho, size_t nTheta,
			 std::vector<SphericCoordinates> & out) const;
    inline float & value();
    inline const float & value() const;
  };
  struct SphericCoordinates {
    DepthMap* map;
    float rho, theta;
    inline SphericCoordinates(DepthMap* map = NULL, float rho = 0.0f, float theta = 0.0f);
    inline BinIndex toBinIndex() const;
    inline CartesianCoordinates toCartesianCoordinates() const;
  };
  struct CartesianCoordinates {
    DepthMap* map;
    float x, y;
    inline CartesianCoordinates(DepthMap* map = NULL, float x = 0.0f, float y = 0.0f);
    inline void add(float x, float y);
    inline BinIndex toBinIndex() const;
    inline SphericCoordinates toSphericCoordinates() const;
  };
  struct BinRay {
    DepthMap* map;
    float theta;
    inline BinRay(DepthMap* map = NULL, float theta = 0.0f);
    inline size_t getIBinFromDepth(float depth) const;
    inline const BinIndex operator[] (size_t i) const;
    inline BinIndex operator[] (size_t i);
  };
  class Map { //vector with smart pointers
  private:
    int* nRefs;
    std::vector<std::vector<float> >* map;
    size_t d2;
    inline void decrement();
  public:
    inline Map();
    inline Map(size_t d1, size_t d2, float init = 0.0f);
    inline Map(const Map & src);
    inline ~Map();
    inline Map & operator= (const Map & src);
    inline size_t size1 () const;
    inline size_t size2 () const;
    inline const float & operator() (size_t i, size_t j) const;
    inline float & operator() (size_t i, size_t j);
    inline Map clone() const;
  };
public:
  DepthMap(size_t nBinsRho, size_t nBinsTheta, float maxDepth, float unseenDecay,
	   float focalLength);
public:
  // if possible, do not use these functions outside BinIndex, SphericCoordinates and BinRay
  // (so that it will be easier to add features)
  inline size_t getIRhoFromRho(float rho) const;
  inline size_t getIRhoFromRho2(float rho) const;
  inline size_t getIThetaFromTheta(float theta) const;
  inline float getRhoFromIRho(size_t iRho) const;
  inline float getRho1FromIRho(size_t iRho) const;
  inline float getRho2FromIRho(size_t iRho) const;
  inline float getRhoFromIRho2(size_t iRho) const;
  inline float getRho1FromIRho2(size_t iRho) const;
  inline float getRho2FromIRho2(size_t iRho) const;
  inline float getThetaFromITheta(size_t iTheta) const; //can be > PI (by small amount)
  inline float getTheta1FromITheta(size_t iTheta) const; //never > PI
  inline float getTheta2FromITheta(size_t iTheta) const; //can be > PI (by small amount)
  
  BinRay getRayFromPixel(float x, float y, float wImg, float hImg);
  float getSafeTheta(size_t fov);
  inline size_t nBinsRho () const;
  inline size_t nBinsTheta () const;
public:
  void newPixel(float x, float y, float depth, float confidence, float WImg, float hImg);
  void newDisplacement(const matf & pos, const matf & sight); 
  void newDisplacement2(const matf & pos, const matf & sight); 
  void newFrame(matf pixels);
  std::string toString();
  mat3b to2DMap();
  mat3b to2DMap2();
private:
  float maxDepth, focalLength, unseenDecay;
  Map map;
  float theta_sight;
};

#include "depth_map.hpp"

#endif
