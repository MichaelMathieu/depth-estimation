#include "depth_map.h"
#include "common.h"
#include "simulator.h"
#include<cmath>
#include<ostream>
#include<iostream>
using namespace std;

void DepthMap::BinIndex::getPointsInside(size_t nRho, size_t nTheta,
					 vector<SphericCoordinates> & out) const {
  float rho1 = getRho1(), rho2 = getRho2();
  float theta1 = getTheta1(), theta2 = getTheta2();
  float drho = rho2 - rho1, dtheta = theta2-theta1;
  const float epsilon = 0.01f;
  out.resize(nRho*nTheta);
  for (size_t iRho = 0; iRho < nRho; ++iRho)
    for (size_t iTheta = 0; iTheta < nTheta; ++iTheta)
      out[iRho+nRho*iTheta] =
	SphericCoordinates(map,
			   rho1+((float)iRho+epsilon)/((float)nRho-1.0f+2*epsilon)*drho,
			   theta1+((float)iTheta+epsilon)/((float)nTheta-1.0f+2*epsilon)*dtheta);
}

DepthMap::DepthMap(size_t nBinsRho, size_t nBinsTheta, float maxDepth, float unseenDecay,
		   float focalLength)
  :maxDepth(maxDepth), focalLength(focalLength), unseenDecay(unseenDecay),
   map(nBinsTheta, nBinsRho), theta_sight(0.0f) {
  assert((nBinsRho > 0) && (nBinsTheta > 0));
};

DepthMap::BinRay DepthMap::getRayFromPixel(float x, float y, float wImg, float hImg) {
  float theta = atan2(x-wImg/2, focalLength);
  return BinRay(this, theta);
}

void DepthMap::newPixel(float x, float y, float depth, float confidence,
			float wImg, float hImg) {
  float lambda = 0.9f;
  BinRay ray = getRayFromPixel(x, y, wImg, hImg);
  size_t iBin = ray.getIBinFromDepth(depth);
  for (size_t i = 0; i < iBin; ++i)
    ray[i].value() = lambda * ray[i].value() + (1.0f - lambda) * (1.0f - confidence);
  ray[iBin].value() = lambda * ray[iBin].value() + (1.0f - lambda) * confidence;
}

float DepthMap::getSafeTheta(size_t fov) {
  assert(fov<nBinsTheta());
  float safeTheta = 0;
  int steerTheta = getIThetaFromTheta(theta_sight); 
  steerTheta = (steerTheta<nBinsTheta())?steerTheta:steerTheta-nBinsTheta();
  printf("steer bin = %d\n", steerTheta);
  int iniTheta = floor(steerTheta-fov/2);
  int endTheta = iniTheta+fov;
  size_t closestBin = nBinsRho()-1;
  for (int iTheta=iniTheta; iTheta<endTheta; iTheta++) {
    // printf("iTheta = %d\n", iTheta);
    float maxConfidence = 1e-1;
    size_t maxConfidenceBin = nBinsRho()-1;
    for (int iRho=0; iRho<nBinsRho(); iRho++) {
      // printf("iRho = %d\n", iRho);
      float confidence = map(iTheta, iRho);
      if (confidence>maxConfidence) {
        maxConfidence = confidence;
        maxConfidenceBin = iRho;
      }
    }
    if (maxConfidenceBin < closestBin) {
      closestBin = maxConfidenceBin;
      safeTheta = -((float)(iTheta) / (float)(nBinsTheta()-1) - 0.5f) * 2.0f * PI;
    }
    // float theta = ((float)(iTheta) / (float)(nBinsTheta()-1) - 0.5f) * 2.0f * PI;
    // printf("theta = %f, maxBin = %d\n", theta, maxConfidenceBin);
    // safeTheta += theta*maxConfidenceBin/nBinsRho();

  }
  printf("closest bin = %d\n", closestBin);
  if (closestBin<(nBinsRho())/5)
    return safeTheta;
  else
    return 0;
}

void DepthMap::newDisplacement(const matf & pos, const matf & sight) {
  // position
  Map new_map(nBinsTheta(), nBinsRho());
  double tx = pos(0,0), ty = pos(1,0);
  vector<SphericCoordinates> coords_tmp;
  for (int iTheta = 0; iTheta < nBinsTheta(); ++iTheta) {
    for (int iRho = 0; iRho < nBinsRho(); ++iRho) {
     
      BinIndex newBin = BinIndex(this, iRho, iTheta);
      newBin.getPointsInside(5, 5, coords_tmp);
      float value = 0.0f;
      // printf("iTheta = %d, iRho = %d\n", iTheta, iRho);
      for (size_t iPt = 0; iPt < coords_tmp.size(); ++iPt) {
      	CartesianCoordinates pt = coords_tmp[iPt].toCartesianCoordinates();
      	pt.add(pos(0,0), pos(1,0));
        // printf("iPt = %d\n", iPt);
      	value += pt.toBinIndex().value();
      }
      new_map(newBin.iTheta, newBin.iRho) = value/(float)coords_tmp.size();//*unseenDecay;
    }
  }
  map = new_map;
  // angle
  float theta = atan2(sight(1,0), sight(0,0));
  if (theta < 0.0f)
    theta = theta + 2.0f*PI;
  theta_sight = theta;
}

void DepthMap::newFrame(const matf & pixels, const matf & confidence) {
  int w = pixels.size().width, h = pixels.size().height;
  
  // int jmin = h/4-20;
  // int jmax = h/4+20;
  int jmin = h/2-1;
  int jmax = h/2;
  for (int j = jmin; j < jmax; ++j)
    for (int i = 0; i < w; ++i)
      if (confidence(j, i) > 0.5f) {
      	newPixel(i, j, pixels(j, i), 1, pixels.size().width, pixels.size().height);
      }
}

mat3b DepthMap::to2DMap() {
  int size = 300;
  float k = 2.0f*maxDepth;
  mat3b ret(size, size, cv::Vec3b(0.0f, 0.0f, 0.0f));
  for (int i = 1; i < 5; ++i)
    ret(150+i, 150)[1] = 255;
  ret(150, 150)[2] = 255;
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j) {
      float x = ((float)i/(float)size-0.5f)*k;
      float y = ((float)j/(float)size-0.5f)*k;
      if (x == 0 and y == 0)
	continue;
      BinIndex bin = CartesianCoordinates(this, x, y).toBinIndex();
      ret(i, j)[0] = max(ret(i, j)[0], (unsigned char)(255.0f*bin.value()));
    }
  return ret;
}

string DepthMap::toString() {
  ostringstream oss;
  char buffer[128];
  oss << "DepthMap\n";
  for (int i = 0; i < nBinsRho(); ++i) {
    for (int j = 0; j < nBinsTheta(); ++j)
      oss << BinIndex(this, i, j).value() << " ";
    oss << endl;
  }
        
  oss << "\n";
  return oss.str();
}
