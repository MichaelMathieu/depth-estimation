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

// void DepthMap::BinIndex::getNeighbors(vector<SphericCoordinates> & out) const {
//   float rho1 = getRho1(), rho2 = getRho2();
//   float theta1 = getTheta1(), theta2 = getTheta2();
//   float drho = rho2 - rho1, dtheta = theta2-theta1;
//   const float epsilon = 0.01f;
//   out.resize(4);
//   for (size_t iRho = 0; iRho < nRho; ++iRho)
//     for (size_t iTheta = 0; iTheta < nTheta; ++iTheta)
//       out[iRho+nRho*iTheta] =
//   SphericCoordinates(map,
//          rho1+((float)iRho+epsilon)/((float)nRho-1.0f+2*epsilon)*drho,
//          theta1+((float)iTheta+epsilon)/((float)nTheta-1.0f+2*epsilon)*dtheta);
// }

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
  BinRay ray = getRayFromPixel(x, y, wImg, hImg);
  size_t iBin = ray.getIBinFromDepth(depth);
  for (size_t i = 0; i < iBin; ++i)
    ray[i].value() = 0.5f * (ray[i].value() + 1.0f - confidence);
  ray[iBin].value() = 0.5f * (ray[iBin].value() + confidence);
}

float DepthMap::getSafeTheta(size_t fov) {
  assert(fov<nBinsTheta());
  float safeTheta = 0;
  int iniTheta = floor((nBinsTheta()-fov)/2);
  int endTheta = iniTheta+fov;
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
    if (maxConfidenceBin < 5) {
      float theta = ((float)(iTheta) / (float)(nBinsTheta()-1) - 0.5f) * 2.0f * PI;
      return -theta; 
    }
    // float theta = ((float)(iTheta) / (float)(nBinsTheta()-1) - 0.5f) * 2.0f * PI;
    // printf("theta = %f, maxBin = %d\n", theta, maxConfidenceBin);
    // safeTheta += theta*maxConfidenceBin/nBinsRho();

  }
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
      for (size_t iPt = 0; iPt < coords_tmp.size(); ++iPt) {
      	CartesianCoordinates pt = coords_tmp[iPt].toCartesianCoordinates();
      	pt.add(pos(0,0), pos(1,0));
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

void DepthMap::newDisplacement2(const matf & pos, const matf & sight) {
  // position
  Map new_map(nBinsTheta(), nBinsRho());
  double tx = pos(0,0), ty = pos(1,0);
  SphericCoordinates coords_tmp;
  for (int iTheta = 0; iTheta < nBinsTheta(); ++iTheta) {
    for (int iRho = 0; iRho < nBinsRho(); ++iRho) {
      
      float new_rho = getRhoFromIRho(iRho);
      float new_theta = getThetaFromITheta(iTheta);

      coords_tmp = SphericCoordinates(this, new_rho, new_theta);
      CartesianCoordinates pt = coords_tmp.toCartesianCoordinates();
      pt.add(tx, ty);
      
      BinIndex bin = pt.toBinIndex();

      float theta1 = bin.getTheta1();
      float theta2 = bin.getTheta2();
      float dtheta = theta2 - theta1;
      float rho1 = bin.getRho1();
      float rho2 = bin.getRho2();
      float drho = rho2 - rho1;

      float r1 = (coords_tmp.rho - rho1)/drho;
      float r2 = (rho2 - coords_tmp.rho)/drho;
      float t1 = (coords_tmp.theta - theta1)/dtheta;
      float t2 = (theta2 - coords_tmp.theta)/dtheta;

      // printf("--\n");
      // printf("bin\n");
      // printf("iTheta: %d, iRho: %d\n", bin.iTheta, bin.iRho);
      // printf("theta: %f, rho: %f\n", pt.toSphericCoordinates().rho, pt.toSphericCoordinates().theta);
      // printf("new bin\n");
      // printf("iTheta: %d, iRho: %d\n", iTheta, iRho);
      // printf("theta: %f, rho: %f\n", new_rho, new_theta);

      // new_map(iTheta, iRho) = bin.value();

      if (bin.iRho<nBinsRho()-1) {
        new_map(iTheta, iRho) = map(bin.iTheta, bin.iRho)*r1*t1;
        new_map(iTheta, iRho) += map(bin.iTheta, bin.iRho+1)*r2*t1;
        if (bin.iTheta<nBinsTheta()-1) {
          new_map(iTheta, iRho) += map(bin.iTheta+1, bin.iRho)*r1*t2;
          new_map(iTheta, iRho) += map(bin.iTheta+1, bin.iRho+1)*r2*t2;
        }
        else {
          new_map(iTheta, iRho) += map(0, bin.iRho)*r1*t2;
          new_map(iTheta, iRho) += map(0, bin.iRho+1)*r2*t2;
        }
      }
      else {
        new_map(iTheta, iRho) = map(bin.iTheta, bin.iRho)*t1;
        if (bin.iTheta<nBinsTheta()-1) {
          new_map(iTheta, iRho) += map(bin.iTheta+1, bin.iRho)*t2;
        }
        else {
          new_map(iTheta, iRho) += map(0, bin.iRho)*t2;
        }
      }
      
      //float decay = (new_map(newBin.iRho, newBin.iTheta)<MAX_VARIANCE/1.01)?1.01:1;
      // new_map(newBin.iRho, newBin.iTheta) *= decay;

    }  
  }
  map = new_map;
  // angle
  float theta = atan2(sight(1,0), sight(0,0));
  if (theta < 0.0f)
    theta = theta + 2.0f*PI;
  theta_sight = theta;
}

void DepthMap::newFrame(matf pixels) {
  int j = pixels.size().height/2;
  for (int i = 0; i < pixels.size().width; ++i)
    newPixel(i, j, pixels(j, i), 1, pixels.size().width, pixels.size().height);
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

mat3b DepthMap::to2DMap2() {
  int size = 300;
  float k = 2.0f*maxDepth;
  mat3b ret(size, size, cv::Vec3b(0.0f, 0.0f, 0.0f));
  for (int i = 1; i < 5; ++i)
    ret(150+i, 150)[1] = 255;
  ret(150, 150)[2] = 255;
  for (int iTheta = 0; iTheta < nBinsTheta(); ++iTheta) {
    for (int iRho = 0; iRho < nBinsRho(); ++iRho) {
      float rho = getRhoFromIRho2(iRho);
      float theta = getThetaFromITheta(iTheta);
      SphericCoordinates coords_tmp = SphericCoordinates(this, rho, theta);
      CartesianCoordinates pt = coords_tmp.toCartesianCoordinates();
      BinIndex bin = pt.toBinIndex();
      int i = pt.x+150, j = pt.y+150;
      if (i<size && j<size && i>0 && j>0) 
        ret(i, j) = (unsigned char)(255.0f*bin.value());
      // ret(pt.x+1, pt.y) = (unsigned char)(255.0f*bin.value());
      // ret(pt.x, pt.y+1) = (unsigned char)(255.0f*bin.value());
      // ret(pt.x+1, pt.y+1) = (unsigned char)(255.0f*bin.value());
      
    }
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
