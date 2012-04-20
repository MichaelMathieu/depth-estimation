#include "ardrone_api.h"
#include "common.h"
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<fcntl.h>
using namespace std;

ARdroneAPI::ARdroneAPI(const string & fifo_path)
  :DroneAPI(), fifo(0) {
  memset(fifoBuffer, 0, sizeof(char)*(fifoBufferLen+1));
  fifo = open(fifo_path.c_str(), O_WRONLY);
}

ARdroneAPI::~ARdroneAPI() {
  land();
  close(fifo);
}


void ARdroneAPI::next() {
  
}

matf ARdroneAPI::getDepthMap() const {
  return matf(320, 240, 0.0f);
}

matf ARdroneAPI::getIMUAccel() const {
  
}

matf ARdroneAPI::getIMUGyro() const {
  
}

void ARdroneAPI::takeoff() {
  sendOnFifo(TAKEOFF);
}

void ARdroneAPI::land() {
  sendOnFifo(LAND);
}


void ARdroneAPI::setControl(float pitch, float gaz, float roll, float yaw) {
  sendOnFifo(CONTROL, pitch, gaz, roll, yaw);
}

string ARdroneAPI::toString() const {
  return "Ardrone API";
}

void ARdroneAPI::sendOnFifo(Order order, float pitch, float gaz,
			    float roll, float yaw) {
  memset(fifoBuffer, ' ', sizeof(char)*fifoBufferLen);
  switch (order) {
  case TAKEOFF:    
    fifoBuffer[0] = 'T';
    break;
  case LAND:
    fifoBuffer[0] = 'L';
    break;
  case CONTROL:
    roll = saturate(roll, -1.0f, 1.0f)*100.0f;
    gaz = saturate(gaz, -1.0f, 1.0f)*100.0f;
    pitch = saturate(pitch, -1.0f, 1.0f)*100.0f;
    yaw = saturate(yaw, -1.0f, 1.0f)*100.0f;
    sprintf(fifoBuffer, "C%08d%08d%08d%08d", (char)roll, (char)pitch, (char)gaz, (char)yaw);
    break;
  }
  write(fifo, fifoBuffer, fifoBufferLen);
}
