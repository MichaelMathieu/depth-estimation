#include "ardrone_api.h"
#include<cstdio>
#include<cstdlib>
#include<fcntl.h>
using namespace std;

ARdroneAPI::ARdroneAPI(const string & fifo_path)
  :fifo(0) {
  fifo = open(fifo_path.c_str(), O_WRONLY);
  write(fifo,"T                                ", 33);
}

ARdroneAPI::~ARdroneAPI() {
  write(fifo,"L                                ", 33);
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

void ARdroneAPI::setControl(float pitch, float gaz, float roll, float yaw) {
  char buffer[34];
  sprintf(buffer, "C%08d%08d%08d%08d", (char)roll, (char)pitch, (char)gaz, (char)yaw);
  write(fifo, buffer, 33);
}

string ARdroneAPI::toString() const {
  return "Ardrone API";
}
