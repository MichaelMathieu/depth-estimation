#include "ardrone_api.h"
#include "common.h"
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<fcntl.h>
#include<unistd.h>
#include<ostream>
#include<luaT.h>
#include<lauxlib.h>
#include<lualib.h>
#include<TH/TH.h>
using namespace std;

ARdroneAPI::ARdroneAPI(const string & control_fifo_path, const string & navdata_fifo_path)
  :DroneAPI(), control_fifo(0), navdata_fifo(0),
   last_time(getTimeInSec()), delta_t(0.0f),
   imuD(3, 1, 0.0f), imuGyro(3, 1, 0.0f),
   imuAltitude(0.0f), batteryState(100.0f), droneState(0),
   L(lua_open()), depthMap(0,0) {
  memset(controlFifoBuffer, 0, sizeof(char)*(controlFifoBufferLen+1));
  memset(navdataFifoBuffer, 0, sizeof(char)*(navdataFifoBufferLen+1));
  printf("Getting control FIFO\n");
  control_fifo = open(control_fifo_path.c_str(), O_WRONLY);
  printf("Getting navdata FIFO\n");
  navdata_fifo = open(navdata_fifo_path.c_str(), O_RDONLY | O_NDELAY);
  //navdata_fifo = open(navdata_fifo_path.c_str(), O_RDONLY);
  printf("All good!\n");
  
  luaL_openlibs(L);
  luaL_dofile(L, "../depth_estimation_api.lua");
  lua_pushnil(L);
}

ARdroneAPI::~ARdroneAPI() {
  land();
  close(control_fifo);
  close(navdata_fifo);
  lua_close(L);
}


void ARdroneAPI::next() {
  float vx, vy, vz;
  int gx, gy, gz, bs, a;
  double time = getTimeInSec();
  delta_t = (float)(time - last_time);
  last_time = time;
  while (read(navdata_fifo, navdataFifoBuffer, navdataFifoBufferLen) == navdataFifoBufferLen) {
    sscanf(navdataFifoBuffer, "%d %d %d %d %d %d %f %f %f", &droneState, &bs,
	   &gx, &gy, &gz, &a, &vx, &vy, &vz);
    batteryState = bs;
    imuAltitude = a;
    imuGyro(0,0) = gx;
    imuGyro(1,0) = gy;
    imuGyro(2,0) = gz;
    imuD(0,0) = vx;
    imuD(1,0) = vy;
    imuD(2,0) = vz;
  }
  imuD *= delta_t;

  lua_pop(L, 1);
  lua_getfield(L, LUA_GLOBALSINDEX, "nextFrameDepth");
  lua_call(L, 0, 1);
  THFloatTensor *depth_th = (THFloatTensor*)luaT_toudata(L,-1,luaT_checktypename2id(L, "torch.FloatTensor"));
  depth_th = THFloatTensor_newContiguous(depth_th);
  depthMap = matf(depth_th->size[0], depth_th->size[1], THFloatTensor_data(depth_th));
}

float ARdroneAPI::getDeltaT() const {
  return delta_t;
}

matf ARdroneAPI::getDepthMap() const {
  return depthMap;
}

matf ARdroneAPI::getIMUTranslation() const {
  return imuD;
}

matf ARdroneAPI::getIMUGyro() const {
  return imuGyro;
}

float ARdroneAPI::getIMUAltitude() const {
  return imuAltitude;
}

float ARdroneAPI::getBatteryState() const {
  return batteryState;
}

int ARdroneAPI::getDroneState() const {
  return droneState;
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
  ostringstream oss;
  char buffer[128];
  oss << "ARdroneAPI:\n";
  sprintf(buffer, "  d      = (%.5f %.5f %.5f)", imuD(0,0), imuD(1,0), imuD(2,0));
  oss << buffer << "\n";
  sprintf(buffer, " gyro    = (%.5f %.5f %.5f)", imuGyro(0,0), imuGyro(1,0), imuGyro(2,0));
  oss << buffer << "\n";
  sprintf(buffer, "  alt    = %.5f", imuAltitude);
  oss << buffer << "\n";
  sprintf(buffer, "  battery= %f%%", batteryState);
  oss << buffer << "\n";
  sprintf(buffer, "  state  = %d", droneState);
  oss << buffer << "\n";
  return oss.str();
}

void ARdroneAPI::sendOnFifo(Order order, float pitch, float gaz,
			    float roll, float yaw) {
  memset(controlFifoBuffer, ' ', sizeof(char)*controlFifoBufferLen);
  switch (order) {
  case TAKEOFF:    
    controlFifoBuffer[0] = 'T';
    break;
  case LAND:
    controlFifoBuffer[0] = 'L';
    break;
  case CONTROL:
    roll = saturate(roll, -1.0f, 1.0f)*100.0f;
    gaz = saturate(gaz, -1.0f, 1.0f)*100.0f;
    pitch = saturate(pitch, -1.0f, 1.0f)*100.0f;
    yaw = saturate(yaw, -1.0f, 1.0f)*100.0f;
    sprintf(controlFifoBuffer, "C%08d%08d%08d%08d", (char)roll, (char)pitch, (char)gaz, (char)yaw);
    break;
  }
  write(control_fifo, controlFifoBuffer, controlFifoBufferLen);
}
