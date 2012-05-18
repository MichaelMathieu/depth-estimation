#include "depth_map.h"
// #include "radial_map.h"
// #include "bin_map.h"
// #include "point_map.h"

#include "simulator.h"
#include "ardrone_api.h"
#include <GL/glut.h>
#include <iostream>
#include <ctime>
using namespace std;

int main_window_id;
int win_w = 320, win_h = 180;
float dyaw = 0.0f, pitch =0.0f, roll = 0.0f, gaz = 0.0f;
bool flying = false;
DroneAPI* pApi = NULL;
GLuint map_texture;
DepthMap* pMap = NULL;
// BinMap* pMap = NULL;
// PointMap* pMap = NULL;

void keyboard(int key, bool special, bool down) {
  if (special) {
    switch(key) {
    case GLUT_KEY_LEFT:
      //turn left
      if (down) dyaw = -0.3f; else dyaw = 0.0f;
      break;
    case GLUT_KEY_RIGHT:
      //turn right
      if (down) dyaw = 0.3f; else dyaw = 0.0f;
      break;
    case GLUT_KEY_UP:
      //move up
      if (down) gaz = 100.0f; else gaz = 0.0f;
      break;
    case GLUT_KEY_DOWN:
      //move down
      if (down) gaz = -100.0f; else gaz = 0.0f;
      break;
    }
  } else {
    switch (key) {
    case 'a':
      //move left
      if (down) roll = -0.3f; else roll = 0.0f;
      break;
    case 'd':
      //move right
      if (down) roll = 0.3f; else roll = 0.0f;
      break;
    case 'w':
      //move forward
      if (down) pitch = 0.5f; else pitch = 0.0f;
      break;
    case 's':
      //move backward
      if (down) pitch = -0.3f; else pitch = 0.0f;
      break;
    case ' ':
      //takeoff/land
      if (down) {
	if (flying) pApi->land(); else pApi->takeoff();
	flying = !flying;
      }
    }
  }
  pApi->setControl(pitch, gaz, roll, dyaw);
}

void keyboardDown1(unsigned char key, int, int) {
  keyboard(key, false, true);
}
void keyboardDown2(int key, int, int) {
  keyboard(key, true, true);
}
void keyboardUp1(unsigned char key, int, int) {
  keyboard(key, false, false);
}
void keyboardUp2(int key, int, int) {
  keyboard(key, true, false);
}

#include "opencv/highgui.h"
void idle() {

  float safeTheta = pMap->getSafeTheta(16);
  //printf("safeTheta = %f\n", safeTheta);

  keyboard('w', false, true);
  //roll = safeTheta*0.1f;
  if (safeTheta>0) {
    keyboard('d', false, true);
    keyboard(GLUT_KEY_RIGHT, true, true);
  }
  if (safeTheta<0) {
    keyboard('a', false, true);
    keyboard(GLUT_KEY_LEFT, true, true);
  }
  if (safeTheta == 0) {
    keyboard('a', false, false);
    keyboard('d', false, false);
    keyboard(GLUT_KEY_RIGHT, true, false);
    keyboard(GLUT_KEY_LEFT, true, false);
  }

  printf("avt next\n");
  pApi->next();
  printf("apres next\n");
  cout << pitch << " " << roll << " " << dyaw << " " << gaz << endl;
  cout << pApi->toString() << endl;
  matf frameDMap = pApi->getDepthMap();
  
  if ((frameDMap.size().height != win_h) || (frameDMap.size().width != win_w)) {
    win_w = frameDMap.size().width;
    win_h = frameDMap.size().height;
    printf("%d %d\n", win_h, win_w);
    glutReshapeWindow(win_w, win_h);
  }

  //glDrawPixels(win_w, win_h, GL_LUMINANCE, GL_FLOAT, (float*)((matf)(0.01f*frameDMap)).data);
  //glutSwapBuffers();

  // Displacement disp(pApi->getFilteredTranslation(), pApi->getIMUGyro());
  // pMap->newDisplacement(disp);
  pMap->newDisplacement(pApi->getFilteredTranslation(), pApi->getIMUGyro());  
  pMap->newFrame(frameDMap);

  cv::namedWindow("depth map");
  double m;
  minMaxLoc(frameDMap, NULL, &m);
  //m = 100.;
  cv::imshow("depth map", frameDMap / m);
  cv::namedWindow("2d map");
  cv::imshow("2d map", pMap->to2DMap());
  cvWaitKey(1);
  //cout << pMap->toString() << endl;
  //usleep(100000);
}

void render() {
  
}

int main(int argc, char* argv[]) {
  SimulatedAPI api(320, 240);
  // ARdroneAPI api("control_pipe", "navdata_pipe");
  pApi = &api;
  
  DepthMap map(64, 128, 100, 0.9f, 320);
  //PointMap map(16, 16, 100, 0.9f, 320);
  pMap = &map;
  glutInit(&argc, argv);
  glutInitWindowPosition(0,0);
  glutInitWindowSize(win_w, win_h);
  glutInitDisplayMode(GLUT_LUMINANCE | GLUT_DOUBLE);
  main_window_id = glutCreateWindow("depth map");
  glutIdleFunc(idle);
  glutDisplayFunc(render);
  glutKeyboardFunc(keyboardDown1);
  glutSpecialFunc(keyboardDown2);
  glutKeyboardUpFunc(keyboardUp1);
  glutSpecialUpFunc(keyboardUp2);

  glGenTextures(1, &map_texture);

  glutMainLoop();
  return 0;
}
