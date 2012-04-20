#include "simulator.h"
#include "ardrone_api.h"
#include <GL/glut.h>
#include <iostream>
#include <ctime>
using namespace std;

int main_window_id;
float dyaw = 0.0f, pitch =0.0f, roll = 0.0f, gaz = 0.0f;
DroneAPI* pApi = NULL;
GLuint map_texture;

void keyboard(int key, bool special, bool down) {
  if (special) {
    switch(key) {
    case GLUT_KEY_LEFT:
      //turn left
      if (down) dyaw = -0.25f; else dyaw = 0.0f;
      break;
    case GLUT_KEY_RIGHT:
      //turn right
      if (down) dyaw = 0.25f; else dyaw = 0.0f;
      break;
    case GLUT_KEY_UP:
      //move up
      if (down) gaz = 1.0f; else gaz = 0.0f;
      break;
    case GLUT_KEY_DOWN:
      //move down
      if (down) gaz = -1.0f; else gaz = 0.0f;
      break;
    }
  } else {
    switch (key) {
    case 'a':
      //move left
      if (down) roll = -1.0f; else roll = 0.0f;
      break;
    case 'd':
      //move right
      if (down) roll = 1.0f; else roll = 0.0f;
      break;
    case 'w':
      //move forward
      if (down) pitch = 1.0f; else pitch = 0.0f;
      break;
    case 's':
      //move backward
      if (down) pitch = -1.0f; else pitch = 0.0f;
      break;
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

void idle() {
  pApi->next();
  cout << pitch << " " << roll << " " << dyaw << " " << gaz << endl;
  cout << pApi->toString() << endl;
  matf map = 0.1*pApi->getDepthMap();
  glDrawPixels(320, 240, GL_LUMINANCE, GL_FLOAT, (float*)map.data);
  glutSwapBuffers();
  usleep(100000);
}

void render() {
  
}

int main(int argc, char* argv[]) {
  //SimulatedAPI api(320, 240);
  ARdroneAPI api("API/Examples/Linux/Build/Release/test_pipe");
  pApi = &api;
  glutInit(&argc, argv);
  glutInitWindowPosition(0,0);
  glutInitWindowSize(320, 240);
  glutInitDisplayMode(GLUT_LUMINANCE | GLUT_DOUBLE);
  main_window_id = glutCreateWindow("depth");
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
