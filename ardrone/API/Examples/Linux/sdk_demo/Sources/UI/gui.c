#include <stdlib.h>
#include "gui.h"
#include <ardrone_tool/UI/ardrone_input.h>
#include<ardrone_api.h>
#include<fcntl.h>
#include<string.h> 
 
static void idle(gpointer data) {
  g_print("Waiting for instructions...\n");
  int pipe = open("control_pipe", O_RDONLY);
  int nRead;
  const int instr_size = 33;
  char buffer[instr_size+1];
  buffer[instr_size] = '\0';
  char numberbuffer[9];
  numberbuffer[8] = '\0';
  while((nRead = read(pipe, buffer, instr_size)) != 0) {
    g_print("Recieved instruction %s\n", buffer);
    switch(buffer[0]) {
    case 'T': // take off
      ardrone_tool_set_ui_pad_start(1);
      break;
    case 'L': // land
      ardrone_tool_set_ui_pad_start(0);
      break;
    case 'C': // command (roll pitch gaz yaw)
      {
	float roll, pitch, gaz, yaw;
	strncpy(numberbuffer, buffer+1, 8);
	roll = atoi(numberbuffer);
	strncpy(numberbuffer, buffer+9, 8);
	pitch = atoi(numberbuffer);
	strncpy(numberbuffer, buffer+17, 8);
	gaz = atoi(numberbuffer);
	strncpy(numberbuffer, buffer+25, 8);
	yaw = atoi(numberbuffer);
	ardrone_at_set_progress_cmd(1, roll/100.0f, pitch/100.0f, gaz/100.0f, yaw/100.0f);
	break;
      }
    }
  }
  close(pipe);
  g_print("Pipe closed.\n");
  ardrone_tool_set_ui_pad_start(0); //landing (to be sure)
}
 
void init_gui(int argc, char **argv) {
  g_thread_init(NULL);
  gdk_threads_init();
  gtk_init(&argc, &argv);
  gtk_idle_add(idle, NULL);
}
