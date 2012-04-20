#include <stdlib.h>
#include "gui.h"
#include <ardrone_tool/UI/ardrone_input.h>
#include<ardrone_api.h>
#include<fcntl.h>
#include<string.h>
 
gui_t *gui = NULL;
 
gui_t *get_gui()
{
  return gui;
}
 
 
static void buttons_callback( GtkWidget *widget,
			      gpointer   data )
{
  g_print("Waiting for instructions...\n");
  int pipe = open("test_pipe", O_RDONLY);
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
 
static void on_destroy(GtkWidget *widget, gpointer data)
{
  vp_os_free(gui);
  gtk_main_quit();
}
 
void init_gui(int argc, char **argv)
{
  gui = vp_os_malloc(sizeof (gui_t));
 
  g_thread_init(NULL);
  gdk_threads_init();
  gtk_init(&argc, &argv);
 
  gui->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  g_signal_connect(G_OBJECT(gui->window),
		   "destroy",
		   G_CALLBACK(on_destroy),
		   NULL);
  gui->box = gtk_vbox_new(FALSE, 10);
  gtk_container_add(GTK_CONTAINER(gui->window),
		    gui->box);
  gui->cam = gtk_image_new();
  gtk_box_pack_start(GTK_BOX(gui->box), gui->cam, FALSE, TRUE, 0);
 
  gui->start = gtk_button_new_with_label("Start");
  g_signal_connect (gui->start, "clicked",
		    G_CALLBACK (buttons_callback), NULL);
  gui->stop = gtk_button_new_with_label("Stop");
  g_signal_connect (gui->stop, "clicked",
		    G_CALLBACK (buttons_callback), NULL);
  gtk_widget_set_sensitive(gui->start, TRUE);
  gtk_widget_set_sensitive(gui->stop, FALSE);
 
  gtk_box_pack_start(GTK_BOX(gui->box), gui->start, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(gui->box), gui->stop, TRUE, TRUE, 0);
 
  gtk_widget_show_all(gui->window);
  
}
