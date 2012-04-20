#ifndef GUI_H_
# define GUI_H_
 
# include <gtk/gtk.h>
typedef struct gui
{
  GtkWidget *window;
  GtkWidget *start;
  GtkWidget *stop;
  GtkWidget *box;
  GtkWidget *cam;
} gui_t;
 
gui_t *get_gui();
 
void init_gui(int argc, char **argv);
 
#endif
