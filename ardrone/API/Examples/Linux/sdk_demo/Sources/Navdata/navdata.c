#include <ardrone_tool/Navdata/ardrone_navdata_client.h>

#include <Navdata/navdata.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

int navdata_pipe;
char navdata_buffer[512];

/* Initialization local variables before event loop  */
inline C_RESULT demo_navdata_client_init( void* data )
{
  navdata_pipe = open("navdata_pipe", O_WRONLY);
  return C_OK;
}

/* Receving navdata during the event loop */
inline C_RESULT demo_navdata_client_process( const navdata_unpacked_t* const navdata )
{
  
	const navdata_demo_t*nd = &navdata->navdata_demo;
	sprintf(navdata_buffer, "%010d %03d %07d %07d %07d %09d %016f %016f %016f",
		nd->ctrl_state, nd->vbat_flying_percentage,
		(int)nd->theta, (int)nd->phi, (int)nd->psi,
		nd->altitude,
		nd->vx, nd->vy, nd->vz);
	assert(strlen(navdata_buffer) < PIPE_BUF); //otherwise, write is not atomic and the reader has to be careful
	//if (strlen(navdata_buffer) != 98)
	write(navdata_pipe, navdata_buffer, 98);
	/*
	printf("=====================\nNavdata for flight demonstrations =====================\n\n");

	printf("Control state : %i\n",nd->ctrl_state);
	printf("Battery level : %i mV\n",nd->vbat_flying_percentage);
	printf("Orientation   : [Theta] %4.3f  [Phi] %4.3f  [Psi] %4.3f\n",nd->theta,nd->phi,nd->psi);
	printf("Altitude      : %i\n",nd->altitude);
	printf("Speed         : [vX] %4.3f  [vY] %4.3f  [vZPsi] %4.3f\n",nd->theta,nd->phi,nd->psi);

	printf("\033[8A");
	*/
  return C_OK;
}

/* Relinquish the local resources after the event loop exit */
inline C_RESULT demo_navdata_client_release( void )
{
  close(navdata_pipe);
  return C_OK;
}

/* Registering to navdata client */
BEGIN_NAVDATA_HANDLER_TABLE
  NAVDATA_HANDLER_TABLE_ENTRY(demo_navdata_client_init, demo_navdata_client_process, demo_navdata_client_release, NULL)
END_NAVDATA_HANDLER_TABLE

