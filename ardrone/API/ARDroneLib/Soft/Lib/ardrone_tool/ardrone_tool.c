#include <VP_Os/vp_os_malloc.h>
#include <VP_Os/vp_os_print.h>
#include <VP_Api/vp_api_thread_helper.h>

#include <ardrone_tool/ardrone_tool.h>
#include <ardrone_tool/ardrone_time.h>
#include <ardrone_tool/ardrone_tool_configuration.h>
#include <ardrone_tool/Navdata/ardrone_navdata_client.h>
#include <ardrone_tool/UI/ardrone_input.h>
#include <ardrone_tool/Com/config_com.h>

#include <utils/ardrone_gen_ids.h>

int32_t MiscVar[NB_MISC_VARS] = { 
               DEFAULT_MISC1_VALUE, 
               DEFAULT_MISC2_VALUE,
               DEFAULT_MISC3_VALUE, 
               DEFAULT_MISC4_VALUE
                                };

//static bool_t need_update   = TRUE;
static ardrone_timer_t ardrone_tool_timer;
static int ArdroneToolRefreshTimeInUs = ARDRONE_REFRESH_MS * 1000;
static vp_os_mutex_t ardrone_tool_mutex;
static bool_t ardrone_tool_in_pause = FALSE;
char wifi_ardrone_ip[256] = { WIFI_ARDRONE_IP };
char app_id [MULTICONFIG_ID_SIZE] = "00000000"; // Default application ID.
char app_name [APPLI_NAME_SIZE] = "Default application"; // Default application name.
char usr_id [MULTICONFIG_ID_SIZE] = "00000000"; // Default user ID.
char usr_name [USER_NAME_SIZE] = "Default user"; // Default user name.
char ses_id [MULTICONFIG_ID_SIZE] = "00000000"; // Default session ID.
char ses_name [SESSION_NAME_SIZE] = "Default session"; // Default session name.

#ifndef __SDK_VERSION__
#define __SDK_VERSION__ "1.8" // TEMPORARY LOCATION OF __SDK_VERSION__ !!!
#endif


int usleep(unsigned int usec);

static bool_t send_com_watchdog = FALSE;

void ardrone_tool_send_com_watchdog( void )
{
  send_com_watchdog = TRUE;
}

#ifndef NO_ARDRONE_MAINLOOP
static void ardrone_tool_usage( const char* appname )
{
  printf("%s based on ARDrone Tool\n", appname);
  printf("Be aware to not insert space in your options\n");

  ardrone_tool_display_cmd_line_custom();
}
#endif

static void ardrone_toy_network_adapter_cb( const char* name )
{
	strcpy( COM_CONFIG_NAVDATA()->itfName, name );
}

C_RESULT ardrone_tool_setup_com( const char* ssid )
{
  C_RESULT res = C_OK;

#ifdef CHECK_WIFI_CONFIG
  if( FAILED(vp_com_init(COM_NAVDATA())) )
  {
	  DEBUG_PRINT_SDK("VP_Com : Failed to init com for navdata\n");
	  vp_com_shutdown(COM_NAVDATA());
	  res = C_FAIL;
  }

  vp_com_network_adapter_lookup(COM_NAVDATA(), ardrone_toy_network_adapter_cb);

  if( SUCCEED(res) && FAILED(vp_com_local_config(COM_NAVDATA(), COM_CONFIG_NAVDATA())) )
  {
	  DEBUG_PRINT_SDK("VP_Com : Failed to configure com for navdata\n");
	  vp_com_shutdown(COM_NAVDATA());
	  res = C_FAIL;
  }

  if( ssid != NULL )
  {
	  strcpy( ((vp_com_wifi_connection_t*)wifi_connection())->networkName, ssid );
  }

  if( SUCCEED(res) && FAILED(vp_com_connect(COM_NAVDATA(), COM_CONNECTION_NAVDATA(), NUM_ATTEMPTS)))
  {
	  DEBUG_PRINT_SDK("VP_Com: Failed to connect for navdata\n");
	  vp_com_shutdown(COM_NAVDATA());
	  res = C_FAIL;
  }
#else  
  vp_com_init(COM_NAVDATA());
  vp_com_network_adapter_lookup(COM_NAVDATA(), ardrone_toy_network_adapter_cb);
  vp_com_local_config(COM_NAVDATA(), COM_CONFIG_NAVDATA());

  if( ssid != NULL )
  {
	  strcpy( ((vp_com_wifi_connection_t*)wifi_connection())->networkName, ssid );
  }

  vp_com_connect(COM_NAVDATA(), COM_CONNECTION_NAVDATA(), NUM_ATTEMPTS);
  ((vp_com_wifi_connection_t*)wifi_connection())->is_up=1;
#endif

  return res;
}

#ifdef NO_ARDRONE_MAINLOOP
C_RESULT ardrone_tool_init( const char* ardrone_ip, size_t n, AT_CODEC_FUNCTIONS_PTRS *ptrs, const char *appname, const char *usrname)
{	
	// Initalize mutex and condition
	vp_os_mutex_init(&ardrone_tool_mutex);
	ardrone_tool_in_pause = FALSE;

	// Initialize ardrone_control_config structures;
	ardrone_tool_reset_configuration();
	// ardrone_control_config_default initialisation. Sould not be modified after that !
	vp_os_memcpy ((void *)&ardrone_control_config_default, (const void *)&ardrone_control_config, sizeof (ardrone_control_config_default));
	// initialization of application defined default values
	vp_os_memcpy ((void *)&ardrone_application_default_config, (const void *)&ardrone_control_config, sizeof (ardrone_application_default_config));
	
	//Fill structure AT codec and built the library AT commands.
   if( ptrs != NULL )
	   ardrone_at_init_with_funcs( ardrone_ip, n, ptrs );
   else	
      ardrone_at_init( ardrone_ip, n );

	// Save appname/appid for reconnections
	if (NULL != appname)
	{
	  ardrone_gen_appid (appname, __SDK_VERSION__, app_id, app_name, sizeof (app_name));
	}
	// Save usrname/usrid for reconnections
	if (NULL != usrname)
	{
		ardrone_gen_usrid (usrname, usr_id, usr_name, sizeof (usr_name));
	}
	// Create pseudorandom session id
	ardrone_gen_sessionid (ses_id, ses_name, sizeof (ses_name));

	// Init subsystems
	ardrone_timer_reset(&ardrone_tool_timer);
	ardrone_timer_update(&ardrone_tool_timer);
	
	ardrone_tool_input_init();
	ardrone_control_init();
	ardrone_tool_configuration_init();
	ardrone_navdata_client_init();


   //Opens a connection to AT port.
	ardrone_at_open();

	START_THREAD(navdata_update, 0);
	START_THREAD(ardrone_control, 0);

	// Send start up configuration
	ardrone_at_set_pmode( MiscVar[0] );
	ardrone_at_set_ui_misc( MiscVar[0], MiscVar[1], MiscVar[2], MiscVar[3] );

	return C_OK;
}
#else
C_RESULT ardrone_tool_init(int argc, char **argv)
{
	C_RESULT res;
	int32_t b_value = FALSE;

	// Initalize mutex and condition
	vp_os_mutex_init(&ardrone_tool_mutex);
	ardrone_tool_in_pause = FALSE;

	// Initialize ardrone_control_config structures;
	ardrone_tool_reset_configuration();
	// ardrone_control_config_default initialisation. Sould not be modified after that !
	vp_os_memcpy ((void *)&ardrone_control_config_default, (const void *)&ardrone_control_config, sizeof (ardrone_control_config_default));
	// initialization of application defined default values
	vp_os_memcpy ((void *)&ardrone_application_default_config, (const void *)&ardrone_control_config, sizeof (ardrone_application_default_config));
	ardrone_application_default_config.navdata_demo = b_value;

	// Save appname/appid for reconnections
	if (NULL != argv[0])
	{
	  char *appname = NULL;
	  int lastSlashPos;
	  /* Cut the invoking name to the last / or \ character on the command line
	   * This avoids using differents app_id for applications called from different directories
	   * e.g. if argv[0] is "Build/Release/ardrone_navigation", appname will point to "ardrone_navigation" only
	   */
	  for (lastSlashPos = strlen (argv[0])-1; 
	       lastSlashPos > 0 && 
		 argv[0][lastSlashPos] != '/' && 
		 argv[0][lastSlashPos] != '\\'; 
	       lastSlashPos--);
	  appname = &argv[0][lastSlashPos+1];
	  ardrone_gen_appid (appname, __SDK_VERSION__, app_id, app_name, sizeof (app_name));
	}

	// Create pseudorandom session id
	ardrone_gen_sessionid (ses_id, ses_name, sizeof (ses_name));
	
	//Fill structure AT codec and built the library AT commands.
	ardrone_at_init( wifi_ardrone_ip, strlen( wifi_ardrone_ip) );

	// Init subsystems
	ardrone_timer_reset(&ardrone_tool_timer);
	ardrone_timer_update(&ardrone_tool_timer);
	
	ardrone_tool_input_init();
	ardrone_control_init();
	ardrone_tool_configuration_init();
	ardrone_navdata_client_init();

	// Init custom tool
	res = ardrone_tool_init_custom(argc, argv);

   //Opens a connection to AT port.
	ardrone_at_open();

	START_THREAD(navdata_update, 0);
	START_THREAD(ardrone_control, 0);

	// Send start up configuration
	ardrone_at_set_pmode( MiscVar[0] );
	ardrone_at_set_ui_misc( MiscVar[0], MiscVar[1], MiscVar[2], MiscVar[3] );
	
	return res;
}
#endif

C_RESULT ardrone_tool_set_refresh_time(int refresh_time_in_ms)
{
  ArdroneToolRefreshTimeInUs = refresh_time_in_ms * 1000;

  return C_OK;
}

C_RESULT ardrone_tool_pause( void )
{
	ardrone_navdata_client_suspend();

	vp_os_mutex_lock(&ardrone_tool_mutex);
	ardrone_tool_in_pause = TRUE;
	vp_os_mutex_unlock(&ardrone_tool_mutex);	
	
	return C_OK;
}

C_RESULT ardrone_tool_resume( void )
{
   ardrone_navdata_client_resume();

	vp_os_mutex_lock(&ardrone_tool_mutex);
	ardrone_tool_in_pause = FALSE;
	vp_os_mutex_unlock(&ardrone_tool_mutex);	
	
   return C_OK;
}

C_RESULT ardrone_tool_update()
{
	int delta;

	C_RESULT res = C_OK;
	
	delta = ardrone_timer_delta_us(&ardrone_tool_timer);
	if( delta >= ArdroneToolRefreshTimeInUs)
	{
		// Render frame
		ardrone_timer_update(&ardrone_tool_timer);
		
		if(!ardrone_tool_in_pause)
		{
			ardrone_tool_input_update();
			res = ardrone_tool_update_custom();
		}
		
		if( send_com_watchdog == TRUE )
		{
			ardrone_at_reset_com_watchdog();
			send_com_watchdog = FALSE;
		}
		
		// Send all pushed messages
		ardrone_at_send();
		
		res = ardrone_tool_display_custom();
	}
	else
	{
		usleep(ArdroneToolRefreshTimeInUs - delta);
	}

	return res;
}

C_RESULT ardrone_tool_shutdown()
{
  C_RESULT res = C_OK;
  
#ifndef NO_ARDRONE_MAINLOOP
  res = ardrone_tool_shutdown_custom();
#endif

  // Shutdown subsystems
  ardrone_navdata_client_shutdown();
  ardrone_control_shutdown();
  ardrone_tool_input_shutdown();
 
  JOIN_THREAD(ardrone_control); 
  JOIN_THREAD(navdata_update);

  // Shutdown AT Commands
  ATcodec_exit_thread();
  ATcodec_Shutdown_Library();

  vp_com_disconnect(COM_NAVDATA());
  vp_com_shutdown(COM_NAVDATA());

  PRINT("Custom ardrone tool ended\n");

  return res;
}
#ifndef NO_ARDRONE_MAINLOOP

#include <locale.h>

int main(int argc, char **argv)
{
  C_RESULT res;
  const char* old_locale;
  const char* appname = argv[0];
  int argc_backup = argc;
  char** argv_backup = argv;

  bool_t show_usage = FAILED( ardrone_tool_check_argc_custom(argc) ) ? TRUE : FALSE;

  argc--; argv++;
  while( argc && *argv[0] == '-' )
  {
    if( !strcmp(*argv, "-?") || !strcmp(*argv, "-h") || !strcmp(*argv, "-help") || !strcmp(*argv, "--help") )
    {
      ardrone_tool_usage( appname );
      exit( 0 );
    }
    else if( !ardrone_tool_parse_cmd_line_custom( *argv ) )
    {
      printf("Option %s not recognized\n", *argv);
      show_usage = TRUE;
    }

    argc--; argv++;
  }

  if( show_usage || (argc != 0) )
  {
    ardrone_tool_usage( appname );
    exit(-1);
  }
  
  /* After a first analysis, the arguments are restored so they can be passed to the user-defined functions */
  argc=argc_backup;
  argv=argv_backup;
  
  old_locale = setlocale(LC_NUMERIC, "en_GB.UTF-8");

  if( old_locale == NULL )
  {
    PRINT("You have to install new locales in your dev environment! (avoid the need of conv_coma_to_dot)\n");
    PRINT("As root, do a \"dpkg-reconfigure locales\" and add en_GB.UTF8 to your locale settings\n");
    PRINT("If you have any problem, feel free to contact Pierre Eline (pierre.eline@parrot.com)\n");
  }
  else
  {
    PRINT("Setting locale to %s\n", old_locale);
  }

  vp_com_wifi_config_t *config = (vp_com_wifi_config_t*)wifi_config();
  if(config)
  {
	  vp_os_memset( &wifi_ardrone_ip[0], 0, sizeof(wifi_ardrone_ip) );
	  printf("===================+> %s\n", config->server);
	  strcpy( &wifi_ardrone_ip[0], config->server);
  }

  if( &custom_main )
  {
    return custom_main(argc, argv);
  }
  else
  {
	res = ardrone_tool_setup_com( NULL );

    if( FAILED(res) )
    {
      PRINT("Wifi initialization failed. It means either:\n");
      PRINT("\t* you're not root (it's mandatory because you can set up wifi connection only as root)\n");
      PRINT("\t* wifi device is not present (on your pc or on your card)\n");
      PRINT("\t* you set the wrong name for wifi interface (for example rausb0 instead of wlan0) \n");
      PRINT("\t* ap is not up (reboot card or remove wifi usb dongle)\n");
      PRINT("\t* wifi device has no antenna\n");
    }
    else
    {
      res = ardrone_tool_init(argc, argv);

      while( SUCCEED(res) && ardrone_tool_exit() == FALSE )
      {
        res = ardrone_tool_update();
      }

      res = ardrone_tool_shutdown();
    }
  }

  if( old_locale != NULL )
  {
    setlocale(LC_NUMERIC, old_locale);
  }

  return SUCCEED(res) ? 0 : -1;
}
#endif // ! WITH_ARDRONE_MAIN_LOOP

// Default implementation for weak functions
#ifndef _WIN32
	C_RESULT ardrone_tool_init_custom(int argc, char **argv) { return C_OK; }
	C_RESULT ardrone_tool_update_custom() { return C_OK; }
	C_RESULT ardrone_tool_display_custom() { return C_OK; }
	C_RESULT ardrone_tool_shutdown_custom() { return C_OK; }
	bool_t   ardrone_tool_exit() { return FALSE; }
	C_RESULT ardrone_tool_check_argc_custom( int32_t argc) { return C_OK; }
	void ardrone_tool_display_cmd_line_custom( void ) {}
	bool_t ardrone_tool_parse_cmd_line_custom( const char* cmd ) { return TRUE; }
#endif


