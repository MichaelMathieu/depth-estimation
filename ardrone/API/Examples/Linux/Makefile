
.PHONY:

all:
	@$(MAKE) -C ../../ARDroneLib/Soft/Build USE_LINUX=yes
	@$(MAKE) -C sdk_demo/Build USE_LINUX=yes 

$(MAKECMDGOALS):
	@$(MAKE) -C ../../ARDroneLib/Soft/Build USE_LINUX=yes $(MAKECMDGOALS)
	@$(MAKE) -C sdk_demo/Build USE_LINUX=yes $(MAKECMDGOALS)


