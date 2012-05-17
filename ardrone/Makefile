LINK=`pkg-config --libs opencv` -lglut -lGLU -lGL -lluaT -ltorch-lua -lTH
FLAGS=-g -W -Wall -Wextra
CXX=g++
CPP=simulator.cpp common.cpp ardrone_api.cpp radial_depth_map.cpp depth_map.cpp
OBJ=$(CPP:.cpp=.o)

all:	simulator ardroneSDK

.cpp.o:
	${CXX} ${FLAGS} -c $< -o $@

simulator: ${CPP} ${OBJ} test_simulator.cpp test_simulator.o
	${CXX} ${FLAGS} ${OBJ} test_simulator.cpp ${LINK} -o simulator

control: ${CPP} ${OBJ} simple_control_svg.cpp simple_control_svg.o
	${CXX} ${FLAGS} ${OBJ} simple_control_svg.cpp ${LINK} -o simple_control

ardroneSDK:
	cd API/Examples/Linux/ && make && cd - && cp API/Examples/Linux/Build/Release/linux_sdk_demo ardrone_sdk
	rm -f control_pipe && mkfifo control_pipe
	rm -f navdata_pipe && mkfifo navdata_pipe

clean:
	rm -rf simulator test_simulator.o ${OBJ}