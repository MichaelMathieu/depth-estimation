LINK=`pkg-config --libs opencv` -lglut -lGLU -lGL
FLAGS=-g -W -Wall -Wextra
CXX=g++
CPP=simulator.cpp common.cpp ardrone_api.cpp depth_map.cpp
OBJ=$(CPP:.cpp=.o)

all:	simulator dmap control

.cpp.o:
	${CXX} -c $< -o $@

simulator: ${CPP} ${OBJ} test_simulator.cpp test_simulator.o
	${CXX} ${FLAGS} ${OBJ} test_simulator.cpp ${LINK} -o simulator

control: ${CPP} ${OBJ} simple_control_svg.cpp simple_control_svg.o
	${CXX} ${FLAGS} ${OBJ} simple_control_svg.cpp ${LINK} -o simple_control

clean:
	rm -rf simulator test_simulator.o ${OBJ}