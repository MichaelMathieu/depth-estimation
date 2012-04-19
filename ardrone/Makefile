LINK=`pkg-config --libs opencv` -lglut -lGLU -lGL
FLAGS=-g -W -Wall -Wextra
CXX=g++
CPP=simulator.cpp common.cpp
OBJ=$(CPP:.cpp=.o)

.cpp.o:
	${CXX} -c $< -o $@

simulator: ${CPP} ${OBJ} test_simulator.cpp test_simulator.o
	${CXX} ${FLAGS} ${OBJ} test_simulator.cpp ${LINK} -o simulator

clean:
	rm -rf simulator test_simulator.o ${OBJ}