CXX=g++
FLAGS=-W -fpic -O3 -funroll-loops

all: oextract

oextract:
	${CXX} ${FLAGS} extract_output.cpp -shared -o extractoutput.so

clean:
	rm -rf extractoutput.so