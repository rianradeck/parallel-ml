.PHONY: run

run: code.exe
	./code.exe

code.exe: src/code.cpp
	g++ -fopenmp -o code ./src/code.cpp
