extension := 

ifeq ($(os), windows_nt)
	extension := .exe
endif

.PHONY: run

run: code$(extension)
	./code$(extension)

code$(extension): src/code.cpp
	g++ -std=gnu++17 -fopenmp -o code$(extension) ./src/code.cpp
