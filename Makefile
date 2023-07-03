extension := 

ifeq ($(os), windows_nt)
	extension := .exe
endif

.PHONY: run

run: code$(extension)
	./code$(extension)

code$(extension): src/code.cpp
	g++ -std=gnu++17 -fopenmp -o code$(extension) ./src/code.cpp

# run: codeGPU$(extension)
# 	./codeGPU$(extension)

# codeGPU$(extension): src/code.cu
# 	nvcc ./src/code.cu -o codeGPU$(extension) 