.PHONY: compile run

compile: program

program: program.c
	mpicc -Wall -fopenmp -o program program.c -lm

run: compile
	mpirun -np 4 ./program 1e-5
