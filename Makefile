.PHONY: compile run

PREC=1.5e-6
POINTS=1000

compile: program

program: program.c
	mpicc -Wall -fopenmp -o program program.c -lm

run: compile
	mpirun --use-hwthread-cpus -np 2 ./program $(PREC) $(POINTS)
	mpirun --use-hwthread-cpus -np 3 ./program $(PREC) $(POINTS)
	mpirun --use-hwthread-cpus -np 4 ./program $(PREC) $(POINTS)
	mpirun --use-hwthread-cpus -np 5 ./program $(PREC) $(POINTS)
	mpirun --use-hwthread-cpus -np 6 ./program $(PREC) $(POINTS)
	mpirun --use-hwthread-cpus -np 7 ./program $(PREC) $(POINTS)
	mpirun --use-hwthread-cpus -np 8 ./program $(PREC) $(POINTS)
