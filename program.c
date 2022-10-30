#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#define SOLUTION 0.06225419868854213

double f(double x, double y, double z)
{
	if (x * x + y * y + z * z <= 1.0) {
		return sin(x * x + z * z) * y;
	} else {
		return 0.0;
	}
}

double get_random(double min, double range)
{
	return min + (double)rand() * range / (double)RAND_MAX;
}

void generate_points(int count, double* buffer)
{
	#pragma omp parallel for
	for (int idx = 0; idx < count; ++idx) {
		buffer[idx * 3] = get_random(0.0, 1.0);
		buffer[idx * 3 + 1] = get_random(0.0, 1.0);
		buffer[idx * 3 + 2] = get_random(0.0, 1.0);
	}
}

void master(int rank, int n_proc, double epsilon, int points)
{
	double sum = 0;
	int n_points = 0;

	int points_per_iter_local = points / (n_proc - 1);
	int points_per_iter = points_per_iter_local * (n_proc - 1);

	int* scatter_count = malloc(sizeof(int) * n_proc);
	int* scatter_offset = malloc(sizeof(int) * n_proc);

	scatter_count[0] = 0;
	scatter_offset[0] = 0;

	for (int idx = 1; idx < n_proc; ++idx) {
		scatter_count[idx] = points_per_iter_local * 3;
		scatter_offset[idx] = (idx - 1) * points_per_iter_local * 3;
	}

	double* points_buf = malloc(
		sizeof(double) * points_per_iter * 3);

	generate_points(points_per_iter, points_buf);

	int iterations = 0;

	do {
		n_points += points_per_iter;

		MPI_Bcast(
			&points_per_iter_local,
			1,
			MPI_INT,
			0,
			MPI_COMM_WORLD);

		MPI_Scatterv(
			points_buf,
			scatter_count,
			scatter_offset,
			MPI_DOUBLE,
			NULL,
			0,
			MPI_DOUBLE,
			0,
			MPI_COMM_WORLD);

		generate_points(points_per_iter, points_buf);

		double value = 0;
		double zero = 0;

		MPI_Reduce(
			&zero,
			&value,
			1,
			MPI_DOUBLE,
			MPI_SUM,
			0,
			MPI_COMM_WORLD);

		sum += value;
		++iterations;
	} while (fabs((sum / (double)n_points) - SOLUTION) >= epsilon);

	free(points_buf);
	free(scatter_offset);
	free(scatter_count);

	int zero = 0;

	MPI_Bcast(
		&zero,
		1,
		MPI_INT,
		0,
		MPI_COMM_WORLD);

	double result = sum / (double)n_points;

	printf("Result = %.17lf\n", result);
	printf("Error = %.17lf\n", fabs(result - SOLUTION));
	printf("Iterations = %d\n", iterations);
}

void worker(int rank, int n_proc)
{
	int m = 0;

	double* buffer;

	while (1)
	{
		int points;

		MPI_Bcast(
			&points,
			1,
			MPI_INT,
			0,
			MPI_COMM_WORLD);

		if (points == 0) {
			break;
		}

		if (!m) {
			buffer = malloc(sizeof(double) * points * 3);
			m = 1;
		}

		MPI_Scatterv(
			NULL,
			NULL,
			NULL,
			MPI_DOUBLE,
			buffer,
			points * 3,
			MPI_DOUBLE,
			0,
			MPI_COMM_WORLD);

		double sum = 0;

		for (int idx = 0; idx < points; ++idx) {
			sum += f(
				buffer[idx * 3],
				buffer[idx * 3 + 1],
				buffer[idx * 3 + 2]);
		}

		MPI_Reduce(
			&sum,
			NULL,
			1,
			MPI_DOUBLE,
			MPI_SUM,
			0,
			MPI_COMM_WORLD);
	}

	if (m) {
		free(buffer);
	}
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	double start = MPI_Wtime();

	int rank;
	int n_proc;

	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		double epsilon = atof(argv[1]);
		printf("e = %.17lf\n", epsilon);

		int points = atoi(argv[2]);
		printf("Points = %d\n", points);

		master(rank, n_proc, epsilon, points);
	} else {
		worker(rank, n_proc);
	}

	double local_time = MPI_Wtime() - start;
	double total_time;

	MPI_Reduce(
		&local_time,
		&total_time,
		1,
		MPI_DOUBLE,
		MPI_MAX,
		0,
		MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Time = %.17lf\n", total_time);
	}

	MPI_Finalize();

	return 0;
}
