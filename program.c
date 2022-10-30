#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#define SOLUTION 0.06225419868854213

struct point
{
	double x;
	double y;
	double z;
};

double f(struct point* p)
{
	if (p->x * p->x + p->y * p->y + p->z * p->z <= 1.0) {
		return sin(p->x * p->x + p->z * p->z) * p->y;
	} else {
		return 0.0;
	}
}

double get_random(double min, double range)
{
	return min + (double)rand() * range / (double)RAND_MAX;
}

void generate_points(int count, struct point* buffer)
{
	for (int idx = 0; idx < count; ++idx) {
		buffer[idx].x = get_random(0.0, 1.0);
		buffer[idx].y = get_random(0.0, 1.0);
		buffer[idx].z = get_random(0.0, 1.0);
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

	struct point* points_buf = malloc(
		sizeof(struct point) * points_per_iter);

	generate_points(points_per_iter, points_buf);

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

		double value;
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
	} while (fabs((sum / (double)n_points) - SOLUTION) >= epsilon);

	free(points_buf);

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
}

void worker(int rank, int n_proc)
{
	int m = 0;

	struct point* buffer;

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
			buffer = malloc(sizeof(struct point) * points);
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

		#pragma omp parallel for reduction(+:sum)
		for (int idx = 0; idx < points; ++idx) {
			sum += f(buffer + idx);
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
