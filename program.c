#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#define POINTS 100
#define SOLUTION 100e8000

struct point
{
	float x;
	float y;
	float z;
};

float f(struct point* p)
{
	if (p->x * p->x + p->y * p->y + p->z * p->z <= 1.0f) {
		return sin(p->x * p->x + p->z * p->z) * p->y;
	} else {
		return 0.0f;
	}
}

float get_random(float min, float range)
{
	return min + (float)rand() * range / (float)RAND_MAX;
}

struct point* generate_points(int count)
{
	struct point* buffer = malloc(sizeof(struct point) * count);

	for (int idx = 0; idx < count; ++idx) {
		buffer[idx].x = get_random(0.0f, 1.0f);
		buffer[idx].y = get_random(0.0f, 1.0f);
		buffer[idx].z = get_random(0.0f, 1.0f);
	}

	return buffer;
}

void master(int rank, int n_proc, float epsilon)
{
	float sum = 0;
	int n_points = 0;

	int points_per_iter = (n_proc - 1) * POINTS;

	int* init_scatter_count = malloc(sizeof(int) * n_proc - 1);
	int* init_scatter_offset = malloc(sizeof(int) * n_proc);

	int* scatter_count = malloc(sizeof(int) * n_proc - 1);
	int* scatter_offset = malloc(sizeof(int) * n_proc);

	init_scatter_count[0] = 0;
	init_scatter_offset[0] = 0;

	scatter_count[0] = 0;
	scatter_offset[0] = 0;

	for (int idx = 1; idx < n_proc; ++idx) {
		init_scatter_count[idx] = 1;
		init_scatter_offset[idx] = 0;

		scatter_count[idx] = POINTS * 3;
		scatter_offset[idx] = (idx - 1) * POINTS * 3;
	}

	for (int iter = 0; iter < 100; ++iter) {
		struct point* points_buf = generate_points(points_per_iter);
		int points = POINTS;
		n_points += points_per_iter;

		MPI_Scatterv(
			&points,
			init_scatter_count,
			init_scatter_offset,
			MPI_INT,
			NULL,
			0,
			MPI_INT,
			0,
			MPI_COMM_WORLD);

		MPI_Scatterv(
			points_buf,
			scatter_count,
			scatter_offset,
			MPI_FLOAT,
			NULL,
			0,
			MPI_FLOAT,
			0,
			MPI_COMM_WORLD);

		free(points_buf);

		float value;
		float zero = 0;

		MPI_Reduce(
			&zero,
			&value,
			1,
			MPI_FLOAT,
			MPI_SUM,
			0,
			MPI_COMM_WORLD);

		sum += value;
	}

	int zero = 0;

	MPI_Scatterv(
		&zero,
		init_scatter_count,
		init_scatter_offset,
		MPI_INT,
		NULL,
		0,
		MPI_INT,
		0,
		MPI_COMM_WORLD);

	float result = sum / (float)n_points;
	printf("%f\n", result);
}

void worker(int rank, int n_proc)
{
	while (1)
	{
		int points;

		MPI_Scatterv(
			NULL,
			NULL,
			NULL,
			MPI_INT,
			&points,
			1,
			MPI_INT,
			0,
			MPI_COMM_WORLD);

		if (points == 0) {
			break;
		}

		struct point* buffer = malloc(sizeof(struct point) * points);
		MPI_Scatterv(
			NULL,
			NULL,
			NULL,
			MPI_FLOAT,
			buffer,
			points * 3,
			MPI_FLOAT,
			0,
			MPI_COMM_WORLD);

		float sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int idx = 0; idx < points; ++idx) {
			sum += f(buffer + idx);
		}

		free(buffer);

		MPI_Reduce(
			&sum,
			NULL,
			1,
			MPI_FLOAT,
			MPI_SUM,
			0,
			MPI_COMM_WORLD);
	}
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	double start = MPI_Wtime();

	int rank;
	int n_proc;

	float epsilon;

	sscanf(argv[1], "%f", &epsilon);

	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		master(rank, n_proc, epsilon);
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
		printf("%lf\n", total_time);
	}

	MPI_Finalize();

	return 0;
}
