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

	for (int iter = 0; iter < 100; ++iter) {
		struct point* points_buf = generate_points(points_per_iter);
		int points = POINTS;
		n_points += points_per_iter;

		for (int p_rank = 1; p_rank < n_proc; ++p_rank) {
			MPI_Send(
				&points,
				1,
				MPI_INT,
				p_rank,
				1,
				MPI_COMM_WORLD);
		}

		for (int p_rank = 1; p_rank < n_proc; ++p_rank) {
			MPI_Send(
				points_buf + points * (p_rank - 1),
				points * 3,
				MPI_FLOAT,
				p_rank,
				2,
				MPI_COMM_WORLD);
		}

		free(points_buf);

		MPI_Status status;
		float value;
		for (int p_rank = 1; p_rank < n_proc; ++p_rank) {
			MPI_Recv(
				&value,
				1,
				MPI_FLOAT,
				MPI_ANY_SOURCE,
				3,
				MPI_COMM_WORLD,
				&status);

			sum += value;
		}
	}

	int zero = 0;
	for (int p_rank = 1; p_rank < n_proc; ++p_rank) {
		MPI_Send(
			&zero,
			1,
			MPI_INT,
			p_rank,
			1,
			MPI_COMM_WORLD);
	}

	float result = sum / (float)n_points;
	printf("%f\n", result);
}

void worker(int rank, int n_proc)
{
	while (1)
	{
		int points;
		MPI_Status status;
		MPI_Recv(&points, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

		if (points == 0) {
			break;
		}

		struct point* buffer = malloc(sizeof(struct point) * points);
		MPI_Recv(
			buffer,
			points * 3,
			MPI_FLOAT,
			0,
			2,
			MPI_COMM_WORLD,
			&status);

		float sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int idx = 0; idx < points; ++idx) {
			sum += f(buffer + idx);
		}

		free(buffer);

		MPI_Send(&sum, 1, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
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
