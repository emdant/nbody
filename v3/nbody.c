#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SEED 100
#define N_BODIES 30000
#define N_ITER 10
#define SOFTENING 1e-9f

#define workload(rank, nProc, nBodies) ((nBodies / nProc) + (rank < (nBodies % nProc) ? 1 : 0))
#define startIndex(rank, nProc, nBodies) (rank < (nBodies % nProc) ? rank * (nBodies / nProc + 1) : (nBodies % nProc) *  (nBodies / nProc + 1) + (rank - (nBodies % nProc)) * (nBodies / nProc))
#define endIndex(rank, nProc, nBodies) (rank < (nBodies % nProc) ? startIndex(rank, nProc, nBodies) + (nBodies / nProc + 1) : startIndex(rank, nProc, nBodies) + (nBodies / nProc))

typedef struct { float x[N_BODIES], y[N_BODIES], z[N_BODIES], vx[N_BODIES], vy[N_BODIES], vz[N_BODIES]; } Bodies;

int size, rank;

void randomizeBodies(Bodies *bodies) {
  for (int i = 0; i < N_BODIES; i++) {
    bodies->x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
  for (int i = 0; i < N_BODIES; i++) {
    bodies->y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
  for (int i = 0; i < N_BODIES; i++) {
    bodies->z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
  for (int i = 0; i < N_BODIES; i++) {
    bodies->vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
  for (int i = 0; i < N_BODIES; i++) {
    bodies->vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
  for (int i = 0; i < N_BODIES; i++) {
    bodies->vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Bodies *b, float dt, int n) {
  int start = startIndex(rank, size, N_BODIES);
  int end = endIndex(rank, size, N_BODIES);

  for (int i = start; i < end; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = b->x[j] - b->x[i];
      float dy = b->y[j] - b->y[i];
      float dz = b->z[j] - b->z[j];
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    b->vx[i] += dt*Fx; b->vy[i] += dt*Fy; b->vy[i] += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  srand(SEED);

  MPI_Init(NULL, NULL);

  double start, end; // Time benchmarking

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int nIters = N_ITER;  // simulation iterations
  const float dt = 0.01f; // time step

  Bodies bodies;
  randomizeBodies(&bodies); // Init pos / vel data

  int *receiveCounts = malloc(sizeof(int) * size), *displacements = malloc(sizeof(int) * size);
  for (int i = 0; i < size; i++) {
    receiveCounts[i] = workload(i, size, N_BODIES);
    displacements[i] = startIndex(i, size, N_BODIES);
  }

  printf("Starting computation...\n");

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  for (int iter = 1; iter <= nIters; iter++) {

    bodyForce(&bodies, dt, N_BODIES); // compute interbody forces

    int start = startIndex(rank, size, N_BODIES);
    int end = endIndex(rank, size, N_BODIES);
    for (int i = start; i < end; i++) { // integrate position
      bodies.x[i] += bodies.vx[i] * dt;
      bodies.y[i] += bodies.vy[i] * dt;
      bodies.z[i] += bodies.vz[i] * dt;
    }

    MPI_Allgatherv(bodies.x + displacements[rank], receiveCounts[rank], MPI_FLOAT, bodies.x, receiveCounts, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(bodies.y + displacements[rank], receiveCounts[rank], MPI_FLOAT, bodies.y, receiveCounts, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(bodies.z + displacements[rank], receiveCounts[rank], MPI_FLOAT, bodies.z, receiveCounts, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(bodies.vx + displacements[rank], receiveCounts[rank], MPI_FLOAT, bodies.vx, receiveCounts, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(bodies.vy + displacements[rank], receiveCounts[rank], MPI_FLOAT, bodies.vy, receiveCounts, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(bodies.vz + displacements[rank], receiveCounts[rank], MPI_FLOAT, bodies.vz, receiveCounts, displacements, MPI_FLOAT, MPI_COMM_WORLD);

  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  MPI_Finalize();

  #ifdef DEBUG
  if (rank == 0)
    for(int i = 0; i < N_BODIES; i++)
      printf("%.3f ", bodies[i].x);
    printf("\n");
  #endif

  if (rank == 0)
    printf("Time in seconds = %f\n", end - start);

}