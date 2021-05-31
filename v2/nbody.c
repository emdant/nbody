#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG

#ifdef DEBUG
#define N_BODIES 1000
#else
#define N_BODIES 30000
#endif

#define SEED 100
#define N_ITER 100
#define SOFTENING 1e-9f

#define workload(rank, nProc, nBodies) ((nBodies / nProc) + (rank < (nBodies % nProc) ? 1 : 0))
#define startIndex(rank, nProc, nBodies) (rank < (nBodies % nProc) ? rank * (nBodies / nProc + 1) : (nBodies % nProc) *  (nBodies / nProc + 1) + (rank - (nBodies % nProc)) * (nBodies / nProc))
#define endIndex(rank, nProc, nBodies) (rank < (nBodies % nProc) ? startIndex(rank, nProc, nBodies) + (nBodies / nProc + 1) : startIndex(rank, nProc, nBodies) + (nBodies / nProc))

typedef struct { float x, y, z, vx, vy, vz; } Body;

int size, rank;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  int start = startIndex(rank, size, N_BODIES);
  int end = endIndex(rank, size, N_BODIES);

  for (int i = start; i < end; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  srand(SEED);

  MPI_Init(NULL, NULL);

  double start, end; // Time benchmarking

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Datatype body;
  MPI_Type_contiguous(6, MPI_FLOAT, &body);
  MPI_Type_commit(&body);

  const int nIters = N_ITER;  // simulation iterations
  const float dt = 0.01f; // time step

  int bytes = N_BODIES * sizeof(Body);
  float *buf = (float*) malloc(bytes);
  Body *bodies = (Body*) buf;

  randomizeBodies(buf, 6 * N_BODIES); // Init pos / vel data

  int *receiveCounts = malloc(sizeof(int) * size), *displacements = malloc(sizeof(int) * size);
  for (int i = 0; i < size; i++) {
    receiveCounts[i] = workload(i, size, N_BODIES);
    displacements[i] = startIndex(i, size, N_BODIES);
  }

  printf("Starting computation...\n");

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  for (int iter = 1; iter <= nIters; iter++) {

    bodyForce(bodies, dt, N_BODIES); // compute interbody forces

    int start = startIndex(rank, size, N_BODIES);
    int end = endIndex(rank, size, N_BODIES);
    for (int i = start; i < end; i++) { // integrate position
      bodies[i].x += bodies[i].vx * dt;
      bodies[i].y += bodies[i].vy * dt;
      bodies[i].z += bodies[i].vz * dt;
    }

    MPI_Allgatherv(bodies + displacements[rank], receiveCounts[rank], body, bodies, receiveCounts, displacements, body, MPI_COMM_WORLD);
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

  free(buf);
}