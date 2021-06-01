#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEED 100
#define SOFTENING 1e-9f

#define workload(rank, nProc, nBodies) ((nBodies / nProc) + (rank < (nBodies % nProc) ? 1 : 0))
#define startIndex(rank, nProc, nBodies) (rank < (nBodies % nProc) ? rank * (nBodies / nProc + 1) : (nBodies % nProc) *  (nBodies / nProc + 1) + (rank - (nBodies % nProc)) * (nBodies / nProc))
#define endIndex(rank, nProc, nBodies) (rank < (nBodies % nProc) ? startIndex(rank, nProc, nBodies) + (nBodies / nProc + 1) : startIndex(rank, nProc, nBodies) + (nBodies / nProc))

#define DATA_TAG 100

typedef struct body {
  float x, y, z, vx, vy, vz; 
} Body;

void swap(void **addr1, void **addr2) {
  void *temp = *addr1;
  *addr1 = *addr2;
  *addr2 = temp;
}

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] =  20.0f * (rand() / (float)RAND_MAX) - 10.0f;
  }
}

int main(int argc, char** argv) {
  srand(SEED);

  MPI_Init(&argc, &argv);

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <n_iterations> <n_bodies>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const int nIters = atoi(argv[1]); // simulation iterations
  const int nBodies = atoi(argv[2]); // number of bodies

  int commSize, rank;
  double timeStart, timeEnd; // Time benchmarking

  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Datatype body;
  MPI_Type_contiguous(6, MPI_FLOAT, &body);
  MPI_Type_commit(&body);

  const float dt = 0.01f; // time step

  int bytes = nBodies * sizeof(Body);
  Body *computationBodies = malloc(bytes);
  Body *receiveBodies = malloc(bytes);

  randomizeBodies((float *) computationBodies, 6 * nBodies); // Init pos / vel data

  int *receiveCounts = malloc(sizeof(int) * commSize), *displacements = malloc(sizeof(int) * commSize);
  for (int i = 0; i < commSize; i++) {
    receiveCounts[i] = workload(i, commSize, nBodies);
    displacements[i] = startIndex(i, commSize, nBodies);
  }

  printf("Starting computation...\n");

  MPI_Barrier(MPI_COMM_WORLD);
  timeStart = MPI_Wtime();

  int start = startIndex(rank, commSize, nBodies);
  int end = endIndex(rank, commSize, nBodies);

  MPI_Request *receiveRequests = malloc(sizeof(MPI_Request) * (commSize - 1));
  MPI_Request *sendRequests = malloc(sizeof(MPI_Request) * (commSize - 1));
  for (int iter = 1; iter <= nIters; iter++) {

    for (int i = 0, j = 0; i < commSize; i++) 
      if (i != rank)
        MPI_Irecv(&receiveBodies[displacements[i]], receiveCounts[i], body, i, DATA_TAG, MPI_COMM_WORLD, &receiveRequests[j++]);

    for (int i = start; i < end; i++) { 
      float Fx = 0.0f;
      float Fy = 0.0f;
      float Fz = 0.0f;

      for (int j = 0; j < nBodies; j++) {
        float dx = computationBodies[j].x - computationBodies[i].x;
        float dy = computationBodies[j].y - computationBodies[i].y;
        float dz = computationBodies[j].z - computationBodies[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }

      computationBodies[i].vx += dt * Fx;
      computationBodies[i].vy += dt * Fy;
      computationBodies[i].vz += dt * Fz;
    }

    for (int i = start; i < end; i++) { // integrate position
      computationBodies[i].x += computationBodies[i].vx * dt;
      computationBodies[i].y += computationBodies[i].vy * dt;
      computationBodies[i].z += computationBodies[i].vz * dt;
    }

    if (iter != 1)
      MPI_Waitall(commSize - 1, sendRequests, MPI_STATUSES_IGNORE);

    for (int i = 0, j = 0; i < commSize; i++)
      if (i != rank) 
        MPI_Irsend(&computationBodies[displacements[rank]], receiveCounts[rank], body, i, DATA_TAG, MPI_COMM_WORLD, &sendRequests[j++]);    
    
    MPI_Waitall(commSize - 1, receiveRequests, MPI_STATUSES_IGNORE);

    memcpy(&receiveBodies[displacements[rank]], &computationBodies[displacements[rank]], sizeof(Body) * receiveCounts[rank]);

    swap((void **) &computationBodies, (void **) &receiveBodies);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  timeEnd = MPI_Wtime();
  MPI_Finalize();

  #ifdef DEBUG
  if (rank == 0) {
    for(int i = 0; i < nBodies; i++)
      printf("%.3f ", computationBodies[i].x);
    printf("\n");
  }
  #endif

  if (rank == 0)
    printf("%f\n", timeEnd - timeStart);

  free(computationBodies);
  free(receiveBodies);
  free(receiveCounts);
  free(displacements);
  free(receiveRequests);
  free(sendRequests);
}