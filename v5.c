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

#define ROOT 0
#define DATA_TAG 100

typedef struct body {
  float x, y, z, vx, vy, vz; 
} Body;

/************************  Service Functions  ************************/

void swap(void **addr1, void **addr2) {
  void *temp = *addr1;
  *addr1 = *addr2;
  *addr2 = temp;
}

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 20.0f * (rand() / (float)RAND_MAX) - 10.0f;
  }
}

/************************  Program  ************************/

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

  float *Fx = malloc(sizeof(float) * receiveCounts[rank]);
  float *Fy = malloc(sizeof(float) * receiveCounts[rank]);
  float *Fz = malloc(sizeof(float) * receiveCounts[rank]);

  for (int i = 0; i < receiveCounts[rank]; i++) {
    Fx[i] = 0.0f;
    Fy[i] = 0.0f;
    Fz[i] = 0.0f;
  }

  MPI_Request *receiveActiveRequests = malloc(sizeof(MPI_Request) * (commSize - 1));
  MPI_Request *receiveNextRequests = malloc(sizeof(MPI_Request) * (commSize - 1));
  MPI_Request *sendRequests = malloc(sizeof(MPI_Request) * (commSize - 1));

  /************************  Starting computation  ************************/
  printf("Starting computation...\n");

  MPI_Barrier(MPI_COMM_WORLD);
  timeStart = MPI_Wtime();

  int start = startIndex(rank, commSize, nBodies);
  int end = endIndex(rank, commSize, nBodies);

  for (int iter = 1; iter <= nIters; iter++) {
    int received = 0;

    while (received != commSize) {
      int process = received;
      
      if (iter != 1) {
        if (received != 0) {
          MPI_Waitany(commSize - 1, receiveActiveRequests, &process, MPI_STATUS_IGNORE);
          process = process + (process < rank ? 0 : 1);
        }
        else
          process = rank;
      }
      received++;

      int currentProcessStart = displacements[process];
      int currentProcessEnd =  currentProcessStart + receiveCounts[process];

      // Fxyz is an array of size workload(rank)
      for (int i = start, iData = 0; i < end; i++, iData++) {
        for (int j = currentProcessStart; j < currentProcessEnd; j++) {
          float dx = computationBodies[j].x - computationBodies[i].x;
          float dy = computationBodies[j].y - computationBodies[i].y;
          float dz = computationBodies[j].z - computationBodies[i].z;
          float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
          float invDist = 1.0f / sqrtf(distSqr);
          float invDist3 = invDist * invDist * invDist;

          Fx[iData] += (dx * invDist3);
          Fy[iData] += (dy * invDist3);
          Fz[iData] += (dz * invDist3);
        }
      }

      if (process != rank) {
        int j = process - (process > rank ? 1 : 0);
        MPI_Irecv(&receiveBodies[displacements[process]], receiveCounts[process], body, process, DATA_TAG, MPI_COMM_WORLD, &receiveNextRequests[j]);
      }
    }

    for (int i = start, iData = 0; i < end; i++, iData++) {
      computationBodies[i].vx += dt * Fx[iData];
      computationBodies[i].vy += dt * Fy[iData];
      computationBodies[i].vz += dt * Fz[iData];
      
      Fx[iData] = 0.0f;
      Fy[iData] = 0.0f;
      Fz[iData] = 0.0f;

      computationBodies[i].x += computationBodies[i].vx * dt;
      computationBodies[i].y += computationBodies[i].vy * dt;
      computationBodies[i].z += computationBodies[i].vz * dt;
    }

    // Check that the all the last send have completed
    if (iter != 1)
      MPI_Waitall(commSize - 1, sendRequests, MPI_STATUSES_IGNORE);

    for (int i = 0, j = 0; i < commSize; i++)
      if (i != rank)
        MPI_Irsend(&computationBodies[displacements[rank]], receiveCounts[rank], body, i, DATA_TAG, MPI_COMM_WORLD, &sendRequests[j++]);

    memcpy(&receiveBodies[displacements[rank]], &computationBodies[displacements[rank]], sizeof(Body) * receiveCounts[rank]);
    
    swap((void **) &receiveActiveRequests, (void **) &receiveNextRequests);
    swap((void **) &computationBodies, (void **) &receiveBodies);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  timeEnd = MPI_Wtime();
  MPI_Finalize();

  #ifdef DEBUG
  if (rank == ROOT) {
    for(int i = 0; i < nBodies; i++)
      printf("%.3f ", computationBodies[i].x);
    printf("\n");
  }
  #endif

  if (rank == 0)
    printf("%f\n", timeEnd - timeStart);

  free(receiveCounts);
  free(displacements);
  free(Fx);
  free(Fy);
  free(Fz);
  free(receiveActiveRequests);
  free(receiveNextRequests);
  free(sendRequests);
  free(computationBodies);
  free(receiveBodies);
}