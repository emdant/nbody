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

/************************  Data Structures  ************************/

typedef struct block {
  int process;
  int firstIteration;
} Block;


typedef struct blockqueue {
  Block *blocks;
  int size;
  int head;
  int tail;
} BlockQueue;

void initializeQueue(BlockQueue *q, int size) {
  q->blocks = malloc(sizeof(Block) * (size + 1));
  q->size = size + 1;
  q->head = q->tail = 0;
}

void enqueue(BlockQueue *q, Block block) {
  if ((q->tail + 1) % q->size == q->head)
    return;

  q->blocks[q->tail] = block;
  q->tail = (q->tail + 1) % q->size;
}

Block dequeue(BlockQueue *q) {
  Block invalid = {-1, 0};
  if (q->tail == q->head)
    return invalid;

  Block temp = q->blocks[q->head];
  q->head = (q->head + 1) % q->size;
  return temp;
}

int isEmpty(BlockQueue *q) {
  return q->tail == q->head;
}

void freeQueue(BlockQueue *q) {
  free(q->blocks);
}

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

  BlockQueue *activeQueue = malloc(sizeof(BlockQueue)), *nextQueue = malloc(sizeof(BlockQueue));
  initializeQueue(activeQueue, commSize);
  initializeQueue(nextQueue, commSize);

  float *Fx = malloc(sizeof(float) * receiveCounts[rank]);
  float *Fy = malloc(sizeof(float) * receiveCounts[rank]);
  float *Fz = malloc(sizeof(float) * receiveCounts[rank]);

  for (int i = 0; i < receiveCounts[rank]; i++) {
    Fx[i] = 0.0f;
    Fy[i] = 0.0f;
    Fz[i] = 0.0f;
  }

  /************************  Starting computation  ************************/
  printf("Starting computation...\n");

  MPI_Barrier(MPI_COMM_WORLD);
  timeStart = MPI_Wtime();

  MPI_Request *receiveRequests = malloc(sizeof(MPI_Request) * (commSize));
  MPI_Request *sendRequests = malloc(sizeof(MPI_Request) * (commSize));

  int start = startIndex(rank, commSize, nBodies);
  int end = endIndex(rank, commSize, nBodies);

  for (int i = 0; i < commSize; i++) {
    Block block = {.process = i, .firstIteration = 1};
    MPI_Irecv(&receiveBodies[displacements[i]], receiveCounts[i], body, i, DATA_TAG, MPI_COMM_WORLD, &receiveRequests[i]);
    enqueue(activeQueue, block);
  }
  
  for (int iter = 1; iter <= nIters; iter++) {

    while (!isEmpty(activeQueue)) {
      Block currentBlock = dequeue(activeQueue);
      int completed = 0;

      MPI_Test(&receiveRequests[currentBlock.process], &completed, MPI_STATUS_IGNORE);

      if (!completed && !currentBlock.firstIteration) {
        enqueue(activeQueue, currentBlock);
      } else {
        int currentProcessStart = startIndex(currentBlock.process, commSize, nBodies);
        int currentProcessEnd = endIndex(currentBlock.process, commSize, nBodies);

        // Fxyz is an array of size workload(rank)
        for (int i = start, i_data = 0; i < end; i++, i_data++) {
          
          for (int j = currentProcessStart; j < currentProcessEnd; j++) {
            float dx = computationBodies[j].x - computationBodies[i].x;
            float dy = computationBodies[j].y - computationBodies[i].y;
            float dz = computationBodies[j].z - computationBodies[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx[i_data] += (dx * invDist3);
            Fy[i_data] += (dy * invDist3);
            Fz[i_data] += (dz * invDist3);
          }
        }
        
        if (currentBlock.firstIteration) 
          currentBlock.firstIteration = 0;
        else if (iter != nIters)
          MPI_Irecv(&receiveBodies[displacements[currentBlock.process]], receiveCounts[currentBlock.process], body, currentBlock.process, DATA_TAG, MPI_COMM_WORLD, &receiveRequests[currentBlock.process]);

        enqueue(nextQueue, currentBlock);
      }
    }

    for (int i = start, i_data = 0; i < end; i++, i_data++) {
      computationBodies[i].vx += dt * Fx[i_data];
      Fx[i_data] = 0.0f;
      computationBodies[i].vy += dt * Fy[i_data];
      Fy[i_data] = 0.0f;
      computationBodies[i].vz += dt * Fz[i_data];
      Fz[i_data] = 0.0f;
    }

    for (int i = start; i < end; i++) {
      computationBodies[i].x += computationBodies[i].vx * dt;
      computationBodies[i].y += computationBodies[i].vy * dt;
      computationBodies[i].z += computationBodies[i].vz * dt;
    }

    // Check that the all the last send have completed
    if (iter != 1)
      MPI_Waitall(commSize, sendRequests, MPI_STATUSES_IGNORE);

    for (int i = 0; i < commSize; i++)
      MPI_Irsend(&computationBodies[displacements[rank]], receiveCounts[rank], body, i, DATA_TAG, MPI_COMM_WORLD, &sendRequests[i]);

    memcpy(&receiveBodies[displacements[rank]], &computationBodies[displacements[rank]], sizeof(Body) * receiveCounts[rank]);
    
    swap((void **) &activeQueue, (void **) &nextQueue);
    swap((void **) &computationBodies, (void **) &receiveBodies);
  }

  MPI_Waitall(commSize, receiveRequests, MPI_STATUSES_IGNORE);

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
    printf("Time in seconds = %f\n", timeEnd - timeStart);

  free(receiveCounts);
  free(displacements);
  freeQueue(activeQueue);
  freeQueue(nextQueue);
  free(activeQueue);
  free(nextQueue);
  free(Fx);
  free(Fy);
  free(Fz);
  free(receiveRequests);
  free(sendRequests);
  free(computationBodies);
  free(receiveBodies);
}