#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG

#define SEED 100
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) { 
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

  int nBodies = 1000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 100;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *bodies = (Body*)buf;

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  printf("Starting computation...\n");
  int start = time(NULL);

  for (int iter = 1; iter <= nIters; iter++) {

    bodyForce(bodies, dt, nBodies); // compute interbody forces

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      bodies[i].x += bodies[i].vx*dt;
      bodies[i].y += bodies[i].vy*dt;
      bodies[i].z += bodies[i].vz*dt;
    }
  }

  int end = time(NULL);

  #ifdef DEBUG
  for(int i = 0; i < nBodies; i++)
    printf("%.3f ", bodies[i].x);
  printf("\n");
  #endif

  printf("Total time: %d\n", end - start);
  free(buf);
}