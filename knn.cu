#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <time.h>
#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
int d = 1 << 5;


__global__ void generatekey(float *Cx, float *Cy, float *Cz, int lenghtC,
                            int *keys, int d) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lenghtC;
       i = i + gridDim.x * blockDim.x) {
    int digit1 = Cx[i] * d;
    int digit2 = Cy[i] * d;
    int digit3 = Cz[i] * d;
    keys[i] = d * d * digit1 + d * digit2 + digit3;
  }
}
__global__ void findCellStartends(int *keys, int len, int *starts, int *ends) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i = i + gridDim.x * blockDim.x) {

    if (i > 0) {
      if (keys[i] != keys[i - 1]) {
        starts[keys[i]] = i;
      }
    } else {
      starts[keys[0]] = 0;
    }

    if (i != len - 1) {
      if (keys[i] != keys[i + 1]) {
        ends[keys[i]] = i;
      }
    } else {
      ends[keys[len - 1]] = len - 1;
    }
  }
}

__device__ float findBoarderdistance(float pointx, float pointy, float pointz,
                                     int d, int s, int digit1, int digit2,
                                     int digit3) {
  float bdist = 100;

  if (digit1 + s < d - 1) {
    bdist = -pointx + (float)(digit1 + s + 1) / d;
  }
  if (digit2 + s < d - 1) {
    bdist = fminf(-pointy + (float)(digit2 + s + 1) / d, bdist);
  }
  if (digit3 + s < d - 1) {
    bdist = fminf(-pointz + (float)(digit3 + s + 1) / d, bdist);
  }
  if (digit1 - s > 0) {
    bdist = fminf(+pointx + (float)(-digit1 + s) / d, bdist);
  }
  if (digit2 - s > 0) {
    bdist = fminf(+pointy + (float)(-digit2 + s) / d, bdist);
  }
  if (digit3 - s > 0) {
    bdist = fminf(+pointz + (float)(-digit3 + s) / d, bdist);
  }
  return bdist;
}

__device__ void searchCell(float pointx, float pointy, float pointz, int start,
                           int end, float *Cx, float *Cy, float *Cz,
                           float *minq, float *x, float *y, float *z) {
  float dist;
  if (start >= 0) {
    for (int j = start; j <= end; j++) {
      dist = (Cx[j] - pointx) * (Cx[j] - pointx) +
             (Cy[j] - pointy) * (Cy[j] - pointy) +
             (Cz[j] - pointz) * (Cz[j] - pointz);
      dist = sqrt(dist);
      if (dist < *minq) {
        *minq = dist;
        *x = Cx[j];
        *y = Cy[j];
        *z = Cz[j];
      }
    }
  }
}

__global__ void searchGrid(float *Qx, float *Qy, float *Qz, int *starts,
                           int *ends, float *Cx, float *Cy, float *Cz,
                           float *Resx, float *Resy, float *Resz, int d,
                           int lenghtQ) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lenghtQ;
       i = i + gridDim.x * blockDim.x) {

    float pointx = Qx[i];
    float pointy = Qy[i];
    float pointz = Qz[i];
    int digit1 = pointx * d;
    int digit2 = pointy * d;
    int digit3 = pointz * d;
    int key = d * d * digit1 + d * digit2 + digit3;
    float x = 0, y = 0, z = 0;
    float minq = 100;
    int s = 0;

    float bdist = findBoarderdistance(pointx, pointy, pointz, d, s, digit1,
                                      digit2, digit3);
    searchCell(pointx, pointy, pointz, starts[key], ends[key], Cx, Cy, Cz,
               &minq, &x, &y, &z);

    int digx, digy, digz;

    while (1) {
      s++;
      if (minq < bdist) {
        break;
      }
      for (digx = digit1 - s; digx <= digit1 + s; digx++) {

        for (digy = digit2 - s; digy <= digit2 + s; digy++) {

          for (digz = digit3 - s; digz <= digit3 + s; digz++) {

            if (digy == digit2 - s || digy == digit2 + s ||
                digz == digit3 - s || digz == digit3 + s ||
                digx == digit1 - s || digx == digit1 + s) {
              key = d * d * digx + d * digy + digz;
              if (key >= 0 && key < d * d * d) {

                searchCell(pointx, pointy, pointz, starts[key], ends[key], Cx,
                           Cy, Cz, &minq, &x, &y, &z);
              }
            }
          }
        }
      }
      bdist = findBoarderdistance(pointx, pointy, pointz, d, s, digit1, digit2,
                                  digit3);
    }

    Resx[i] = x;
    Resy[i] = y;
    Resz[i] = z;
  }
}

__global__ void distsQmin(float *Qx, float *Qy, float *Qz, float *Cx, float *Cy,
                          float *Cz, int lenghtC, float *x, float *y, float *z,
                          int lenghtQ) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lenghtQ;
       i = i + gridDim.x * blockDim.x) {
    float minq = 100;
    float dist;
    float tempx, tempy, tempz;
    for (int j = 0; j < lenghtC; j++) {
      dist = (Cx[j] - Qx[i]) * (Cx[j] - Qx[i]) +
             (Cy[j] - Qy[i]) * (Cy[j] - Qy[i]) +
             (Cz[j] - Qz[i]) * (Cz[j] - Qz[i]);
      dist = sqrt(dist);
      if (dist < minq) {
        minq = dist;
        tempx = Cx[j];
        tempy = Cy[j];
        tempz = Cz[j];
      }
    }
    x[i] = tempx;
    y[i] = tempy;
    z[i] = tempz;
  }
}

__global__ void gridval(int *starts, int *ends, float *Cx, float *Cy, float *Cz,
                        int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int digit1 = Cx[i] * d;
  int digit2 = Cy[i] * d;
  int digit3 = Cz[i] * d;
  int key = d * d * digit1 + d * digit2 + digit3;
  if (i > ends[key] || i < starts[key]) {
    printf("Error in grid construction \n");
  }
  if (i > 0) {
    digit1 = Cx[i - 1] * d;
    digit2 = Cy[i - 1] * d;
    digit3 = Cz[i - 1] * d;
    int key2 = d * d * digit1 + d * digit2 + digit3;
    if (key2 > key) {
      printf("Error not sorted\n");
    }
  }
}
void init_rand_points(float *p, int n) {
  int i;

  for (i = 0; i < n; i++) {
    p[i] = (float)(rand() - 1000) / (float)RAND_MAX;
  }
}

int main(int argc, char **argv) {

  /*Take the input lengths*/

   if (argc != 4) {
   printf("Enter size of the set and queries as arguments and the number of "
         "grid cells as arguments");
   exit(0);
  }
  srand(time(NULL));

  int lenghtC = 1 << atoi(argv[1]);
  int lenghtQ = 1 << atoi(argv[2]);
  int d = 1 << atoi(argv[3]);
  int numblocks = 1 << 10;
  int threadsPerBlock = 32;

  printf("Size of set 2 to %d  size of quiry set 2 to %d grid dimentions %d x %d x %d\n",atoi(argv[1]), atoi(argv[2]),d,d,d);
  printf("%d Threadblocks with %d threads per block \n", numblocks, threadsPerBlock );
  float *Csx, *Csy, *Csz, *Qsx, *Qsy, *Qsz;

  Csx = (float *)malloc(sizeof(float) * lenghtC);
  Csy = (float *)malloc(sizeof(float) * lenghtC);
  Csz = (float *)malloc(sizeof(float) * lenghtC);
  Qsx = (float *)malloc(sizeof(float) * lenghtQ);
  Qsy = (float *)malloc(sizeof(float) * lenghtQ);
  Qsz = (float *)malloc(sizeof(float) * lenghtQ);

  /*  Put numbers in the arrays*/
  init_rand_points(Csx, lenghtC);
  init_rand_points(Csy, lenghtC);
  init_rand_points(Csz, lenghtC);
  init_rand_points(Qsx, lenghtQ);
  init_rand_points(Qsy, lenghtQ);
  init_rand_points(Qsz, lenghtQ);

  /*Allocate space in device memory*/
  float *Cx, *Cy, *Cz, *Qx, *Qy, *Qz;
  float *Resx, *Resy, *Resz;

  CUDA_CALL(cudaMalloc(&Cx, lenghtC * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Cy, lenghtC * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Cz, lenghtC * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Qx, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Qy, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Qz, lenghtQ * sizeof(float)));

  /*Copy the numbers in the device */
  CUDA_CALL(cudaMemcpy(Cx, Csx, lenghtC * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Cy, Csy, lenghtC * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Cz, Csz, lenghtC * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(Qx, Qsx, lenghtQ * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Qy, Qsy, lenghtQ * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Qz, Qsz, lenghtQ * sizeof(float), cudaMemcpyHostToDevice));

  /*Make pointers so we can use the thrust libraty*/
  thrust ::device_ptr<float> Cx_ptr(Cx);
  thrust ::device_ptr<float> Cy_ptr(Cy);
  thrust ::device_ptr<float> Cz_ptr(Cz);
  thrust ::device_ptr<float> Qx_ptr(Qx);
  thrust ::device_ptr<float> Qy_ptr(Qy);
  thrust ::device_ptr<float> Qz_ptr(Qz);

  /*Find the grid node for each point in C*/
  float gridMakeTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int *keysC;
  CUDA_CALL(cudaMalloc(&keysC, lenghtC * sizeof(int)));

  generatekey<<<numblocks, threadsPerBlock>>>(Cx, Cy, Cz, lenghtC, keysC, d);
  cudaDeviceSynchronize();

  /*Sort by grid node*/
  thrust ::device_ptr<int> kc(keysC);
  thrust ::stable_sort_by_key(
      kc, kc + lenghtC, make_zip_iterator(make_tuple(Cx_ptr, Cy_ptr, Cz_ptr)));

  int *starts;
  CUDA_CALL(cudaMalloc(&starts, d * d * d * (sizeof(int))));
  CUDA_CALL(cudaMemset(starts, -1, d * d * d * (sizeof(int))));
  int *ends;
  CUDA_CALL(cudaMalloc(&ends, d * d * d * (sizeof(int))));
  CUDA_CALL(cudaMemset(ends, -1, d * d * d * (sizeof(int))));

  /*Find where its node starts and ends */
  findCellStartends<<<numblocks, threadsPerBlock>>>(keysC, lenghtC, starts,
                                                    ends);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gridMakeTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Time for grid creation   %f  ms \n", gridMakeTime);

  /*Validate the Grid*/
  gridval<<<lenghtC / threadsPerBlock, threadsPerBlock>>>(starts, ends, Cx, Cy,
                                                          Cz, d);

  CUDA_CALL(cudaMalloc(&Resx, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Resy, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Resz, lenghtQ * sizeof(float)));

  float elapsedTime;
  cudaEvent_t startse, stopse;
  cudaEventCreate(&startse);
  cudaEventCreate(&stopse);
  cudaEventRecord(startse, 0);

  /*Searrch for each query point */
  searchGrid<<<numblocks, threadsPerBlock>>>(Qx, Qy, Qz, starts, ends, Cx, Cy,
                                             Cz, Resx, Resy, Resz, d, lenghtQ);

  cudaEventRecord(stopse, 0);
  cudaEventSynchronize(stopse);
  cudaEventElapsedTime(&elapsedTime, startse, stopse);
  cudaEventDestroy(startse);
  cudaEventDestroy(stopse);
  printf("Search Time   %f ms  \n", elapsedTime);
  float *x;
  float *z;
  float *y;

  CUDA_CALL(cudaMalloc(&x, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&y, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&z, lenghtQ * sizeof(float)));
  /*Validation*/

  distsQmin<<<numblocks, threadsPerBlock>>>(Qx, Qy, Qz, Cx, Cy, Cz, lenghtC, x,
  y, z,lenghtQ);

  cudaDeviceSynchronize();

  float *gridx, *gridy, *gridz, *minx, *miny, *minz;

  gridx = (float *)malloc(lenghtQ * sizeof(float));
  gridy = (float *)malloc(lenghtQ * sizeof(float));
  gridz = (float *)malloc(lenghtQ * sizeof(float));
  minx = (float *)malloc(lenghtQ * sizeof(float));
  miny = (float *)malloc(lenghtQ * sizeof(float));
  minz = (float *)malloc(lenghtQ * sizeof(float));

  cudaMemcpy(gridx, Resx, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(gridy, Resy, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(gridz, Resz, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpy(minx, x, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(miny, y, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(minz, z, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);


  float *Qhx, *Qhy, *Qhz;
  Qhx = (float *)malloc(lenghtQ * sizeof(float));
  Qhy = (float *)malloc(lenghtQ * sizeof(float));
  Qhz = (float *)malloc(lenghtQ * sizeof(float));

  cudaMemcpy(Qhx, Qx, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Qhy, Qy, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Qhz, Qz, lenghtQ * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  int s = 0;
  int c = 0;
  for (int i = 0; i < lenghtQ; i++) {
    if (minx[i] != gridx[i] || miny[i] != gridy[i] || minz[i] != gridz[i]) {
      s++;
    } else {
      c++;
    }
  }
  printf("Wrong number of points %d ", s);
  printf("Ritght number of points  %d \n", c);

  CUDA_CALL(cudaFree(Cx));
  CUDA_CALL(cudaFree(Cy));
  CUDA_CALL(cudaFree(Cz));
  CUDA_CALL(cudaFree(Qx));
  CUDA_CALL(cudaFree(Qy));
  CUDA_CALL(cudaFree(Qz));
  CUDA_CALL(cudaFree(Resx));
  CUDA_CALL(cudaFree(Resy));
  CUDA_CALL(cudaFree(Resz));
  CUDA_CALL(cudaFree(keysC));
  CUDA_CALL(cudaFree(ends));
  CUDA_CALL(cudaFree(starts));
  CUDA_CALL(cudaFree(x));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(z));

  return 0;
}
