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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int digit1 = Cx[i] * d;
  int digit2 = Cy[i] * d;
  int digit3 = Cz[i] * d;
  keys[i] = d * d * digit1 + d * digit2 + digit3;
}

__global__ void findCellStartends(int *keys, int len, int *starts, int *ends) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int support[];
  support[threadIdx.x] = keys[i];
  if (threadIdx.x == blockDim.x - 1) {
    support[blockDim.x] = keys[i + 1];
  }

  if (threadIdx.x == 0) {
    if (i > 0) {
      support[blockDim.x + 1] = keys[i - 1];
    }
  }
  __syncthreads();

  if (i > 0) {
    if (threadIdx.x > 0) {
      if (support[threadIdx.x] != support[threadIdx.x - 1]) {
        starts[support[threadIdx.x]] = i;
      }
    } else {
      if (support[threadIdx.x] != support[blockDim.x + 1]) {
        starts[support[threadIdx.x]] = i;
      }
    }
  } else {
    starts[support[0]] = 0;
  }
  if (i != len - 1) {

    if (support[threadIdx.x] != support[threadIdx.x + 1]) {
      ends[support[threadIdx.x]] = i;
    }
  } else {
    ends[support[threadIdx.x]] = len - 1;
  }
}

__global__ void findCellStart(int *keys, int len, int *starts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0) {
    if (keys[i] != keys[i - 1]) {
      starts[keys[i]] = i;
    }
  } else {
    starts[keys[0]] = 0;
  }
}

__global__ void distsQmin(float *Qx, float *Qy, float *Qz, float *Cx, float *Cy,
                          float *Cz, int lenghtC, float *x, float *y,
                          float *z) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
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

__device__ float findBoarderdistance(float pointx, float pointy, float pointz,
                                     int d, int s, int digit1, int digit2,
                                     int digit3) {
  float bdist = 100;

  if (digit1 + s < d - 1) {
    bdist = -pointx + (float)(digit1 + s + 1) / d;
  }
  if (digit2 + s < d - 1) {
    bdist = min(-pointy + (float)(digit2 + s + 1) / d, bdist);
  }
  if (digit3 + s < d - 1) {
    bdist = min(-pointz + (float)(digit3 + s + 1) / d, bdist);
  }
  if (digit1 - s > 0) {
    bdist = min(+pointx + (float)(-digit1 + s) / d, bdist);
  }
  if (digit2 - s > 0) {
    bdist = min(+pointy + (float)(-digit2 + s) / d, bdist);
  }
  if (digit3 - s > 0) {
    bdist = min(+pointz + (float)(-digit3 + s) / d, bdist);
  }
  return bdist;
}
__global__ void searchGrid(float *Qx, float *Qy, float *Qz, int *starts,
                           int *ends, float *Cx, float *Cy, float *Cz,
                           float *Resx, float *Resy, float *Resz, int d,
                           int lenghtQ, int *startsQ, int max) {

  for (int t = blockIdx.x; t < d * d * d; t = t + gridDim.x) {
    if (startsQ[t] >= 0) {

      extern __shared__ float buf[];
      float *Cxshared = buf;
      float *Cyshared = (float *)&Cxshared[-starts[t] + ends[t] + 1];
      float *Czshared = (float *)&Cyshared[-starts[t] + ends[t] + 1];

      for (int i = starts[t] + threadIdx.x; i <= ends[t]; i = i + blockDim.x) {
        Cxshared[i - starts[t]] = Cx[i];
        Cyshared[i - starts[t]] = Cy[i];
        Czshared[i - starts[t]] = Cz[i];
      }
      __syncthreads();

      int i = startsQ[t] + threadIdx.x;

      while (1) {
        if (i >= lenghtQ) {
          break;
        }

        float pointx = Qx[i];
        float pointy = Qy[i];
        float pointz = Qz[i];
        int digit1 = pointx * d;
        int digit2 = pointy * d;
        int digit3 = pointz * d;
        int key = d * d * digit1 + d * digit2 + digit3;

        if (key != t) {
          break;
        }
        float x = 0, y = 0, z = 0;
        float minq = 100;
        int s = 0;
        float bdist = findBoarderdistance(pointx, pointy, pointz, d, s, digit1,
                                          digit2, digit3);

        float dist;
        int digx, digy, digz;
        if (starts[key] >= 0) {
          for (int j = 0; j < ends[key] - starts[key] + 1; j++) {
            dist = (Cxshared[j] - pointx) * (Cxshared[j] - pointx) +
                   (Cyshared[j] - pointy) * (Cyshared[j] - pointy) +
                   (Czshared[j] - pointz) * (Czshared[j] - pointz);
            dist = sqrt(dist);
            if (dist < minq) {
              minq = dist;
              x = Cxshared[j];
              y = Cyshared[j];
              z = Czshared[j];
            }
          }
        }

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
                    searchCell(pointx, pointy, pointz, starts[key], ends[key],
                               Cx, Cy, Cz, &minq, &x, &y, &z);
                  }
                }
              }
            }
          }
          bdist = findBoarderdistance(pointx, pointy, pointz, d, s, digit1,
                                      digit2, digit3);
        }

        Resx[i] = x;
        Resy[i] = y;
        Resz[i] = z;
        i = i + blockDim.x;
      }
    }
  }
}

__global__ void find_maximum_kernel(int *starts, int *ends, int *max,
                                    int *mutex, int n) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  extern __shared__ int cache[];

  int temp = -1.0;
  while (index + offset < n) {
    if (temp < -starts[index + offset] + ends[index + offset] + 1) {
      temp = -starts[index + offset] + ends[index + offset] + 1;
    }
    offset += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  // reduction
  unsigned int i = blockDim.x / 2;
  while (i != 0) {
    if (threadIdx.x < i) {
      if (cache[threadIdx.x] < cache[threadIdx.x + i]) {
        cache[threadIdx.x] = cache[threadIdx.x + i];
      }
    }

    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x == 0) {
    while (atomicCAS(mutex, 0, 1) != 0)
      ; // lock
    if (*max < cache[0]) {
      *max = cache[0];
    }
    atomicExch(mutex, 0); // unlock
  }
}

void init_rand_points(float *p, int n) {
  int i;

  for (i = 0; i < n; i++) {
    p[i] = (float)(rand() - 1000) / (float)RAND_MAX;
  }
}
__global__ void checkmax(int max, int *starts, int *ends) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (ends[i] - starts[i] + 1 > max) {

    printf("Error Didin't find max\n");
  }
}

int main(int argc, char **argv) {

  if (argc != 3) {
    printf("Enter size of the set and queries as arguments \n");
    exit(0);
  }
  srand(time(NULL));

  int lenghtC = 1 << atoi(argv[1]);
  int lenghtQ = 1 << atoi(argv[2]);
  int d = 1 << 6;
  int numblocks = 1 << 10;
  int threadsPerBlock = 32;

  printf("Size of set 2 to %d  size of quiry set 2 to %d grid dimentions %d x "
         "%d x %d\n",
         atoi(argv[1]), atoi(argv[2]),d,d,d);
  printf("%d Threadblocks with %d threads per block \n", numblocks, threadsPerBlock );


  float *Csx, *Csy, *Csz, *Qsx, *Qsy, *Qsz;

  Csx = (float *)malloc(sizeof(float) * lenghtC);
  Csy = (float *)malloc(sizeof(float) * lenghtC);
  Csz = (float *)malloc(sizeof(float) * lenghtC);
  Qsx = (float *)malloc(sizeof(float) * lenghtQ);
  Qsy = (float *)malloc(sizeof(float) * lenghtQ);
  Qsz = (float *)malloc(sizeof(float) * lenghtQ);

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

  cudaMemcpy(Cx, Csx, lenghtC * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Cy, Csy, lenghtC * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Cz, Csz, lenghtC * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(Qx, Qsx, lenghtQ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Qy, Qsy, lenghtQ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Qz, Qsz, lenghtQ * sizeof(float), cudaMemcpyHostToDevice);

  /*Make pointers so we can use the thrust libraty*/
  thrust ::device_ptr<float> Cx_ptr(Cx);
  thrust ::device_ptr<float> Cy_ptr(Cy);
  thrust ::device_ptr<float> Cz_ptr(Cz);
  thrust ::device_ptr<float> Qx_ptr(Qx);
  thrust ::device_ptr<float> Qy_ptr(Qy);
  thrust ::device_ptr<float> Qz_ptr(Qz);

  /*Count time for grid creation*/
  float gridMakeTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  /*Find the grid node for each point in C*/
  int *keysC;
  CUDA_CALL(cudaMalloc(&keysC, lenghtC * sizeof(int)));

  generatekey<<<lenghtC / threadsPerBlock, threadsPerBlock>>>(
      Cx, Cy, Cz, lenghtC, keysC, d);
  cudaDeviceSynchronize();

  /*Sort by grid node*/
  thrust ::device_ptr<int> kc(keysC);
  thrust ::stable_sort_by_key(
      kc, kc + lenghtC, make_zip_iterator(make_tuple(Cx_ptr, Cy_ptr, Cz_ptr)));

  /*Find where its key starts and where it ends in the sorted array*/
  int *starts;
  CUDA_CALL(cudaMalloc(&starts, d * d * d * (sizeof(int))));
  CUDA_CALL(cudaMemset(starts, -1, d * d * d * (sizeof(int))));
  int *ends;
  CUDA_CALL(cudaMalloc(&ends, d * d * d * (sizeof(int))));
  CUDA_CALL(cudaMemset(ends, -1, d * d * d * (sizeof(int))));
  findCellStartends<<<lenghtC / threadsPerBlock, threadsPerBlock,
                      (threadsPerBlock + 2) * sizeof(int)>>>(keysC, lenghtC,
                                                             starts, ends);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gridMakeTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Time for grid creation   %f  ms \n", gridMakeTime);

  /*Count sorting time for Q ie Putting Q in the grid*/
  float sortTime;

  cudaEvent_t startsort, stopsort;
  cudaEventCreate(&startsort);
  cudaEventCreate(&stopsort);
  cudaEventRecord(startsort, 0);

  int *keysQ;
  CUDA_CALL(cudaMalloc(&keysQ, lenghtQ * sizeof(int)));
  generatekey<<<lenghtQ / threadsPerBlock, threadsPerBlock>>>(
      Qx, Qy, Qz, lenghtQ, keysQ, d);
  cudaDeviceSynchronize();
  thrust ::device_ptr<int> kq(keysQ);
  thrust ::stable_sort_by_key(
      kq, kq + lenghtQ, make_zip_iterator(make_tuple(Qx_ptr, Qy_ptr, Qz_ptr)));
  cudaDeviceSynchronize();
  int *startsQ;
  CUDA_CALL(cudaMalloc(&startsQ, d * d * d * (sizeof(int))));
  CUDA_CALL(cudaMemset(startsQ, -1, d * d * d * (sizeof(int))));
  findCellStart<<<lenghtQ / threadsPerBlock, threadsPerBlock>>>(keysQ, lenghtQ,
                                                                startsQ);

  cudaEventRecord(stopsort, 0);
  cudaEventSynchronize(stopsort);
  cudaEventElapsedTime(&sortTime, startsort, stopsort);
  cudaEventDestroy(startsort);
  cudaEventDestroy(stopsort);

  printf("Sorting Q time  %f  ms \n", sortTime);

  gridval<<<lenghtC / threadsPerBlock, threadsPerBlock>>>(starts, ends, Cx, Cy,
                                                          Cz, d);

  CUDA_CALL(cudaMalloc(&Resx, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Resy, lenghtQ * sizeof(float)));
  CUDA_CALL(cudaMalloc(&Resz, lenghtQ * sizeof(float)));

  float maxtime;
  cudaEvent_t startmax, stopmax;
  cudaEventCreate(&startmax);
  cudaEventCreate(&stopmax);
  cudaEventRecord(startmax, 0);

  int *mutex;
  CUDA_CALL(cudaMalloc(&mutex, sizeof(int)));
  CUDA_CALL(cudaMemset(mutex, 0, (sizeof(int))));
  int *d_max;
  CUDA_CALL(cudaMalloc(&d_max, sizeof(int)));
  CUDA_CALL(cudaMemset(d_max, -1, (sizeof(int))));
  int h_max;
  find_maximum_kernel<<<numblocks, threadsPerBlock,
                        threadsPerBlock * sizeof(int)>>>(starts, ends, d_max,
                                                         mutex, d * d * d);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(stopmax, 0);
  cudaEventSynchronize(stopmax);
  cudaEventElapsedTime(&maxtime, startmax, stopmax);
  cudaEventDestroy(startmax);
  cudaEventDestroy(stopmax);
  printf("Finding  Maximum number of points in a shell time  %f ms  \n", maxtime);

  checkmax<<<d * d * d / threadsPerBlock, threadsPerBlock>>>(h_max, starts,
                                                             ends);
  cudaDeviceSynchronize();
  float elapsedTime;
  cudaEvent_t startse, stopse;
  cudaEventCreate(&startse);
  cudaEventCreate(&stopse);
  cudaEventRecord(startse, 0);

  searchGrid<<<numblocks, threadsPerBlock, 3 * h_max * sizeof(float)>>>(
      Qx, Qy, Qz, starts, ends, Cx, Cy, Cz, Resx, Resy, Resz, d, lenghtQ,
      startsQ, h_max);

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

  /*    Validation  */


  distsQmin<<<lenghtQ / threadsPerBlock, threadsPerBlock>>>(
      Qx, Qy, Qz, Cx, Cy, Cz, lenghtC, x, y, z);

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
  CUDA_CALL(cudaFree(keysQ));
  CUDA_CALL(cudaFree(ends));
  CUDA_CALL(cudaFree(starts));
  CUDA_CALL(cudaFree(startsQ));
  CUDA_CALL(cudaFree(x));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(z));

  return 0;
}
