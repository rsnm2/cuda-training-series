#include <stdio.h>

__global__ void hello(){

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello from index: %u (block: %u, thread: %u)\n", index, blockIdx.x, threadIdx.x);
}

int main(){

  hello<<<2, 2>>>();
  cudaDeviceSynchronize();
}

