#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void kernal(float *d_array, int n, float *d_sum) {
    __shared__ float v[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n)
        v[tid] = d_array[i];
    else
        v[tid] = 0;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /=2) {
        if (tid < stride) {
            // printf("at stride : %d blockIdx.x: %d, threadIdx.x: %d , tid : %d tid + stide: %d add %f with %f \n",stride, blockIdx.x, threadIdx.x,tid,tid + stride,v[tid],v[tid + stride]);
            v[tid] += v[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_sum, v[0]);
    }
    
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }
    float *h_array = NULL;
    int n = 0;
    float x;
    while (fscanf(file, "%f", &x) == 1) {
        h_array = (float *) realloc(h_array, (n + 1) * sizeof(float));
        h_array[n] = x;
        n++;
    }
    fclose(file);
    float *d_array;
    float *d_sum;
    float sum;
    int size = n * sizeof(float);
    cudaMalloc(&d_array, size);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);    
    int GRID_SIZE=ceil(float(n)/BLOCK_SIZE);
    // printf("GRID_SIZE: %d\n", GRID_SIZE);
    kernal<<<GRID_SIZE, BLOCK_SIZE>>>(d_array, n, d_sum);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", sum);
    cudaFree(d_array);
    cudaFree(d_sum);
    free(h_array);

    return 0;
}
