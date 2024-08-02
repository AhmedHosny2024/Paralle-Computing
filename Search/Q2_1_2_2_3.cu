#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

__global__ void binarySearch(float *array, int size, float target, int *results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid;
    int end = size - 1;
    while (start <= end) {
        int mid = (start + end) / 2;
        if (array[mid] == target) {
            results[tid] = mid;
            return;
        } else if (array[mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    results[tid] = -1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <target>\n", argv[0]);
        return 1;
    }

    // Read input array from file
    char *filename = argv[1];
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    float target = atof(argv[2]);
    float *array = NULL;
    int n = 0;
    float x;
    while (fscanf(file, "%f", &x) == 1) {
        array = (float *) realloc(array, (n + 1) * sizeof(float));
        array[n] = x;
        n++;
    }
    fclose(file);
    float *d_array;
    cudaMalloc(&d_array, n * sizeof(float));
    cudaMemcpy(d_array, array, n * sizeof(float), cudaMemcpyHostToDevice);
    int *d_result;
    cudaMalloc(&d_result, sizeof(int));
    binarySearch<<<1, BLOCK_SIZE>>>(d_array, n, target, d_result);
    int result ;
    
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (result < n) {
        printf("Index of target element: %d\n", result);
    } else {
        printf("Target element not found\n");
    }
    cudaFree(d_array);
    cudaFree(d_result);
    free(array);

    return 0;
}
