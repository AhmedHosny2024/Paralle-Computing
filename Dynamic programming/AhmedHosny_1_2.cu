#include <stdio.h>
#include <stdlib.h>

#define MAX_DEPTH 4

__global__ void recursiveSearch(float *array, int start, int end, float target, int *result, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (depth == 0 && idx == 0) {
        *result = -1;
    }

    if (depth >= MAX_DEPTH || end - start <= 32) {
        int low = start;
        int high = end;

        while (low <= high) {
            int mid = (low + high) / 2;

            if (array[mid] == target) {
                *result = mid;
                return;
            } else if (array[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

    } else {
        int size = end - start + 1;
        int chunkSize = size / blockDim.x;

        int subStart = start + idx * chunkSize;
        int subEnd = subStart + chunkSize - 1;

        if (tid == blockDim.x - 1) {
            subEnd = end;
        }

        recursiveSearch<<<1, 32>>>(array, subStart, subEnd, target, result, depth + 1);
    }
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_file target\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        fprintf(stderr, "Error opening the input file.\n");
        return EXIT_FAILURE;
    }

    int size = 0;
    float temp;
    while (fscanf(inputFile, "%f", &temp) == 1) {
        size++;
    }

    fseek(inputFile, 0, SEEK_SET);

    float *hostArr = (float *)malloc(size * sizeof(float));
    if (hostArr == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(inputFile);
        return EXIT_FAILURE;
    }

    float target = atof(argv[2]);

    for (int i = 0; i < size; ++i) {
        fscanf(inputFile, "%f", &hostArr[i]);
    }

    fclose(inputFile);

    float *deviceArr;
    checkCudaError(cudaMalloc(&deviceArr, size * sizeof(float)), "allocating device array");
    checkCudaError(cudaMemcpy(deviceArr, hostArr, size * sizeof(float), cudaMemcpyHostToDevice), "copying array to device");

    int *deviceRes;
    checkCudaError(cudaMalloc(&deviceRes, sizeof(int)), "allocating result on device");

    checkCudaError(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH), "setting device sync depth limit");
    recursiveSearch<<<1, 32>>>(deviceArr, 0, size - 1, target, deviceRes, 0);

    checkCudaError(cudaDeviceSynchronize(), "synchronizing device");

    int hostRes;
    checkCudaError(cudaMemcpy(&hostRes, deviceRes, sizeof(int), cudaMemcpyDeviceToHost), "copying result to host");

    printf("%d\n", hostRes);

    free(hostArr);
    cudaFree(deviceArr);
    cudaFree(deviceRes);

    return EXIT_SUCCESS;
}
