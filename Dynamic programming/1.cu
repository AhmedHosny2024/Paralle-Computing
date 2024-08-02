#include <stdio.h>

#define MAX_DEPTH 4

__global__ void recursiveSearch(float *array, int start, int end, float target, int *result, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (depth == 0 && idx == 0) {
        *result = -1;
    }

    if (depth >= MAX_DEPTH  || end - start <= 32) {
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
        // if (depth == 0) {
        //     printf("depth: %d, subStart: %d, subEnd: %d\n", depth, subStart, subEnd);
        // }
        
        if (tid == blockDim.x - 1) {
            subEnd = end;
        }

        
        recursiveSearch<<<1, 32>>>(array, subStart, subEnd, target, result, depth + 1);
        // cudaDeviceSynchronize();
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_file target\n", argv[0]);
        return 1;
    }

    FILE *inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        fprintf(stderr, "Error opening the input file.\n");
        return 1;
    }

    int size = 0;
    float temp;
    while (fscanf(inputFile, "%f", &temp) == 1) {
        size++;
    }

    fseek(inputFile, 0, SEEK_SET);

    float *deviceArr;
    float *hostArr = (float *)malloc(size * sizeof(float));
    float target = atof(argv[2]);

    for (int i = 0; i < size; ++i) {
        fscanf(inputFile, "%f", &hostArr[i]);
    }

    fclose(inputFile);

    cudaMalloc(&deviceArr, size * sizeof(float));
    cudaMemcpy(deviceArr, hostArr, size * sizeof(float), cudaMemcpyHostToDevice);

    int *deviceRes, hostRes;
    cudaMalloc(&deviceRes, sizeof(int));

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    recursiveSearch<<<1, 32>>>(deviceArr, 0, size - 1, target, deviceRes, 0);

    cudaDeviceSynchronize();

    cudaMemcpy(&hostRes, deviceRes, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", hostRes);

    free(hostArr);
    cudaFree(deviceArr);
    cudaFree(deviceRes);

    return 0;
}
