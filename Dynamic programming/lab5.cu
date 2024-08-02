/*
Name: Abdelaziz Salah
BN: 2, Sec: 2
@Desc: This file contains an implementation for the binary search but using 
dynamic sharing.
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define N 1000
#define M 500

#define MAX_DEPTH 5
#define MAX_ERR 1e-3
__global__ void MatAdd(float A[N][M], float B[N][M], float C[N][M])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Note: the first indexing specifies the row (y-axis), the second one specifies the column (x-axis)
    C[j][i] = A[j][i] + B[j][i];
}

void lab4(){

     // statically allocate the matrices
     float a[N][M], b[N][M], c[N][M];

    // Initialize a, b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            a[i][j] = i * 1.1;
            b[i][j] = j * 1.1;
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C; // Device pointer for the 2D array

    cudaMalloc((void**)&d_A, sizeof(float) * N * M);
    cudaMalloc((void**)&d_B, sizeof(float) * N * M);
    cudaMalloc((void**)&d_C, sizeof(float) * N * M);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, a, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16);

    // Note that M here specifies the number of columns (on the X-axis), while N specifies the rows
    dim3 GridSize ((M - 1) / ThreadsPerBlock.x + 1, (N - 1) / ThreadsPerBlock.y + 1);

    // Casting the single pointer to an array of pointers
    MatAdd<<<GridSize, ThreadsPerBlock>>>((float(*) [M])d_A, (float(*) [M])d_B, (float(*) [M])d_C);

    // Transfer data back to host memory
    cudaMemcpy(c, d_C, sizeof(float) * N * M, cudaMemcpyDeviceToHost);


    // Verification
    for(int i = 0; i < N; i++){
      for(int j = 0; j < M; j++){
         assert(fabs(c[i][j] - a[i][j] - b[i][j]) < MAX_ERR);
      }
    }
    printf("PASSED\n");

    // Deallocate device memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);

    // No need to deallocate host memory
}

__global__ void subBinarySearch(int bgn, int end, float* array, int * result, const float target) { 
        int low = bgn;
        int high = end;

        while (low <= high) {
            int mid = (low + high) / 2;

            if (array[mid] == target) {
                /// found
                *result = mid;
                return;
            } else if (array[mid] >= target) {
                /// look left
                high = mid - 1;
            } else {
                /// look right
                low = mid + 1;
            }
        }
}

__global__ void recursiveBinarySearch(float *array,const int maxDepth, const float target,
                           int * result, int depth, int bgn, int end) {
                            
    /// getting the index.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /// base case 
    if (depth == 0 && idx == 0) {
        *result = -1;
    }

    /// limiting the depth and the size of elements for each thread.
    if (depth < maxDepth  &&  - bgn + end > 32) {
        /// get the total number of elements
        int numberOfElements =  - bgn + end + 1;

        /// define the number of blocks needed to evaluate this numberOfElements.
        int numberOfBlocks = numberOfElements / blockDim.x;

        /// applying the binary search 
        /// each thread should bgn from the current bgn and padded by the index of the thread
        /// multiplied by the chunkSize 
        int threadBegin = bgn + idx * numberOfBlocks;
        int threadEnd = threadBegin + numberOfBlocks - 1;

        /// define the last index. 
        if (threadIdx.x == blockDim.x - 1) {
            threadEnd = end;
        } 
        
        /// applying recursion.
        recursiveBinarySearch<<<1, 32>>>(array, maxDepth, target, result, depth + 1,threadBegin, threadEnd);
        
    } else {
      /// 1,1 because I just need 1 thread to work on this block
      /// and I need only one block
      subBinarySearch<<<1,1>>>(bgn, end, array, result, target);
    }
}


void lab2(){

     // statically allocate the matrices
     float a[N][M], b[N][M], c[N][M];

    // Initialize a, b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            a[i][j] = i * 1.1;
            b[i][j] = j * 1.1;
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C; // Device pointer for the 2D array

    cudaMalloc((void**)&d_A, sizeof(float) * N * M);
    cudaMalloc((void**)&d_B, sizeof(float) * N * M);
    cudaMalloc((void**)&d_C, sizeof(float) * N * M);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, a, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16);

    // Note that M here specifies the number of columns (on the X-axis), while N specifies the rows
    dim3 GridSize ((M - 1) / ThreadsPerBlock.x + 1, (N - 1) / ThreadsPerBlock.y + 1);

    // Casting the single pointer to an array of pointers
    MatAdd<<<GridSize, ThreadsPerBlock>>>((float(*) [M])d_A, (float(*) [M])d_B, (float(*) [M])d_C);

    // Transfer data back to host memory
    cudaMemcpy(c, d_C, sizeof(float) * N * M, cudaMemcpyDeviceToHost);


    // Verification
    for(int i = 0; i < N; i++){
      for(int j = 0; j < M; j++){
         assert(fabs(c[i][j] - a[i][j] - b[i][j]) < MAX_ERR);
      }
    }
    printf("PASSED\n");

    // Deallocate device memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);

    // No need to deallocate host memory
}


/// old binary search 
__global__ void binarySearch(float *array, int size, float target, int *results) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tidx;
    int end = size - 1;
    while (start <= end) {
        int mid = (start + end) / 2;
        if (array[mid] == target) {
            results[tidx] = mid;
            return;
        } else if (array[mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    results[tidx] = -1;
}

void printArr (float* arr, int n) { 
  for (int i = 0; i < n ; i++) {
    printf("%f", arr[i]); 
    printf(" "); 
  }
  printf("\n"); 
}

float* readInputArrayFromFile(char* filename, int& arraySize) { 
    // char *filename = argv[1];
    float* array = NULL ; 
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    /// Read the target from the command
    // float target = atof(argv[2]);

    /// define data on the CPU
    // float *array = NULL;
    // int arraySize = 0;
    float number;
    while (fscanf(file, "%f", &number) == 1) {
        array = (float *) realloc(array, (arraySize + 1) * sizeof(float));
        array[arraySize] = number;
        arraySize++;
    }
    // printf("Before Call");
    // printArr(array, arraySize); 
    fclose(file);
    return array; 
}
void printResult(int targetIndex, int arraySize) { 
   if (targetIndex < arraySize && targetIndex != -1) {
        printf("Index of target element: %d :)\n", targetIndex);
    } else {
        printf("Target does not exist :( \n");
    }
}

int main(int argc, char *argv[]) {
    /// 1. make sure that the command is correct and has 3 exact arguments
    if (argc != 3) {
        printf("The input command is not correct \n");
        return -1;
    }

    // 2. Read the arguments from the command
    char *filePath = argv[1];
    float target = atof(argv[2]);

    /// 3. open the file
    FILE *file = fopen(filePath, "r");
    if (file == NULL) {
        perror("Error in the opening of the file :()");
        return -1;
    }

    /// 4. define data on the CPU
    int arraySize = 0;
    float* array = readInputArrayFromFile(filePath, arraySize); 
    
    /// 5. allocating the data on the device
    float *d_array;
    cudaMalloc(&d_array, sizeof(float)* arraySize);

    /// 6. copying data to the device
    cudaMemcpy(d_array, array, sizeof(float)* arraySize, cudaMemcpyHostToDevice);

    /// 7. allocating memory for the result on the gpu. 
    int *d_res;
    cudaMalloc(&d_res, sizeof(int));

    /// 8. calling the kernal 
    recursiveBinarySearch<<<1, 8>>>(d_array,MAX_DEPTH,  target, d_res, 0,0, arraySize - 1);
    
    /// 9. copying data from the gpu to the cpu
    int targetIndex ;
    cudaMemcpy(&targetIndex, d_res, sizeof(int), cudaMemcpyDeviceToHost);

    /// 10. printing the result.
    printResult(targetIndex, arraySize); 
 
    /// 11. free the memory
    cudaFree(d_array);
    cudaFree(d_res);
    free(array);

    return 0;
}