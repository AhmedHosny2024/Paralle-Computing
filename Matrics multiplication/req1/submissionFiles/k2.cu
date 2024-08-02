// # Vector addition in CUDA (Kernel3: level2 parallelism-> multiple blocks, each with multiple threads)
// %%cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
// #define N 10000000
#define MAX_ERR 1e-6

/*
    each thread produces one output matrix row/
*/
__global__ void kernel2(float **A, float **B, float **C, int rows, int cols)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= cols)
        return;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;

    for (int j = 0; j < rows; j++)
        // if (i < rows && j < cols)
        C[j][i] = A[j][i] + B[j][i];
}

// Executing kernel
// int block_size = 256;
// int grid_size = ((N + block_size) / block_size);
// vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
int main(int argc, char *argv[])
{
    /// getting arguments
    if (argc != 3)
    {
        fprintf(stderr, "Can not open the file: %s", argv[0]);
        exit(-1);
    }

    /// openning files
    FILE *inputFile = fopen(argv[1], "r");
    FILE *outputFile = fopen(argv[2], "w");
    if (!inputFile || !outputFile)
    {
        fprintf(stderr, "Error opening file.\n");
        exit(-2);
    }

    /// reading # of test cases.
    int tc;
    fscanf(inputFile, "%d", &tc);

    for (int t = 0; t < tc; t++)
    {
        /// instantiating pointers for matricies.
        float **a, **b, **out;
        float **d_a, **d_b, **d_out;

        // reading rows and columns.
        int rows, cols;
        fscanf(inputFile, "%d %d", &rows, &cols);

        // Allocate host memory
        a = (float **)malloc(sizeof(float *) * rows);
        b = (float **)malloc(sizeof(float *) * rows);
        out = (float **)malloc(sizeof(float *) * rows);

        /// Allocating memory for each matrix
        for (int i = 0; i < rows; i++)
        {
            a[i] = (float *)malloc(sizeof(float) * cols);
            b[i] = (float *)malloc(sizeof(float) * cols);
            out[i] = (float *)malloc(sizeof(float) * cols);
        }

        // Read matrix1
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                fscanf(inputFile, "%f", &a[i][j]);

        // Read matrix2
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                fscanf(inputFile, "%f", &b[i][j]);

        // Allocate device memory for matrices
        cudaMalloc((void ***)&d_a, sizeof(float *) * rows);
        cudaMalloc((void ***)&d_b, sizeof(float *) * rows);
        cudaMalloc((void ***)&d_out, sizeof(float *) * rows);

        float **tempA = (float **)malloc(sizeof(float *) * rows);
        float **tempB = (float **)malloc(sizeof(float *) * rows);
        float **tempC = (float **)malloc(sizeof(float *) * rows);

        for (int i = 0; i < rows; i++)
        {
            cudaMalloc((void **)&tempA[i], sizeof(float) * cols);
            cudaMalloc((void **)&tempB[i], sizeof(float) * cols);
            cudaMalloc((void **)&tempC[i], sizeof(float) * cols);
        }

        cudaMemcpy(d_a, tempA, sizeof(float *) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, tempB, sizeof(float *) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, tempC, sizeof(float *) * rows, cudaMemcpyHostToDevice);

        // Transfer data from host to device memory
        for (int i = 0; i < rows; i++)
        {
            cudaMemcpy(tempA[i], a[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
            cudaMemcpy(tempB[i], b[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
        }

        /// define kernal diminsions.
        // # int block_size = 256;
        // # int grid_size = ((N + block_size) / block_size);
        dim3 threadsPerBlock(16, 16);
        dim3 grid_size((cols / threadsPerBlock.x) + 1, (rows / threadsPerBlock.y) + 1);
        kernel2<<<grid_size, threadsPerBlock>>>(d_a, d_b, d_out, rows, cols);

        // Transfer data back to host memory
        // cudaMemcpy(out,d_out, sizeof(float*) * rows, cudaMemcpyDeviceToHost);
        // Copy data back to host
        // Transfer data back to host memory
        for (int i = 0; i < rows; i++)
            cudaMemcpy(out[i], tempC[i], sizeof(float) * cols, cudaMemcpyDeviceToHost);

        // Write result to output file
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                fprintf(outputFile, "%.2f ", out[i][j]);

            fprintf(outputFile, "\n");
        }

        // Verification
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                assert(abs(out[i][j] - a[i][j] - b[i][j]) < 0.1);

        // Free device memory
        for (int i = 0; i < rows; i++)
        {
            cudaFree(tempA[i]);
            cudaFree(tempB[i]);
            cudaFree(tempC[i]);
        }
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        free(tempA);
        free(tempB);
        free(tempC);

        // Free host memory
        for (int i = 0; i < rows; i++)
        {
            free(a[i]);
            free(b[i]);
            free(out[i]);
        }
        free(a);
        free(b);
        free(out);

    }
    // closing files
    fclose(inputFile);
    fclose(outputFile);
}
