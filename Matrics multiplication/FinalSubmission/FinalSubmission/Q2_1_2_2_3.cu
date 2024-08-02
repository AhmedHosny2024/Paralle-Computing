#include <stdio.h>

#define BLOCK_SIZE 1

// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMul(float *A, float *B, float *C, int rows, int cols) {
    int idx = blockIdx.x ;
    if (idx < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += A[idx * cols + j] *B[j];
        }
        C[idx] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    char *inputFile = argv[1];
    char *outputFile = argv[2];

    FILE *inputFp = fopen(inputFile, "r");
    FILE *outputFp = fopen(outputFile, "w");

    int numTests;
    fscanf(inputFp, "%d", &numTests);
    for (int t = 0; t < numTests; ++t) {
        int rows;
        int cols;
        fscanf(inputFp, "%d %d",&rows,&cols);
        
        // allocate memory for matrix, vector and result
        float *matrix = (float *)malloc(rows * cols * sizeof(float));
        float *vector = (float *)malloc(cols * sizeof(float));
        float *result = (float *)malloc(rows * sizeof(float));

        // Read matrix
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                fscanf(inputFp, "%f", &matrix[i * cols + j]);
            }
        }

        // Read vector
        for (int i = 0; i < cols; ++i) {
            fscanf(inputFp, "%f", &vector[i]);
        }

        // CUDA memory allocation
        float *d_matrix, *d_vector, *d_result;
        cudaMalloc((void **)&d_matrix, rows * cols * sizeof(float));
        cudaMalloc((void **)&d_vector, cols * sizeof(float));
        cudaMalloc((void **)&d_result, rows * sizeof(float));

        // Copy matrix and vector to device
        cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector, vector, cols * sizeof(float), cudaMemcpyHostToDevice);

        // Define grid and block sizes
        int numBlocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Launch kernel
        matrixVectorMul<<<numBlocks, BLOCK_SIZE>>>(d_matrix,d_vector, d_result, rows, cols);

        // Copy result back to host
        cudaMemcpy(result, d_result, rows * sizeof(float), cudaMemcpyDeviceToHost);

        // Write result to output file
        for (int i = 0; i < rows; ++i) {
            fprintf(outputFp, "%.1f\n", result[i]);
        }

        // Clean up
        free(matrix);
        free(vector);
        free(result);
        cudaFree(d_matrix);
        cudaFree(d_vector);
        cudaFree(d_result);
    }

    fclose(inputFp);
    fclose(outputFp);

    return 0;
}
