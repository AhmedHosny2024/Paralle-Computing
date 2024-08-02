#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cuda_runtime.h>
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <string>


__global__ void convolution2D(float *input, float *output, int filterSize, float* filterValues, int width, int height, int depth, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z; // batch index
    if (x < width && y < height && b < batchSize) {
        float val = 0.0f;
        for (int d = 0; d < depth; d++) { // Iterate over each channel
            float channelVal = 0.0f;
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    int curX = min(max(x + j, 0), width - 1);
                    int curY = min(max(y + i, 0), height - 1);
                    int inputIndex = (b * depth * width * height) + (d * width * height) + (curY * width + curX);
                    int filterIndex = i * filterSize + j;
                    printf("inputIndex: %d, filterIndex: %d with value %d, filter value %d \n", inputIndex, filterIndex, input[inputIndex], filterValues[filterIndex]);
                    channelVal += input[inputIndex] * filterValues[filterIndex];
                }
            }
            val += channelVal / depth; // Average the results over the channels
        }
        int outputIndex = (b * width * height) + (y * width + x); // Output is a grayscale image, so no need for depth in the index
        output[outputIndex] = val;
    }
}



void reshapeImage(unsigned char *inputImage, unsigned char *reshapedImage, int width, int height, int channels) {
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = (y * width + x) * channels + c;
                int reshapedIndex = c * width * height + y * width + x;
                reshapedImage[reshapedIndex] = inputImage[index];
            }
        }
    }
}

void convertToFloat(unsigned char *inputImage, float *floatImage, int size) {
    for (int i = 0; i < size; ++i) {
        floatImage[i] = static_cast<float>(inputImage[i]) / 255.0f;
    }
}
void convertToChar(float *inputImage, unsigned char *charImage, int size) {
    for (int i = 0; i < size; ++i) {
        charImage[i] = static_cast<unsigned char>(inputImage[i] );
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <input_folder> <output_folder> <batch_size> <filter_path>\n";
        return 1;
    }

    const std::string inputFolder = argv[1];
    const std::string outputFolder = argv[2];
    int batchSize = std::stoi(argv[3]);
    const std::string filterPath = argv[4];
    // Read filter from file
    std::ifstream filterFile(filterPath);
    if (!filterFile.is_open()) {
        std::cerr << "Error opening filter file: " << filterPath << std::endl;
        return 1;
    }
    // Read filter size as the first number in the file and the other numbers as the filter values
    int filterSize;
    filterFile >> filterSize;
    float *filterValues = new float[filterSize*filterSize];
    for (int i = 0; i < filterSize; i++) {
        filterFile >> filterValues[i];
    }

    filterFile.close();

    // Repeat the single row of filter values to fill the 3x3 filter
    for (int i = 1; i < filterSize; i++) {
        memcpy(filterValues + i * filterSize, filterValues, filterSize * sizeof(float));
    }

    // Read input directory using C++17 filesystem
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(inputFolder)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }

    float *d_filterValues;
    cudaMalloc((void **)&d_filterValues, filterSize * filterSize * sizeof(float));
    cudaMemcpy(d_filterValues, filterValues, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice);


    int width, height, channels;
    unsigned char *inputImage;
    int imageSize;
    char inputPath[256];
    char outputPath[256];
    int batchCount = 0;
    int imageCount = 0;
    float *batchInput;
    float *batchOutput;
    int totalSize = 0;

    for (const auto &filePath : files) {
        strcpy(inputPath, filePath.c_str());
        inputImage = stbi_load(inputPath, &width, &height, &channels, 0);
        if (inputImage == nullptr) {
            std::cerr << "Error loading image: " << inputPath << std::endl;
            continue;
        }

        unsigned char *reshapedImage = new unsigned char[channels * width * height];
        reshapeImage(inputImage, reshapedImage, width, height, channels);

        // Convert the reshaped image data to float
        float *floatImage = new float[channels * width * height];
        convertToFloat(reshapedImage, floatImage, channels * width * height);


        imageSize = width * height * channels * sizeof(unsigned char);
        totalSize += imageSize;

        if (files.size() < batchSize) {
            batchSize = files.size();
        }
        
        if (imageCount % batchSize == 0) {
            // Allocate memory for batch on GPU
            cudaMalloc((void **)&batchInput, totalSize);
            cudaMalloc((void **)&batchOutput, totalSize);
        }

        // Copy input image to batch memory
        cudaMemcpy(batchInput + (imageCount % batchSize) * imageSize, floatImage, imageSize, cudaMemcpyHostToDevice);

        imageCount++;

        if (imageCount % batchSize == 0 || imageCount == files.size()) {
            dim3 threadsPerBlock(16, 16, 1);
            dim3 numBlocks(
                       (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       batchSize);  // Add a block for each batch

            // Launch kernel
            convolution2D<<<numBlocks, threadsPerBlock>>>(batchInput, batchOutput, filterSize, d_filterValues, width, height, channels, batchSize);
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }


            unsigned char * charOutputImage = new unsigned char[totalSize];
            float *outputImageFloat = new float[totalSize / sizeof(float)];
            cudaMemcpy(outputImageFloat, batchOutput, totalSize, cudaMemcpyDeviceToHost);
            // print the output image
            // for (int i = 0; i < totalSize / sizeof(float); i++) {
            //     printf("%f ", outputImageFloat[i]);
            // }
            convertToChar(outputImageFloat, charOutputImage, totalSize / sizeof(float));
            // Copy result back to host and save output images
            unsigned char *outputImage = (unsigned char *)malloc(totalSize);
            cudaMemcpy(outputImage, charOutputImage, totalSize, cudaMemcpyDeviceToHost);

            for (int i = 0; i < batchSize; i++) {
                
                sprintf(outputPath, "%s/outputImage%d.jpg", outputFolder.c_str(), batchCount * batchSize + i);
                stbi_write_jpg(outputPath, width, height, 1, outputImage + i * imageSize, 100);
            }

            // Free GPU memory
            cudaFree(batchInput);
            cudaFree(batchOutput);
            free(outputImage);

            batchCount++;
            totalSize = 0;
        }

    }

    return 0;
}