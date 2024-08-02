// Include necessary libraries
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "iostream"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <filesystem>
#include <vector>
#include <string>
namespace fs = std::filesystem;
using namespace std;
#define BLOCK_SIZE 16
#define O_TILE_SIZE 16 // Input tile size

__global__
void kernel_2(const unsigned char *input, unsigned char *output, const float *mask,
    int width, int height, int channels, int batch_size,
    int maskSize) {
    extern __shared__ float input_tile[];

    int x = blockIdx.x * (blockDim.x - maskSize + 1) + threadIdx.x;
    int y = blockIdx.y * (blockDim.y - maskSize + 1) + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int out_index = z * width * height + y * width + x;
    int r = maskSize / 2;
    int input_tile_size_x = blockDim.x + 2 * r - maskSize + 1;
    int input_tile_size_y = blockDim.y + 2 * r - maskSize + 1;
    int input_tile_start_x = blockIdx.x * (blockDim.x - maskSize + 1) - r;
    int input_tile_start_y = blockIdx.y * (blockDim.y - maskSize + 1) - r;
    if (threadIdx.x < input_tile_size_x && threadIdx.y < input_tile_size_y) {
        int input_x = input_tile_start_x + threadIdx.x;
        int input_y = input_tile_start_y + threadIdx.y;
        int input_tile_index = threadIdx.y * input_tile_size_x + threadIdx.x;
        if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
            
            input_tile[input_tile_index * channels ] = (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + 0] + 
                                                       (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + 1] + 
                                                       (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + 2];
            
        } else {       
            input_tile[input_tile_index * channels] = 0.0f;      
        }
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x - maskSize + 1 && threadIdx.y < blockDim.y - maskSize + 1) {
        if (x < width && y < height && z < batch_size) {
            float sum = 0;
            for (int i = 0; i < maskSize; i++) {
                for (int j = 0; j < maskSize; j++) {
                    sum += mask[i * maskSize + j] * input_tile[(threadIdx.y + i) * input_tile_size_x * channels + (threadIdx.x + j) * channels];  
                }
            }

            if (sum < 0) {
                sum = 0;
            } else if (sum > 255) {
                sum = 255;
            }
            output[out_index] = (unsigned char)(sum);
        }
    }
    __syncthreads();
}

void apply_kernel_2(unsigned char *input, unsigned char *output, float *mask,
           int width, int height, int channels, int batch_size,
           int maskSize) {
dim3 block(O_TILE_SIZE + maskSize - 1, O_TILE_SIZE + maskSize - 1, 1);
dim3 grid((width + block.x - 1) / (O_TILE_SIZE), (height + block.y - 1) / (O_TILE_SIZE),
    (batch_size + block.z - 1) / block.z);

int shared_memory_size =
  (O_TILE_SIZE + maskSize - 1) * (O_TILE_SIZE + maskSize - 1) * channels * sizeof(float);
kernel_2 <<< grid, block, shared_memory_size >>>(input, output, mask, width, height, channels, batch_size,
                                                maskSize);

cudaDeviceSynchronize();
}

int readMaskFromFile(float *&mask, const std::string &maskFilePath) {
    FILE *maskFile = fopen(maskFilePath.c_str(), "r");
    if (maskFile == nullptr) {
        cerr << "Failed to open mask file: " << maskFilePath << endl;
        return -1;
    }

    int size;
    fscanf(maskFile, "%d", &size);
    mask = new float[size * size];
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(maskFile, "%f", &mask[i * size + j]);
        }
    }
    
    fclose(maskFile);
    return size;
}

std::vector<std::string> getImageFilePaths(const std::string &inputFolderPath) {
    std::vector<std::string> imageFilePaths;
    for (const auto &entry: fs::directory_iterator(inputFolderPath)) {
        imageFilePaths.push_back(entry.path().string());
    }
    return imageFilePaths;
}

unsigned char *readImageFromFile(const std::string &imageFilePath, int &width, int &height, int &channels) {
    unsigned char *image = stbi_load(imageFilePath.c_str(), &width, &height, &channels, STBI_rgb);
    if (image == nullptr) {
        cerr << "Failed to load image: " << imageFilePath << endl;
        return nullptr;
    }
    return image;
}

bool createDirectory(const std::string& path) {
    std::string command = "mkdir " + path;
    int result = system(command.c_str());
    return (result == 0);
}

int main(int argc, char *argv[]) {
    cudaDeviceProp deviceProperties{};
    int deviceId = 0;
    cudaGetDeviceProperties(&deviceProperties, deviceId);

    if (argc != 5) {
        cout << "Usage: ./a.out input_folder_path output_folder_path batch_size mask.txt" << endl;
        return 1;
    }

    std::string inputFolderPath = argv[1];
    std::string outputFolderPath = argv[2];
    int batchSize = atoi(argv[3]);
    std::string maskFilePath = argv[4];

    float *mask;
    int maskSize = readMaskFromFile(mask, maskFilePath);

    float *deviceMask;
    cudaMalloc(&deviceMask, maskSize * maskSize * sizeof(float));

    cudaMemcpy(deviceMask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<std::string> imageFilePaths = getImageFilePaths(inputFolderPath);

    int numBatches = (imageFilePaths.size() + batchSize - 1) / batchSize;
    for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        int startIdx = batchIdx * batchSize;
        int endIdx = std::min((batchIdx + 1) * batchSize, static_cast<int>(imageFilePaths.size()));
        int imageWidth, imageHeight, channels;
        unsigned char *firstImage = readImageFromFile(imageFilePaths[startIdx], imageWidth, imageHeight, channels);
        if (firstImage == nullptr) {
            continue; 
        }

        unsigned char *outputImages = new unsigned char[imageWidth * imageHeight * (endIdx - startIdx)];

        unsigned char *deviceImages;
        cudaMalloc(&deviceImages, imageWidth * imageHeight * channels * (endIdx - startIdx) * sizeof(unsigned char));

        unsigned char *deviceOutputImages;
        cudaMalloc(&deviceOutputImages, imageWidth * imageHeight * (endIdx - startIdx) * sizeof(unsigned char));

        for (int i = startIdx; i < endIdx; i++) {
            unsigned char *image = readImageFromFile(imageFilePaths[i], imageWidth, imageHeight, channels);
            cudaMemcpy(deviceImages + (i - startIdx) * imageWidth * imageHeight * channels, image,
                       imageWidth * imageHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
            stbi_image_free(image);
        }

        apply_kernel_2(deviceImages, deviceOutputImages, deviceMask, imageWidth, imageHeight, channels,
                      endIdx - startIdx, maskSize);

        cudaMemcpy(outputImages, deviceOutputImages, imageWidth * imageHeight * (endIdx - startIdx) * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost);
        
        if (!createDirectory(outputFolderPath)) {
        }         
        for (int i = startIdx; i < endIdx; i++) {
            std::string outputImagePath = outputFolderPath + "/image_" + std::to_string(i) + ".jpg";
            stbi_write_jpg(outputImagePath.c_str(), imageWidth, imageHeight, 1,
                           outputImages + (i - startIdx) * imageWidth * imageHeight , imageWidth );
        }

        cudaFree(deviceImages);
        cudaFree(deviceOutputImages);

        delete[] outputImages;
        stbi_image_free(firstImage);
    }

    cudaFree(deviceMask);
    delete[] mask;

    return 0;
}

