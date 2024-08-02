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
#define Input_TILE_SIZE 16 

__global__ void kernel_2(const unsigned char *input, unsigned char *output, const float *mask,int width, int height, int channels, int batch_size,int maskSize) {
    
    extern __shared__ float inputTile[];
    long long x = blockIdx.x * (blockDim.x - maskSize + 1) + threadIdx.x;
    long long y = blockIdx.y * (blockDim.y - maskSize + 1) + threadIdx.y;
    long long z = blockIdx.z * blockDim.z + threadIdx.z;
    long long res_idx = z * width * height + y * width + x;
    int radius = maskSize / 2;
    int input_tile_size_x = blockDim.x + 2 * radius - maskSize + 1;
    int input_tile_size_y = blockDim.y + 2 * radius - maskSize + 1;
    int input_tile_x = blockIdx.x * (blockDim.x - maskSize + 1) - radius;
    int input_tile_y = blockIdx.y * (blockDim.y - maskSize + 1) - radius;

    int global_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (global_thread_idx < input_tile_size_y * input_tile_size_x) {
        int i = global_thread_idx / input_tile_size_x;
        int j = global_thread_idx % input_tile_size_x;
        int input_x = input_tile_x + j;
        int input_y = input_tile_y + i;
        int idx = i * input_tile_size_x + j;

        if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
            for (int c = 0; c < channels; c++) {
                inputTile[idx * channels + c] =
                    (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + c];
            }
        } else {
            for (int c = 0; c < channels; c++) {
                inputTile[idx * channels + c] = 0;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < blockDim.x - maskSize + 1 && threadIdx.y < blockDim.y - maskSize + 1) {
        if (x < width && y < height && z < batch_size) {
            float sum = 0;
            for (int i = 0; i < maskSize; i++) {
                for (int j = 0; j < maskSize; j++) {
                    for (int c = 0; c < channels; c++) {
                        sum += mask[i * maskSize + j] *
                               inputTile[(threadIdx.y + i) * input_tile_size_x * channels + (threadIdx.x + j) * channels + c];
                    }
                }
            }
            output[res_idx] = (unsigned char)(sum);
        }
    }
    __syncthreads();
}


void one_batch(unsigned char *input, unsigned char *output, float *mask,int width, int height, int channels, int batch_size,int maskSize) {

dim3 block(Input_TILE_SIZE + maskSize - 1, Input_TILE_SIZE + maskSize - 1, 1);
dim3 grid((width + block.x - 1) / (Input_TILE_SIZE), (height + block.y - 1) / (Input_TILE_SIZE),(batch_size + block.z - 1) / block.z);

int shared_memory_size = (Input_TILE_SIZE + maskSize - 1) * (Input_TILE_SIZE + maskSize - 1) * channels * sizeof(float);
kernel_2 <<<grid, block, shared_memory_size>>>(input, output, mask, width, height, channels, batch_size,maskSize);

cudaDeviceSynchronize();
}

int readMask(float *&mask, const std::string &maskFilePath) {
    FILE *maskFile = freopen(maskFilePath.c_str(), "r", stdin);
    int maskSize;
    cin >> maskSize;
    mask = new float[maskSize * maskSize];
    for (int i = 0; i < maskSize; i++) {
        for (int j = 0; j < maskSize; j++) {
            cin >> mask[i * maskSize + j];
        }
    }
    fclose(maskFile);
    return maskSize;
}

std::vector<std::string> imagePathes(const std::string &inputFolderPath) {
    std::vector<std::string> imageFilePaths;
    for (const auto &entry: fs::directory_iterator(inputFolderPath)) {
        imageFilePaths.push_back(entry.path().string());
    }
    return imageFilePaths;
}

unsigned char *readImages(const std::string &imageFilePath, int &width, int &height, int &channels) {
    unsigned char *image = stbi_load(imageFilePath.c_str(), &width, &height, &channels, STBI_rgb);
    if (image == nullptr) {
        cerr << "Failed to load image: " << imageFilePath << endl;
        return nullptr;
    }
    return image;
}

bool checkDir(const std::string& path) {
    std::string command = "mkdir " + path;
    int result = system(command.c_str());
    return (result == 0);
}

int main(int argc, char *argv[]) {  
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " input_folder output_folder batch_size mask_file_path" << endl;
        return 1;
    }
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int batchSize = atoi(argv[3]);
    std::string maskPath = argv[4];

    float *mask;
    int maskSize = readMask(mask, maskPath);

    std::vector<std::string> imagePaths = imagePathes(inputPath);

    int firstWidth, firstHeight, firstChannels;
    unsigned char *firstImage = readImages(imagePaths[0], firstWidth, firstHeight, firstChannels);
    if (firstImage == nullptr) {
        return 1;
    }

    for (int len=0; len < (int)imagePaths.size(); len+=batchSize)
    {

        int loop = min(batchSize, (int)imagePaths.size()-len);
        unsigned char *outputImages = new unsigned char[firstWidth * firstHeight * loop];

        float *deviceMask;
        cudaMalloc(&deviceMask, maskSize * maskSize * sizeof(float));

        unsigned char *deviceImages;
        cudaMalloc(&deviceImages, firstWidth * firstHeight * firstChannels * loop * sizeof(unsigned char));

        unsigned char *deviceOutputImages;
        cudaMalloc(&deviceOutputImages, firstWidth * firstHeight * loop * sizeof(unsigned char));

        cudaMemcpy(deviceMask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);


        for (int i = len; i < loop+len ; i++) {
            int index=i-len;
            int width, height, channels;
            unsigned char *image = readImages(imagePaths[i], width, height, channels);
            cudaMemcpy(deviceImages + index * firstWidth * firstHeight * firstChannels, image,
                        firstWidth * firstHeight * firstChannels * sizeof(unsigned char),cudaMemcpyHostToDevice);
            stbi_image_free(image);
        }

        one_batch(deviceImages, deviceOutputImages, deviceMask, firstWidth, firstHeight, firstChannels, loop, maskSize);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        cudaDeviceSynchronize();

        cudaMemcpy(outputImages, deviceOutputImages, firstWidth * firstHeight * loop * sizeof(unsigned char),cudaMemcpyDeviceToHost);


        if (!checkDir(outputPath)) {
            std::cout << "Output folder exist " << outputPath << std::endl;
        }       

        for (int i = len; i < loop+len; i++) {
            int index = i - len;
            std::string outputImagePath = outputPath + "/image_" + std::to_string(i) + ".jpg";
            stbi_write_jpg(outputImagePath.c_str(), firstWidth, firstHeight, 1,outputImages + index * firstWidth * firstHeight, firstWidth);
        }

        cudaFree(deviceImages);
        cudaFree(deviceOutputImages);
        cudaFree(deviceMask);

        delete[] outputImages;
    }
    delete[] mask;
    stbi_image_free(firstImage);
    return 0;
}
