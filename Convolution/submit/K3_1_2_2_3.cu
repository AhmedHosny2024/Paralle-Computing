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
#define OUTPUT_TILE_SIZE 16
__global__
void kernel_3(const unsigned char *input, unsigned char *output, const float *mask,
              int width, int height, int channels, int batch_size,
              int maskSize) {
    extern __shared__ float input_tile[];
    long long x = blockIdx.x * blockDim.x + threadIdx.x;
    long long y = blockIdx.y * blockDim.y + threadIdx.y;
    long long z = blockIdx.z * blockDim.z + threadIdx.z;
    long long out_index = z * width * height + y * width + x;
    int r = maskSize / 2;
    int input_tile_size_x = blockDim.x + 2 * r;
    int input_tile_size_y = blockDim.y + 2 * r;
    int input_tile_start_x = blockIdx.x * blockDim.x - r;
    int input_tile_start_y = blockIdx.y * blockDim.y - r;
    for (int i = threadIdx.y; i < input_tile_size_y; i += blockDim.y) {
        for (int j = threadIdx.x; j < input_tile_size_x; j += blockDim.x) {
            int input_x = input_tile_start_x + j;
            int input_y = input_tile_start_y + i;
            int input_tile_index = i * input_tile_size_x + j;

            if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
                
                    input_tile[input_tile_index * channels] =
                        (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + 0] +
                        (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + 1] +
                        (float)input[z * width * height * channels + input_y * width * channels + input_x * channels + 2];
                
            } else {
                for (int c = 0; c < channels; c++) {
                    input_tile[input_tile_index * channels + c] = 0;
                }
            }
        }
    }
    __syncthreads();
    if (x < width && y < height && z < batch_size) {
        float sum = 0;
        for (int i = 0; i < maskSize; i++) {
            for (int j = 0; j < maskSize; j++) {
                int input_tile_x = threadIdx.x + j;
                int input_tile_y = threadIdx.y + i;
                int mask_index = i * maskSize + j;
                    sum += mask[mask_index] * input_tile[(input_tile_y * input_tile_size_x + input_tile_x) * channels];
            }
        }
        output[out_index] = (unsigned char)sum;
    }
    __syncthreads();
}



void one_batch(unsigned char *input, unsigned char *output, float *mask,
                       int width, int height, int channels, int batch_size,
                       int maskSize) {
    dim3 block(OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              (batch_size + block.z - 1) / block.z);

    int shared_memory_size =
            (OUTPUT_TILE_SIZE + 2 * maskSize / 2) * (OUTPUT_TILE_SIZE + 2 * maskSize / 2) * channels * sizeof(float);

    kernel_3 <<< grid, block, shared_memory_size >>>(input, output, mask, width, height, channels, batch_size,
                                                          maskSize);

    cudaDeviceSynchronize();
}


int readMask(float *&mask, const std::string &maskFilePath) {
    FILE *maskFile = freopen(maskFilePath.c_str(), "r", stdin);
    int size;
    cin >> size;
    mask = new float[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cin >> mask[i * size + j];
        }
    }
    fclose(maskFile);
    return size;
}

std::vector<std::string> getImage(const std::string &inputFolderPath) {
    std::vector<std::string> imageFilePaths;
    for (const auto &entry: fs::directory_iterator(inputFolderPath)) {
        imageFilePaths.push_back(entry.path().string());
    }
    return imageFilePaths;
}

unsigned char *readImage(const std::string &imageFilePath, int &width, int &height, int &channels) {
    unsigned char *image = stbi_load(imageFilePath.c_str(), &width, &height, &channels, STBI_rgb);
    if (image == nullptr) {
        cerr << "Failed to load image: " << imageFilePath << endl;
        return nullptr;
    }
    return image;
}



bool createDir(const std::string& path) {
    std::string command = "mkdir " + path;
    int result = system(command.c_str());
    return (result == 0);
}


int main(int argc, char *argv[]) {
  
    if (argc != 5) {
        cout << "Invalid Arguments, waiting for : input_folder_path output_output_path batch_size mask_path" << endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int batchSize = atoi(argv[3]);
    std::string maskPath = argv[4];

    float *mask;
    int maskSize = readMask(mask, maskPath);

    std::vector<std::string> imagePaths = getImage(inputPath);

    int firstWidth, firstHeight, firstChannels;
    unsigned char *firstImage = readImage(imagePaths[0], firstWidth, firstHeight, firstChannels);
    if (firstImage == nullptr) {
        return 1; 
    }
    for (int len = 0 ; len <  (int) imagePaths.size();len+=batchSize){
        int loop = min(batchSize, (int) imagePaths.size()-len);
        unsigned char *outputImages = new unsigned char[firstWidth * firstHeight * loop];

        float *deviceMask;
        cudaMalloc(&deviceMask, maskSize * maskSize * sizeof(float));

        unsigned char *deviceImages;
        cudaMalloc(&deviceImages, firstWidth * firstHeight * firstChannels * loop * sizeof(unsigned char));

        unsigned char *deviceOutputImages;
        cudaMalloc(&deviceOutputImages, firstWidth * firstHeight * loop * sizeof(unsigned char));

        cudaMemcpy(deviceMask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < loop; i++) {
            int width, height, channels;
            unsigned char *image = readImage(imagePaths[i+len], width, height, channels);
            cudaMemcpy(deviceImages + i * firstWidth * firstHeight * firstChannels, image,
                    firstWidth * firstHeight * firstChannels * sizeof(unsigned char),
                    cudaMemcpyHostToDevice);
            stbi_image_free(image);
        }

        one_batch(deviceImages, deviceOutputImages, deviceMask, firstWidth, firstHeight, firstChannels, loop, maskSize);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        cudaDeviceSynchronize();

        cudaMemcpy(outputImages, deviceOutputImages, firstWidth * firstHeight * loop * sizeof(unsigned char),
                cudaMemcpyDeviceToHost);

        if (!createDir(outputPath)) {
            std::cout << "Output folder exist " << outputPath << std::endl;
        }       
        

        for (int i = 0; i < loop; i++) {
            std::string outputImagePath = outputPath + "/image_" + std::to_string(i+len) + ".jpg";
            stbi_write_jpg(outputImagePath.c_str(), firstWidth, firstHeight, 1,
                        outputImages + i * firstWidth * firstHeight, firstWidth);
                        std::cout<<"writing " << outputImagePath.c_str()<<std::endl;
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
