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

__global__
void kernel_1(const unsigned char *inputImage, unsigned char *outputImage, const float *mask,int imageWidth, int imageHeight, int channels, int batchSize, int maskSize) {
    long long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    long long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    long long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

    long long outputIndex = zIndex * imageWidth * imageHeight + yIndex * imageWidth + xIndex;
    int maskRadius = maskSize / 2;
    
    if (xIndex < imageWidth && yIndex < imageHeight && zIndex < batchSize) {
        float sum = 0;
        for (int i = 0; i < maskSize; i++) {
            for (int j = 0; j < maskSize; j++) {
                for (int c = 0; c < channels; c++) {
                    int inputX = xIndex + j - maskRadius;
                    int inputY = yIndex + i - maskRadius;
                    if (inputX >= 0 && inputX < imageWidth && inputY >= 0 && inputY < imageHeight) {
                        sum += mask[i * maskSize + j] *
                               (float) inputImage[zIndex * imageWidth * imageHeight * channels +
                                                 inputY * imageWidth * channels + inputX * channels + c];
                    }
                }
            }
        }
        outputImage[outputIndex] = (unsigned char) (sum);
    }
}

void one_batch(unsigned char *inputImages, unsigned char *outputImages, float *mask,
                       int imageWidth, int imageHeight, int channels, int batchSize,
                       int maskSize) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((imageWidth + block.x - 1) / block.x, (imageHeight + block.y - 1) / block.y,
              (batchSize + block.z - 1) / block.z);

    kernel_1<<<grid, block>>>(inputImages, outputImages, mask, imageWidth, imageHeight, channels,
                                batchSize, maskSize);

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
    cudaDeviceProp deviceProperties{};
    int deviceId = 0;
    cudaGetDeviceProperties(&deviceProperties, deviceId);

    if (argc != 5) {
        cout << "Usage: ./a.out input_folder_path output_folder_path batch_size mask.txt" << endl;
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

    for (int len=0;len<  (int) imagePaths.size();len+=batchSize){

        int loop = min(batchSize, (int)imagePaths.size()-len);
        unsigned char *outputImages = new unsigned char[firstWidth * firstHeight * loop];

        float *deviceMask;
        cudaMalloc(&deviceMask, maskSize * maskSize * sizeof(float));

        unsigned char *deviceImages;
        
        cudaMalloc(&deviceImages, firstWidth * firstHeight * firstChannels * loop * sizeof(unsigned char));

        unsigned char *deviceOutputImages;
        cudaMalloc(&deviceOutputImages, firstWidth * firstHeight * loop * sizeof(unsigned char));

        cudaMemcpy(deviceMask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = len; i < loop + len; i++) {
            int index=i-len;
            int width, height, channels;
            unsigned char *image = readImage(imagePaths[i], width, height, channels);
            cudaMemcpy(deviceImages + index * firstWidth * firstHeight * firstChannels, image,
                    firstWidth * firstHeight * firstChannels * sizeof(unsigned char),
                    cudaMemcpyHostToDevice);
            stbi_image_free(image);
        }
        
        one_batch(deviceImages, deviceOutputImages, deviceMask, firstWidth, firstHeight,
                        firstChannels, loop, maskSize);

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

        for (int i = len; i < loop+len; i++) {
            int index=i-len;
            std::string outputImagePath = outputPath + "/image_" + std::to_string(i) + ".jpg";
            stbi_write_jpg(outputImagePath.c_str(), firstWidth, firstHeight, 1,
                        outputImages + index * firstWidth * firstHeight, firstWidth);
                        //std::cout<<"writing " << outputImagePath.c_str()<<std::endl;
        }

        // Free device memory
        cudaFree(deviceImages);
        cudaFree(deviceOutputImages);
        cudaFree(deviceMask);

        // Free host memory
        delete[] outputImages;
    }
    delete[] mask;
    stbi_image_free(firstImage);
    //std::cout << "done" << std::endl;
    return 0;
}
