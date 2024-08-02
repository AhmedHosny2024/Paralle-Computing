// Include necessary libraries
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "iostream"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <filesystem>
#include <vector>
#include <string>


// Namespace aliases
namespace fs = std::filesystem;
using namespace std;

// Define constants
#define BLOCK_SIZE 16
#define O_TILE_SIZE 16

///////////////////////////////////////////////////////////////////////////////////

// CUDA kernel to apply a mask to an image using tiling
__global__
void kernel_3(const unsigned char *input, unsigned char *output, const float *mask,
              int width, int height, int channels, int batch_size,
              int maskSize) {
    // Allocate shared memory for input tile
    extern __shared__ float input_tile[];

    // Get the pixel index
    long long x = blockIdx.x * blockDim.x + threadIdx.x;
    long long y = blockIdx.y * blockDim.y + threadIdx.y;
    long long z = blockIdx.z * blockDim.z + threadIdx.z;

    // Get output pixel index
    long long out_index = z * width * height + y * width + x;

    // Mask radius
    int r = maskSize / 2;

    // Define input tile size
    int input_tile_size_x = blockDim.x + 2 * r;
    int input_tile_size_y = blockDim.y + 2 * r;

    // Load the input tile
    int input_tile_start_x = blockIdx.x * blockDim.x - r;
    int input_tile_start_y = blockIdx.y * blockDim.y - r;

    // Load input tile into shared memory
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
                    input_tile[input_tile_index * channels ] = 0.0f;      
            }
        }
    }

    // Synchronize threads before processing input tile
    __syncthreads();

    // Check if the pixel is within the image
    if (x < width && y < height && z < batch_size) {
        // Calculate the output pixel value
        float sum = 0;
        for (int i = 0; i < maskSize; i++) {
            for (int j = 0; j < maskSize; j++) {
                int input_tile_x = threadIdx.x + j;
                int input_tile_y = threadIdx.y + i;
                int mask_index = i * maskSize + j;
                sum += mask[mask_index] * input_tile[(input_tile_y * input_tile_size_x + input_tile_x) * channels];
            }
        }

        if (sum < 0) {
            sum = 0;
        } else if (sum > 255) {
            sum = 255;
        }
        output[out_index] = (unsigned char)sum;
    }

    __syncthreads();
}



// Function to run the kernel applying mask using tiling
void apply_kernel_3(unsigned char *input, unsigned char *output, float *mask,
                    int width, int height, int channels, int batch_size,
                    int maskSize) {
    // Calculate the block size
    dim3 block(O_TILE_SIZE, O_TILE_SIZE, 1);
    // Calculate the grid size
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              (batch_size + block.z - 1) / block.z);

    int shared_memory_size =
            (O_TILE_SIZE + 2 * maskSize / 2) * (O_TILE_SIZE + 2 * maskSize / 2) * channels * sizeof(float);

    // Call the kernel
    kernel_3 <<< grid, block, shared_memory_size >>>(input, output, mask, width, height, channels, batch_size,
                                                     maskSize);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();
}


///////////////////////////////////////////////////////////////////////////////////

// Function to read mask from a file
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


// Function to get paths of images in a folder
std::vector<std::string> getImageFilePaths(const std::string &inputFolderPath) {
    std::vector<std::string> imageFilePaths;
    for (const auto &entry: fs::directory_iterator(inputFolderPath)) {
        imageFilePaths.push_back(entry.path().string());
    }
    return imageFilePaths;
}

// Function to read image from file and get its width and height
unsigned char *readImageFromFile(const std::string &imageFilePath, int &width, int &height, int &channels) {
    unsigned char *image = stbi_load(imageFilePath.c_str(), &width, &height, &channels, STBI_rgb);
    if (image == nullptr) {
        cerr << "Failed to load image: " << imageFilePath << endl;
        return nullptr;
    }
    return image;
}



// Function to create new output folder
bool createDirectory(const std::string& path) {
    std::string command = "mkdir " + path;
    int result = system(command.c_str());
    return (result == 0);
}


// Main function
int main(int argc, char *argv[]) {
    // Get device properties
    cudaDeviceProp deviceProperties{};
    int deviceId = 0;
    cudaGetDeviceProperties(&deviceProperties, deviceId);

    // Check if correct number of arguments is provided
    if (argc != 5) {
        cout << "Usage: ./a.out input_folder_path output_folder_path batch_size mask.txt" << endl;
        return 1;
    }

    // Read input arguments
    std::string inputFolderPath = argv[1];
    std::string outputFolderPath = argv[2];
    int batchSize = atoi(argv[3]);
    std::string maskFilePath = argv[4];

    // Read mask from file
    float *mask;
    int maskSize = readMaskFromFile(mask, maskFilePath);

    // Allocate memory for the mask on the device
    float *deviceMask;
    cudaMalloc(&deviceMask, maskSize * maskSize * sizeof(float));

    // Copy mask to the device
    cudaMemcpy(deviceMask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);

    // Get paths of input images
    std::vector<std::string> imageFilePaths = getImageFilePaths(inputFolderPath);

    // Determine the number of batches needed
    int numBatches = (imageFilePaths.size() + batchSize - 1) / batchSize;
    // printf("numBatches: %d\n",numBatches);
    // Loop over batches
    for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        // Determine the range of images for this batch
        int startIdx = batchIdx * batchSize;
        int endIdx = std::min((batchIdx + 1) * batchSize, static_cast<int>(imageFilePaths.size()));
        // printf("startIdx: %d\n",startIdx);
        // printf("endIdx: %d\n",endIdx);
        // Read the width and height of the first image in this batch
        int imageWidth, imageHeight, channels;
        unsigned char *firstImage = readImageFromFile(imageFilePaths[startIdx], imageWidth, imageHeight, channels);
        if (firstImage == nullptr) {
            continue; // Skip to next batch if unable to read the first image
        }

        // Allocate memory for the output images on the host
        unsigned char *outputImages = new unsigned char[imageWidth * imageHeight * (endIdx - startIdx)];

        // Allocate memory for the input images on the device
        unsigned char *deviceImages;
        cudaMalloc(&deviceImages, imageWidth * imageHeight * channels * (endIdx - startIdx) * sizeof(unsigned char));

        // Allocate memory for the output images on the device
        unsigned char *deviceOutputImages;
        cudaMalloc(&deviceOutputImages, imageWidth * imageHeight * (endIdx - startIdx) * sizeof(unsigned char));

        // Read and copy images to the device
        for (int i = startIdx; i < endIdx; i++) {
            unsigned char *image = readImageFromFile(imageFilePaths[i], imageWidth, imageHeight, channels);
            cudaMemcpy(deviceImages + (i - startIdx) * imageWidth * imageHeight * channels, image,
                       imageWidth * imageHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
            stbi_image_free(image);
        }

        // Apply mask to images
        apply_kernel_3(deviceImages, deviceOutputImages, deviceMask, imageWidth, imageHeight, channels,
                      endIdx - startIdx, maskSize);

        // Copy output images to host
        cudaMemcpy(outputImages, deviceOutputImages, imageWidth * imageHeight * (endIdx - startIdx) * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost);
        
        // printf("Saving...\n");
        if (!createDirectory(outputFolderPath)) {
            // std::cout << "Output folder exist " << outputFolderPath << std::endl;
        }         
        // Save output images
        for (int i = startIdx; i < endIdx; i++) {
            std::string outputImagePath = outputFolderPath + "/image_" + std::to_string(i) + ".jpg";
            stbi_write_jpg(outputImagePath.c_str(), imageWidth, imageHeight, 1,
                           outputImages + (i - startIdx) * imageWidth * imageHeight , imageWidth );
            // std::cout << "Writing " << outputImagePath << std::endl;
        }

        // Free device memory
        cudaFree(deviceImages);
        cudaFree(deviceOutputImages);

        // Free host memory
        delete[] outputImages;
        stbi_image_free(firstImage);
    }

    cudaFree(deviceMask);
    // Free mask memory
    delete[] mask;

    // std::cout << "Done" << std::endl;
    return 0;
}

