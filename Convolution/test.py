import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

threshold = 1e-6

# read 2 images
img1 = Image.open("out2/image_0.jpg")
img2 = Image.open("output/test.jpg")

# convert the images to tensors
transform = transforms.Compose([transforms.ToTensor()])
img1 = transform(img1)
img2 = transform(img2)
# convert the images to 255 scale
img1 = img1 * 255
img2 = img2 * 255

# print the shape of the images
print(img1.shape)
print(img2.shape)

# compare the 2 images pixel by pixel and if the pixel values are different print the pixel values
max_diff = 0
for i in range(img1.shape[1]):
    for j in range(img1.shape[2]):
        if img1[0][i][j] != img2[0][i][j]:
            if abs(img1[0][i][j] - img2[0][i][j]) > max_diff:
                max_diff = abs(img1[0][i][j] - img2[0][i][j])
            # print(f"Pixel value at {i}, {j} is different: {img1[0][i][j]} and {img2[0][i][j]} with difference of {img1[0][i][j] - img2[0][i][j]}")

print(f"Max difference: {max_diff}")