import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# Function to apply mask to a batch of images
def apply_mask_to_images(input_folder, output_folder, mask_file, batch_size):
    # Load mask from file
    with open(mask_file, 'r') as f:
        mask_data = f.readlines()
    mask_size = int(mask_data[0].strip())
    mask_values = [[float(val) / (1) for val in row.split()] for row in mask_data[1:]]
    mask = torch.tensor(mask_values)
    print(mask)
    
    # Load images
    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]
    print("image_paths: {}".format(image_paths))
    # Process images in batches
    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size
    print("num_batches:",num_batches)
    transform = transforms.ToTensor()
    mask = mask.view(1, 1, mask_size, mask_size).repeat(1, 3, 1, 1)  # Repeat mask for each channel
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)
        print("start_idx:",start_idx)
        print("end_idx:",end_idx)
        batch_paths = image_paths[start_idx:end_idx]
        batch_images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in batch_paths])

        # Apply mask to images
        batch_size, channels, height, width = batch_images.shape
        output_images = F.conv2d(batch_images, mask, padding=mask_size//2)

        # Save output images
        os.makedirs(output_folder, exist_ok=True)
        for i, output_image in enumerate(output_images):
            output_path = os.path.join(output_folder, f"output_{start_idx + i}.jpg")
            print("Writing output image:",output_path)
            output_image = transforms.ToPILImage()(output_image)
            output_image.save(output_path)

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python final_python.py input output_python batchSize mask.txt")
    else:    
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        batch_size = int(sys.argv[3])
        mask_file = sys.argv[4]
        apply_mask_to_images(input_folder, output_folder, mask_file, batch_size)
